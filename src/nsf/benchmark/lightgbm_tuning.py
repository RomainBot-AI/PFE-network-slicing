"""LightGBM random-search tuning by slice."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nsf.benchmark.deterministic import (
    BenchmarkDataConfig,
    BenchmarkFeatureConfig,
    _make_lgbm_train_frame,
    _prepare_panel,
    _seasonal_scales,
    _series_map,
    _validate_dense,
)
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.evaluation.deterministic import mase, rmse
from nsf.splitting.panel_folds import make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class LightGBMTuningConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    features: BenchmarkFeatureConfig
    n_trials: int
    objective: str
    validation_fraction: float
    output_dir: str
    base_params: dict[str, Any]


def load_lightgbm_tuning_config(path: str | Path) -> LightGBMTuningConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return LightGBMTuningConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        features=BenchmarkFeatureConfig(
            lags=tuple(int(value) for value in raw["features"]["lags"]),
            train_origin_stride=int(raw["features"].get("train_origin_stride", 36)),
            seasonal_scale_period=int(raw["features"].get("seasonal_scale_period", 144)),
        ),
        n_trials=int(raw.get("tuning", {}).get("n_trials", 25)),
        objective=str(raw.get("tuning", {}).get("objective", "MASE")),
        validation_fraction=float(raw.get("tuning", {}).get("validation_fraction", 0.2)),
        output_dir=str(raw.get("output", {}).get("output_dir", "experiments/runs/lightgbm_tuning")),
        base_params=dict(raw.get("base_params", {})),
    )


def _sample_params(rng: random.Random) -> dict[str, Any]:
    return {
        "n_estimators": rng.choice([100, 150, 200, 300, 400, 500]),
        "learning_rate": rng.uniform(0.02, 0.12),
        "num_leaves": rng.choice([15, 31, 63, 127]),
        "min_child_samples": rng.choice([10, 20, 50, 100]),
        "subsample": rng.uniform(0.7, 1.0),
        "colsample_bytree": rng.uniform(0.7, 1.0),
        "reg_alpha": rng.choice([0.0, 0.01, 0.1, 1.0]),
        "reg_lambda": rng.choice([0.0, 0.01, 0.1, 1.0]),
    }


def _slice_series(series_map: dict[str, pd.DataFrame], slice_name: str) -> dict[str, pd.DataFrame]:
    return {uid: sub for uid, sub in series_map.items() if str(sub["slice"].iloc[0]) == slice_name}


def _chronological_train_val(x: pd.DataFrame, y: pd.Series, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Rows are generated in chronological order per series. Sort by origin-like lag features is not possible here,
    # so keep the deterministic generation order and use the last block as validation.
    val_size = max(1, int(round(len(x) * validation_fraction)))
    train_size = len(x) - val_size
    if train_size <= 0:
        raise ValueError("Validation fraction leaves no training rows")
    return x.iloc[:train_size], x.iloc[train_size:], y.iloc[:train_size], y.iloc[train_size:]


def tune_lightgbm_by_slice(config: LightGBMTuningConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise RuntimeError("LightGBM tuning requires lightgbm. Install with .venv/bin/python -m pip install lightgbm") from exc

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    panel = _prepare_panel(read_panel(config.data.panel_csv))
    timestamps = pd.DatetimeIndex(sorted(panel["ds"].unique()))
    series_map = _series_map(panel)
    _validate_dense(series_map, timestamps)
    folds = make_panel_folds(
        timestamps=timestamps,
        input_size=config.backtest.input_size,
        horizon=config.backtest.horizon,
        n_folds=config.backtest.n_folds,
        fold_stride=config.backtest.fold_stride,
        expanding=config.backtest.expanding,
    )
    # Tune on the first fold only to keep the search budget bounded, then evaluate chosen params on all folds in benchmark.
    fold = folds[0]
    scales = _seasonal_scales(series_map, fold, config.features.seasonal_scale_period)
    trials = []
    best_params_by_slice = {}

    for slice_name in sorted(panel["slice"].unique()):
        scoped = _slice_series(series_map, str(slice_name))
        best_score = float("inf")
        best_params = None
        for trial_idx in range(config.n_trials):
            params = _sample_params(rng)
            params.update(config.base_params)
            horizon_scores = []
            start = time.perf_counter()
            for horizon in range(1, config.backtest.horizon + 1):
                x, y = _make_lgbm_train_frame(
                    series_map=scoped,
                    fold=fold,
                    lags=config.features.lags,
                    horizon=horizon,
                    stride=config.features.train_origin_stride,
                )
                x_train, x_val, y_train, y_val = _chronological_train_val(x, y, config.validation_fraction)
                model_params = {
                    "random_state": config.seed + trial_idx,
                    "n_jobs": 1,
                    "verbosity": -1,
                }
                model_params.update(params)
                model = LGBMRegressor(**model_params)
                model.fit(x_train, y_train)
                pred = np.maximum(0.0, np.expm1(model.predict(x_val)))
                true = np.maximum(0.0, np.expm1(y_val.to_numpy(dtype=float)))
                if config.objective.upper() == "RMSE":
                    horizon_scores.append(rmse(true, pred))
                else:
                    horizon_scores.append(mase(true, pred, scales.get(str(slice_name), 1.0)))
            elapsed = time.perf_counter() - start
            score = float(np.mean(horizon_scores))
            row = {
                "slice": str(slice_name),
                "trial": trial_idx,
                "objective": config.objective.upper(),
                "score": score,
                "seconds": elapsed,
            }
            row.update(params)
            trials.append(row)
            if score < best_score:
                best_score = score
                best_params = params
        best_params_by_slice[str(slice_name)] = best_params or {}

    paths = {
        "trials": output_dir / "lightgbm_tuning_trials.csv",
        "best_params": output_dir / "lightgbm_best_params_by_slice.yaml",
        "run_meta": output_dir / "run_meta.json",
        "resolved_config": output_dir / "resolved_config.yaml",
    }
    pd.DataFrame(trials).to_csv(ensure_parent(paths["trials"]), index=False)
    paths["best_params"].write_text(yaml.safe_dump({"best_params_by_slice": best_params_by_slice}, sort_keys=False), encoding="utf-8")
    paths["run_meta"].write_text(
        json.dumps(
            {
                "slices": len(best_params_by_slice),
                "n_trials": config.n_trials,
                "objective": config.objective.upper(),
                "tuned_fold": fold.fold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["resolved_config"].write_text(
        yaml.safe_dump(
            {
                "seed": config.seed,
                "data": config.data.__dict__,
                "backtest": config.backtest.__dict__,
                "features": {
                    "lags": list(config.features.lags),
                    "train_origin_stride": config.features.train_origin_stride,
                    "seasonal_scale_period": config.features.seasonal_scale_period,
                },
                "tuning": {
                    "n_trials": config.n_trials,
                    "objective": config.objective,
                    "validation_fraction": config.validation_fraction,
                },
                "base_params": config.base_params,
                "output": {"output_dir": config.output_dir},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return paths
