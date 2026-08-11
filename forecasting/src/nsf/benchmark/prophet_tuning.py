"""Small grid search for Prophet hyperparameters."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nsf.benchmark.deterministic import BenchmarkDataConfig, _metric_rows, _prepare_panel, _series_map, _validate_dense
from nsf.benchmark.lstm_tuning import _seasonal_scale, _slice_series
from nsf.benchmark.prophet_benchmark import ProphetModelConfig, _fit_predict_prophet, _seasonality_value
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.splitting.panel_folds import make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class ProphetTuningConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    objective: str
    slices: tuple[str, ...]
    max_series_per_slice: int | None
    train_tail: int | None
    grid: dict[str, list[Any]]
    base_model: ProphetModelConfig
    output_dir: str


def load_prophet_tuning_config(path: str | Path) -> ProphetTuningConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tuning = raw.get("tuning", {})
    training = raw.get("training", {})
    model = raw.get("base_model", {})
    max_series_per_slice = training.get("max_series_per_slice")
    train_tail = training.get("train_tail")
    return ProphetTuningConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        objective=str(tuning.get("objective", "MASE")).upper(),
        slices=tuple(str(value) for value in training.get("slices", [])),
        max_series_per_slice=None if max_series_per_slice in (None, 0) else int(max_series_per_slice),
        train_tail=None if train_tail in (None, 0) else int(train_tail),
        grid={str(key): list(values) for key, values in tuning["grid"].items()},
        base_model=ProphetModelConfig(
            daily_seasonality=_seasonality_value(model.get("daily_seasonality", True)),
            weekly_seasonality=_seasonality_value(model.get("weekly_seasonality", True)),
            yearly_seasonality=_seasonality_value(model.get("yearly_seasonality", False)),
            seasonality_mode=str(model.get("seasonality_mode", "additive")),
            changepoint_prior_scale=float(model.get("changepoint_prior_scale", 0.05)),
            seasonality_prior_scale=float(model.get("seasonality_prior_scale", 10.0)),
            interval_width=float(model.get("interval_width", 0.8)),
            log_transform=bool(model.get("log_transform", True)),
        ),
        output_dir=str(raw.get("output", {}).get("output_dir", "forecasting/experiments/runs/prophet_tuning")),
    )


def _model_with_params(base: ProphetModelConfig, params: dict[str, Any]) -> ProphetModelConfig:
    return ProphetModelConfig(
        daily_seasonality=_seasonality_value(params.get("daily_seasonality", base.daily_seasonality)),
        weekly_seasonality=_seasonality_value(params.get("weekly_seasonality", base.weekly_seasonality)),
        yearly_seasonality=_seasonality_value(params.get("yearly_seasonality", base.yearly_seasonality)),
        seasonality_mode=str(params.get("seasonality_mode", base.seasonality_mode)),
        changepoint_prior_scale=float(params.get("changepoint_prior_scale", base.changepoint_prior_scale)),
        seasonality_prior_scale=float(params.get("seasonality_prior_scale", base.seasonality_prior_scale)),
        interval_width=float(params.get("interval_width", base.interval_width)),
        log_transform=bool(params.get("log_transform", base.log_transform)),
    )


def _grid_rows(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[key] for key in keys))]


def tune_prophet(config: ProphetTuningConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
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
    slice_names = sorted(panel["slice"].unique())
    if config.slices:
        requested = set(config.slices)
        slice_names = [slice_name for slice_name in slice_names if slice_name in requested]
        missing = sorted(requested - set(slice_names))
        if missing:
            raise ValueError(f"Unknown slices requested: {missing}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_rows = []
    best_score = float("inf")
    best_params: dict[str, Any] = {}

    for trial_idx, params in enumerate(_grid_rows(config.grid)):
        model_config = _model_with_params(config.base_model, params)
        prediction_rows = []
        scales = {}
        start = time.perf_counter()
        trained_models = 0
        for fold in folds:
            for slice_name in slice_names:
                series_list = _slice_series(series_map, str(slice_name))
                if config.max_series_per_slice is not None:
                    series_list = series_list[: config.max_series_per_slice]
                scales[(fold.fold, str(slice_name))] = _seasonal_scale(series_list, fold.train_end_idx, period=144)
                for sub in series_list:
                    train = sub.iloc[fold.train_start_idx : fold.train_end_idx + 1]
                    if config.train_tail is not None:
                        train = train.tail(config.train_tail)
                    target = sub.iloc[fold.target_start_idx : fold.target_end_idx + 1]
                    pred, _train_seconds, _inference_seconds = _fit_predict_prophet(train, target["ds"], model_config)
                    trained_models += 1
                    for step_idx in range(config.backtest.horizon):
                        prediction_rows.append(
                            {
                                "fold": fold.fold,
                                "model": "prophet",
                                "unique_id": str(sub["unique_id"].iloc[0]),
                                "slice": str(slice_name),
                                "horizon": step_idx + 1,
                                "y_true": float(target["y"].iloc[step_idx]),
                                "y_pred": float(pred[step_idx]),
                            }
                        )
        metrics = _metric_rows(pd.DataFrame(prediction_rows), scales)
        score = float(metrics[config.objective].mean())
        elapsed = time.perf_counter() - start
        row = {
            "trial": trial_idx,
            "objective": config.objective,
            "score": score,
            "seconds": elapsed,
            "trained_models": trained_models,
        }
        row.update(params)
        trial_rows.append(row)
        if score < best_score:
            best_score = score
            best_params = dict(params)

    paths = {
        "trials": output_dir / "prophet_tuning_trials.csv",
        "best_params": output_dir / "prophet_best_params.yaml",
        "run_meta": output_dir / "run_meta.json",
        "resolved_config": output_dir / "resolved_config.yaml",
    }
    pd.DataFrame(trial_rows).to_csv(ensure_parent(paths["trials"]), index=False)
    paths["best_params"].write_text(yaml.safe_dump({"best_params": best_params}, sort_keys=False), encoding="utf-8")
    paths["run_meta"].write_text(
        json.dumps(
            {
                "trials": len(trial_rows),
                "best_score": best_score,
                "objective": config.objective,
                "folds": config.backtest.n_folds,
                "slices": list(slice_names),
                "max_series_per_slice": config.max_series_per_slice,
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
                "base_model": config.base_model.__dict__,
                "training": {
                    "slices": list(config.slices),
                    "max_series_per_slice": config.max_series_per_slice,
                    "train_tail": config.train_tail,
                },
                "tuning": {"objective": config.objective, "grid": config.grid},
                "output": {"output_dir": config.output_dir},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return paths
