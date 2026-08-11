"""Small explicit candidate search for PatchTST."""

from __future__ import annotations

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
from nsf.benchmark.patchtst_benchmark import (
    PatchTSTBenchmarkConfig,
    PatchTSTModelConfig,
    PatchTSTTrainingConfig,
    _apply_params,
    _fit_predict_patchtst,
    _training_frame,
)
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.splitting.panel_folds import make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class PatchTSTTuningConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    objective: str
    slices: tuple[str, ...]
    base_model: PatchTSTModelConfig
    candidates: tuple[dict[str, Any], ...]
    device: str
    output_dir: str


def _model_config(raw: dict[str, Any]) -> PatchTSTModelConfig:
    return PatchTSTModelConfig(
        max_steps=int(raw.get("max_steps", 100)),
        learning_rate=float(raw.get("learning_rate", 5e-4)),
        batch_size=int(raw.get("batch_size", 32)),
        windows_batch_size=int(raw.get("windows_batch_size", 512)),
        inference_windows_batch_size=int(raw.get("inference_windows_batch_size", 512)),
        step_size=int(raw.get("step_size", 288)),
        scaler_type=str(raw.get("scaler_type", "standard")),
        encoder_layers=int(raw.get("encoder_layers", 2)),
        n_heads=int(raw.get("n_heads", 4)),
        hidden_size=int(raw.get("hidden_size", 64)),
        linear_hidden_size=int(raw.get("linear_hidden_size", 128)),
        dropout=float(raw.get("dropout", 0.2)),
        fc_dropout=float(raw.get("fc_dropout", 0.2)),
        head_dropout=float(raw.get("head_dropout", 0.0)),
        attn_dropout=float(raw.get("attn_dropout", 0.0)),
        patch_len=int(raw.get("patch_len", 32)),
        stride=int(raw.get("stride", 16)),
        revin=bool(raw.get("revin", True)),
        revin_affine=bool(raw.get("revin_affine", False)),
        revin_subtract_last=bool(raw.get("revin_subtract_last", True)),
        activation=str(raw.get("activation", "gelu")),
        res_attention=bool(raw.get("res_attention", True)),
        batch_normalization=bool(raw.get("batch_normalization", False)),
        learn_pos_embed=bool(raw.get("learn_pos_embed", True)),
        val_check_steps=int(raw.get("val_check_steps", 25)),
        early_stop_patience_steps=int(raw.get("early_stop_patience_steps", -1)),
    )


def load_patchtst_tuning_config(path: str | Path) -> PatchTSTTuningConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tuning = raw.get("tuning", {})
    training = raw.get("training", {})
    return PatchTSTTuningConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        objective=str(tuning.get("objective", "MASE")).upper(),
        slices=tuple(str(value) for value in training.get("slices", [])),
        base_model=_model_config(raw.get("base_model", {})),
        candidates=tuple(dict(candidate) for candidate in tuning["candidates"]),
        device=str(training.get("device", "auto")),
        output_dir=str(raw.get("output", {}).get("output_dir", "forecasting/experiments/runs/patchtst_tuning")),
    )


def tune_patchtst(config: PatchTSTTuningConfig) -> dict[str, Path]:
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

    for trial_idx, candidate in enumerate(config.candidates):
        model_config = _apply_params(config.base_model, candidate)
        prediction_rows = []
        scales = {}
        trained_models = 0
        start = time.perf_counter()
        print(f"[PatchTST tuning] trial={trial_idx + 1}/{len(config.candidates)} params={candidate}", flush=True)
        run_config = PatchTSTBenchmarkConfig(
            seed=config.seed,
            data=config.data,
            backtest=config.backtest,
            model=model_config,
            training=PatchTSTTrainingConfig(device=config.device, slices=config.slices),
            output_dir=config.output_dir,
        )
        for fold in folds:
            for slice_name in slice_names:
                series_list = _slice_series(series_map, str(slice_name))
                scales[(fold.fold, str(slice_name))] = _seasonal_scale(series_list, fold.train_end_idx, period=144)
                train_df = _training_frame(series_list, fold)
                forecast, _train_seconds, _inference_seconds, _parameter_count = _fit_predict_patchtst(train_df, run_config, alias="patchtst")
                trained_models += 1
                forecast = forecast.rename(columns={"patchtst": "y_pred"})
                forecast["y_pred"] = np.maximum(0.0, np.expm1(forecast["y_pred"].to_numpy(dtype=float)))
                forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
                for unique_id, pred_sub in forecast.groupby("unique_id", sort=False):
                    target = series_map[str(unique_id)].iloc[fold.target_start_idx : fold.target_end_idx + 1]
                    pred_values = pred_sub.sort_values("ds")["y_pred"].to_numpy(dtype=float)
                    for step_idx in range(config.backtest.horizon):
                        prediction_rows.append(
                            {
                                "fold": fold.fold,
                                "model": "patchtst",
                                "unique_id": str(unique_id),
                                "slice": str(slice_name),
                                "horizon": step_idx + 1,
                                "y_true": float(target["y"].iloc[step_idx]),
                                "y_pred": float(pred_values[step_idx]),
                            }
                        )
        metrics = _metric_rows(pd.DataFrame(prediction_rows), scales)
        score = float(metrics[config.objective].mean())
        elapsed = time.perf_counter() - start
        row = {
            "trial": trial_idx,
            "candidate": str(candidate.get("name", f"trial_{trial_idx}")),
            "objective": config.objective,
            "score": score,
            "seconds": elapsed,
            "trained_models": trained_models,
        }
        row.update(model_config.__dict__)
        trial_rows.append(row)
        print(f"[PatchTST tuning] trial={trial_idx + 1} score={score:.6f} seconds={elapsed:.2f}", flush=True)
        if score < best_score:
            best_score = score
            best_params = dict(model_config.__dict__)

    paths = {
        "trials": output_dir / "patchtst_tuning_trials.csv",
        "best_params": output_dir / "patchtst_best_params.yaml",
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
                "training": {"device": config.device, "slices": list(config.slices)},
                "tuning": {"objective": config.objective, "candidates": list(config.candidates)},
                "output": {"output_dir": config.output_dir},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return paths
