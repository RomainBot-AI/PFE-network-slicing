"""Per-slice N-HiTS benchmark using NeuralForecast."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nsf.benchmark.deterministic import (
    BenchmarkDataConfig,
    _benchmark_summary,
    _benchmark_summary_by_slice,
    _metric_rows,
    _prepare_panel,
    _series_map,
    _summary_metrics,
    _validate_dense,
)
from nsf.benchmark.lstm_tuning import _seasonal_scale, _slice_series
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.splitting.panel_folds import folds_to_frame, leakage_audit, make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class NHITSModelConfig:
    max_steps: int = 300
    learning_rate: float = 1e-3
    batch_size: int = 32
    windows_batch_size: int = 1024
    step_size: int = 288
    scaler_type: str = "standard"
    stack_types: tuple[str, ...] = ("identity", "identity", "identity")
    n_blocks: tuple[int, ...] = (1, 1, 1)
    mlp_units: tuple[tuple[int, ...], ...] = ((256, 256), (256, 256), (256, 256))
    n_pool_kernel_size: tuple[int, ...] = (2, 2, 1)
    n_freq_downsample: tuple[int, ...] = (4, 2, 1)
    dropout_prob_theta: float = 0.0
    val_check_steps: int = 50
    early_stop_patience_steps: int = -1


@dataclass(frozen=True)
class NHITSTrainingConfig:
    device: str = "auto"
    slices: tuple[str, ...] = field(default_factory=tuple)
    params_path: str | None = None


@dataclass(frozen=True)
class NHITSBenchmarkConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    model: NHITSModelConfig
    training: NHITSTrainingConfig
    output_dir: str


def _tuple(raw: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    values = raw if raw is not None else default
    return tuple(values)


def _nested_tuple(raw: Any, default: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    values = raw if raw is not None else default
    return tuple(tuple(int(item) for item in row) for row in values)


def load_nhits_benchmark_config(path: str | Path) -> NHITSBenchmarkConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model_raw = raw.get("model", {})
    training_raw = raw.get("training", {})
    return NHITSBenchmarkConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        model=NHITSModelConfig(
            max_steps=int(model_raw.get("max_steps", 300)),
            learning_rate=float(model_raw.get("learning_rate", 1e-3)),
            batch_size=int(model_raw.get("batch_size", 32)),
            windows_batch_size=int(model_raw.get("windows_batch_size", 1024)),
            step_size=int(model_raw.get("step_size", 288)),
            scaler_type=str(model_raw.get("scaler_type", "standard")),
            stack_types=tuple(str(value) for value in _tuple(model_raw.get("stack_types"), ("identity", "identity", "identity"))),
            n_blocks=tuple(int(value) for value in _tuple(model_raw.get("n_blocks"), (1, 1, 1))),
            mlp_units=_nested_tuple(model_raw.get("mlp_units"), ((256, 256), (256, 256), (256, 256))),
            n_pool_kernel_size=tuple(int(value) for value in _tuple(model_raw.get("n_pool_kernel_size"), (2, 2, 1))),
            n_freq_downsample=tuple(int(value) for value in _tuple(model_raw.get("n_freq_downsample"), (4, 2, 1))),
            dropout_prob_theta=float(model_raw.get("dropout_prob_theta", 0.0)),
            val_check_steps=int(model_raw.get("val_check_steps", 50)),
            early_stop_patience_steps=int(model_raw.get("early_stop_patience_steps", -1)),
        ),
        training=NHITSTrainingConfig(
            device=str(training_raw.get("device", "auto")),
            slices=tuple(str(value) for value in training_raw.get("slices", [])),
            params_path=training_raw.get("params_path"),
        ),
        output_dir=str(raw.get("output", {}).get("output_dir", "experiments/runs/nhits_benchmark")),
    )


def _trainer_kwargs(device: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    if device == "cuda":
        kwargs.update({"accelerator": "gpu", "devices": 1})
    elif device == "cpu":
        kwargs.update({"accelerator": "cpu"})
    else:
        kwargs.update({"accelerator": "auto"})
    return kwargs


def _parameter_count(model: Any) -> int | None:
    try:
        return int(sum(param.numel() for param in model.parameters()))
    except Exception:
        return None


def _make_model(config: NHITSBenchmarkConfig, alias: str):
    from neuralforecast.models import NHITS

    model_config = config.model
    return NHITS(
        h=config.backtest.horizon,
        input_size=config.backtest.input_size,
        max_steps=model_config.max_steps,
        learning_rate=model_config.learning_rate,
        batch_size=model_config.batch_size,
        windows_batch_size=model_config.windows_batch_size,
        step_size=model_config.step_size,
        scaler_type=model_config.scaler_type,
        stack_types=list(model_config.stack_types),
        n_blocks=list(model_config.n_blocks),
        mlp_units=[list(row) for row in model_config.mlp_units],
        n_pool_kernel_size=list(model_config.n_pool_kernel_size),
        n_freq_downsample=list(model_config.n_freq_downsample),
        dropout_prob_theta=model_config.dropout_prob_theta,
        val_check_steps=model_config.val_check_steps,
        early_stop_patience_steps=model_config.early_stop_patience_steps,
        random_seed=config.seed,
        alias=alias,
        **_trainer_kwargs(config.training.device),
    )


def _load_best_params(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    params = raw.get("best_params_by_slice", raw)
    return {str(slice_name): dict(values or {}) for slice_name, values in params.items()}


def _apply_tuned_params(model: NHITSModelConfig, params: dict[str, Any]) -> NHITSModelConfig:
    if not params:
        return model
    width = int(params.get("mlp_width", model.mlp_units[0][0]))
    n_blocks = int(params.get("n_blocks", model.n_blocks[0]))
    stack_count = len(model.stack_types)
    return NHITSModelConfig(
        max_steps=int(params.get("max_steps", model.max_steps)),
        learning_rate=float(params.get("learning_rate", model.learning_rate)),
        batch_size=int(params.get("batch_size", model.batch_size)),
        windows_batch_size=int(params.get("windows_batch_size", model.windows_batch_size)),
        step_size=int(params.get("step_size", model.step_size)),
        scaler_type=model.scaler_type,
        stack_types=model.stack_types,
        n_blocks=tuple(n_blocks for _ in range(stack_count)),
        mlp_units=tuple((width, width) for _ in range(stack_count)),
        n_pool_kernel_size=model.n_pool_kernel_size,
        n_freq_downsample=model.n_freq_downsample,
        dropout_prob_theta=float(params.get("dropout_prob_theta", model.dropout_prob_theta)),
        val_check_steps=model.val_check_steps,
        early_stop_patience_steps=model.early_stop_patience_steps,
    )


def _fit_predict_nhits(train_df: pd.DataFrame, config: NHITSBenchmarkConfig, alias: str) -> tuple[pd.DataFrame, float, float, int | None]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    from neuralforecast import NeuralForecast

    model = _make_model(config, alias)
    nf = NeuralForecast(models=[model], freq=config.data.frequency)
    train_start = time.perf_counter()
    nf.fit(df=train_df)
    train_seconds = time.perf_counter() - train_start
    infer_start = time.perf_counter()
    forecast = nf.predict()
    inference_seconds = time.perf_counter() - infer_start
    return forecast, train_seconds, inference_seconds, _parameter_count(nf.models[0])


def _training_frame(series_list: list[pd.DataFrame], fold) -> pd.DataFrame:
    rows = []
    for sub in series_list:
        train = sub.iloc[: fold.train_end_idx + 1, :][["unique_id", "ds", "y"]].copy()
        train["y"] = np.log1p(np.maximum(0.0, train["y"].to_numpy(dtype=float)))
        rows.append(train)
    return pd.concat(rows, ignore_index=True)


def run_nhits_benchmark(config: NHITSBenchmarkConfig) -> dict[str, Path]:
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
    if config.training.slices:
        requested = set(config.training.slices)
        slice_names = [slice_name for slice_name in slice_names if slice_name in requested]
        missing = sorted(requested - set(slice_names))
        if missing:
            raise ValueError(f"Unknown slices requested: {missing}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    timing_rows = []
    metadata_rows = []
    scales = {}
    best_params = _load_best_params(config.training.params_path)

    for fold in folds:
        for slice_name in slice_names:
            series_list = _slice_series(series_map, str(slice_name))
            train_df = _training_frame(series_list, fold)
            alias = "nhits"
            scale = _seasonal_scale(series_list, fold.train_end_idx, period=144)
            scales[(fold.fold, str(slice_name))] = scale
            tuned_model = _apply_tuned_params(config.model, best_params.get(str(slice_name), {}))
            run_config = replace(config, model=tuned_model)
            forecast, train_seconds, inference_seconds, parameter_count = _fit_predict_nhits(train_df, run_config, alias=alias)
            forecast = forecast.rename(columns={alias: "y_pred"})
            forecast["y_pred"] = np.maximum(0.0, np.expm1(forecast["y_pred"].to_numpy(dtype=float)))
            forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
            timing_rows.append(
                {
                    "model": "nhits",
                    "fold": fold.fold,
                    "slice": str(slice_name),
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "train_rows": len(train_df),
                    "eval_series": forecast["unique_id"].nunique(),
                    "device": config.training.device,
                    "params_source": config.training.params_path or "",
                }
            )
            metadata_rows.append(
                {
                    "model": "nhits",
                    "fold": fold.fold,
                    "slice": str(slice_name),
                    "parameter_count": parameter_count,
                    "implementation": "neuralforecast.models.NHITS",
                    "training_scope": "per_slice",
                    "trained_models": 1,
                    "params_source": config.training.params_path or "",
                }
            )
            for unique_id, pred_sub in forecast.groupby("unique_id", sort=False):
                original = series_map[str(unique_id)]
                target = original.iloc[fold.target_start_idx : fold.target_end_idx + 1]
                pred_values = pred_sub.sort_values("ds")["y_pred"].to_numpy(dtype=float)
                if len(pred_values) != config.backtest.horizon:
                    raise ValueError(f"Expected {config.backtest.horizon} predictions for {unique_id}, got {len(pred_values)}")
                for step_idx in range(config.backtest.horizon):
                    prediction_rows.append(
                        {
                            "fold": fold.fold,
                            "model": "nhits",
                            "training_scope": "per_slice",
                            "trained_slice": str(slice_name),
                            "unique_id": str(unique_id),
                            "slice": str(slice_name),
                            "origin_timestamp": fold.train_end,
                            "timestamp": target["ds"].iloc[step_idx],
                            "horizon": step_idx + 1,
                            "y_true": float(target["y"].iloc[step_idx]),
                            "y_pred": float(pred_values[step_idx]),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    metrics_by_fold = _metric_rows(predictions, scales)
    metrics = _summary_metrics(metrics_by_fold)
    summary = _benchmark_summary(metrics)
    summary_by_slice = _benchmark_summary_by_slice(metrics)
    timing = pd.DataFrame(timing_rows)
    metadata = pd.DataFrame(metadata_rows)
    paths = {
        "resolved_config": output_dir / "resolved_config.yaml",
        "run_meta": output_dir / "run_meta.json",
        "folds": output_dir / "folds.csv",
        "leakage_audit": output_dir / "leakage_audit.csv",
        "predictions": output_dir / "predictions.csv",
        "metrics_by_fold": output_dir / "metrics_by_fold.csv",
        "metrics": output_dir / "metrics.csv",
        "benchmark_summary": output_dir / "benchmark_summary.csv",
        "benchmark_summary_by_slice": output_dir / "benchmark_summary_by_slice.csv",
        "timing": output_dir / "timing.csv",
        "model_metadata": output_dir / "model_metadata.csv",
    }
    paths["resolved_config"].write_text(
        yaml.safe_dump(
            {
                "seed": config.seed,
                "data": config.data.__dict__,
                "backtest": config.backtest.__dict__,
                "model": {
                    "max_steps": config.model.max_steps,
                    "learning_rate": config.model.learning_rate,
                    "batch_size": config.model.batch_size,
                    "windows_batch_size": config.model.windows_batch_size,
                    "step_size": config.model.step_size,
                    "scaler_type": config.model.scaler_type,
                    "stack_types": list(config.model.stack_types),
                    "n_blocks": list(config.model.n_blocks),
                    "mlp_units": [list(row) for row in config.model.mlp_units],
                    "n_pool_kernel_size": list(config.model.n_pool_kernel_size),
                    "n_freq_downsample": list(config.model.n_freq_downsample),
                    "dropout_prob_theta": config.model.dropout_prob_theta,
                    "val_check_steps": config.model.val_check_steps,
                    "early_stop_patience_steps": config.model.early_stop_patience_steps,
                },
                "training": {
                    "device": config.training.device,
                    "slices": list(config.training.slices),
                    "params_path": config.training.params_path,
                },
                "output": {"output_dir": config.output_dir},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    paths["run_meta"].write_text(
        json.dumps(
            {
                "panel_rows": int(len(panel)),
                "series": int(panel["unique_id"].nunique()),
                "folds": len(folds),
                "horizon": config.backtest.horizon,
                "predictions": int(len(predictions)),
                "trained_models": int(len(metadata)),
                "training_scope": "per_slice",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    folds_to_frame(folds).to_csv(ensure_parent(paths["folds"]), index=False)
    leakage_audit(folds).to_csv(ensure_parent(paths["leakage_audit"]), index=False)
    predictions.to_csv(ensure_parent(paths["predictions"]), index=False)
    metrics_by_fold.to_csv(ensure_parent(paths["metrics_by_fold"]), index=False)
    metrics.to_csv(ensure_parent(paths["metrics"]), index=False)
    summary.to_csv(ensure_parent(paths["benchmark_summary"]), index=False)
    summary_by_slice.to_csv(ensure_parent(paths["benchmark_summary_by_slice"]), index=False)
    timing.to_csv(ensure_parent(paths["timing"]), index=False)
    metadata.to_csv(ensure_parent(paths["model_metadata"]), index=False)
    return paths
