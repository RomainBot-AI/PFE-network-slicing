"""DeepAR probabilistic benchmark for subnet/slice forecasting."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nsf.benchmark.deterministic import (
    BenchmarkDataConfig,
    BenchmarkOutputConfig,
    _benchmark_summary,
    _benchmark_summary_by_slice,
    _metric_rows,
    _prepare_panel,
    _series_map,
    _summary_metrics,
    _validate_dense,
)
from nsf.benchmark.lstm_tuning import _seasonal_scale, _slice_series
from nsf.benchmark.probabilistic_lightgbm import (
    _enforce_monotonic_quantiles,
    _probabilistic_metric_rows,
    _probabilistic_summary,
    _probabilistic_summary_by_slice,
    _quantile_column,
    _summary_probabilistic_metrics,
    _validate_quantiles,
)
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.splitting.panel_folds import folds_to_frame, leakage_audit, make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class ProbabilisticDeepARModelConfig:
    name: str = "deepar"
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    distribution: str = "Normal"
    max_steps: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    windows_batch_size: int = 512
    step_size: int = 12
    scaler_type: str = "standard"
    lstm_n_layers: int = 1
    lstm_hidden_size: int = 64
    lstm_dropout: float = 0.1
    decoder_hidden_layers: int = 0
    decoder_hidden_size: int = 0
    trajectory_samples: int = 200
    num_samples: int = 200
    val_check_steps: int = 50
    early_stop_patience_steps: int = -1


@dataclass(frozen=True)
class ProbabilisticDeepARTrainingConfig:
    device: str = "auto"
    slices: tuple[str, ...] = field(default_factory=tuple)
    seasonal_scale_period: int = 144


@dataclass(frozen=True)
class ProbabilisticDeepARConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    model: ProbabilisticDeepARModelConfig
    training: ProbabilisticDeepARTrainingConfig
    output: BenchmarkOutputConfig


def load_probabilistic_deepar_config(path: str | Path) -> ProbabilisticDeepARConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model_raw = raw.get("model", {})
    training_raw = raw.get("training", {})
    return ProbabilisticDeepARConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        model=ProbabilisticDeepARModelConfig(
            name=str(model_raw.get("name", "deepar")),
            quantiles=tuple(float(value) for value in model_raw.get("quantiles", [0.1, 0.5, 0.9])),
            distribution=str(model_raw.get("distribution", "Normal")),
            max_steps=int(model_raw.get("max_steps", 100)),
            learning_rate=float(model_raw.get("learning_rate", 1e-3)),
            batch_size=int(model_raw.get("batch_size", 32)),
            windows_batch_size=int(model_raw.get("windows_batch_size", 512)),
            step_size=int(model_raw.get("step_size", 12)),
            scaler_type=str(model_raw.get("scaler_type", "standard")),
            lstm_n_layers=int(model_raw.get("lstm_n_layers", 1)),
            lstm_hidden_size=int(model_raw.get("lstm_hidden_size", 64)),
            lstm_dropout=float(model_raw.get("lstm_dropout", 0.1)),
            decoder_hidden_layers=int(model_raw.get("decoder_hidden_layers", 0)),
            decoder_hidden_size=int(model_raw.get("decoder_hidden_size", 0)),
            trajectory_samples=int(model_raw.get("trajectory_samples", 200)),
            num_samples=int(model_raw.get("num_samples", 200)),
            val_check_steps=int(model_raw.get("val_check_steps", 50)),
            early_stop_patience_steps=int(model_raw.get("early_stop_patience_steps", -1)),
        ),
        training=ProbabilisticDeepARTrainingConfig(
            device=str(training_raw.get("device", "auto")),
            slices=tuple(str(value) for value in training_raw.get("slices", [])),
            seasonal_scale_period=int(training_raw.get("seasonal_scale_period", 144)),
        ),
        output=BenchmarkOutputConfig(**raw.get("output", {})),
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


def _distribution_level(quantiles: tuple[float, ...]) -> int:
    lower = min(quantiles)
    upper = max(quantiles)
    if not np.isclose(lower, 1.0 - upper):
        raise ValueError("DeepAR DistributionLoss expects symmetric lower/upper quantiles")
    return int(round((upper - lower) * 100))


def _make_model(config: ProbabilisticDeepARConfig, alias: str):
    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.models import DeepAR

    model = config.model
    quantiles = _validate_quantiles(model.quantiles)
    return DeepAR(
        h=config.backtest.horizon,
        input_size=config.backtest.input_size,
        lstm_n_layers=model.lstm_n_layers,
        lstm_hidden_size=model.lstm_hidden_size,
        lstm_dropout=model.lstm_dropout,
        decoder_hidden_layers=model.decoder_hidden_layers,
        decoder_hidden_size=model.decoder_hidden_size,
        trajectory_samples=model.trajectory_samples,
        loss=DistributionLoss(
            distribution=model.distribution,
            level=[_distribution_level(quantiles)],
            num_samples=model.num_samples,
        ),
        max_steps=model.max_steps,
        learning_rate=model.learning_rate,
        batch_size=model.batch_size,
        windows_batch_size=model.windows_batch_size,
        step_size=model.step_size,
        scaler_type=model.scaler_type,
        val_check_steps=model.val_check_steps,
        early_stop_patience_steps=model.early_stop_patience_steps,
        random_seed=config.seed,
        alias=alias,
        **_trainer_kwargs(config.training.device),
    )


def _training_frame(series_list: list[pd.DataFrame], fold) -> pd.DataFrame:
    rows = []
    for sub in series_list:
        train = sub.iloc[: fold.train_end_idx + 1, :][["unique_id", "ds", "y"]].copy()
        train["y"] = np.log1p(np.maximum(0.0, train["y"].to_numpy(dtype=float)))
        rows.append(train)
    return pd.concat(rows, ignore_index=True)


def _fit_predict_deepar(train_df: pd.DataFrame, config: ProbabilisticDeepARConfig, alias: str) -> tuple[pd.DataFrame, float, float, int | None]:
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


def _forecast_to_quantiles(forecast: pd.DataFrame, alias: str, quantiles: tuple[float, ...]) -> pd.DataFrame:
    level = _distribution_level(quantiles)
    lower_candidates = (f"{alias}-lo-{float(level):.1f}", f"{alias}-lo-{level}")
    upper_candidates = (f"{alias}-hi-{float(level):.1f}", f"{alias}-hi-{level}")
    lower_col = next((col for col in lower_candidates if col in forecast.columns), None)
    median_col = f"{alias}-median"
    upper_col = next((col for col in upper_candidates if col in forecast.columns), None)
    missing = []
    if lower_col is None:
        missing.append("/".join(lower_candidates))
    if median_col not in forecast.columns:
        missing.append(median_col)
    if upper_col is None:
        missing.append("/".join(upper_candidates))
    if missing:
        raise ValueError(f"DeepAR forecast is missing expected quantile columns: {missing}. Available: {forecast.columns.tolist()}")
    assert lower_col is not None
    assert upper_col is not None
    out = forecast[["unique_id", "ds", lower_col, median_col, upper_col]].copy()
    out[_quantile_column(min(quantiles))] = np.maximum(0.0, np.expm1(out[lower_col].to_numpy(dtype=float)))
    out[_quantile_column(0.5)] = np.maximum(0.0, np.expm1(out[median_col].to_numpy(dtype=float)))
    out[_quantile_column(max(quantiles))] = np.maximum(0.0, np.expm1(out[upper_col].to_numpy(dtype=float)))
    return out[["unique_id", "ds", *[_quantile_column(q) for q in quantiles]]]


def run_probabilistic_deepar_benchmark(config: ProbabilisticDeepARConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
    quantiles = _validate_quantiles(config.model.quantiles)
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

    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    timing_rows = []
    metadata_rows = []
    scales = {}
    alias = config.model.name

    for fold in folds:
        for slice_name in slice_names:
            series_list = _slice_series(series_map, str(slice_name))
            train_df = _training_frame(series_list, fold)
            scales[(fold.fold, str(slice_name))] = _seasonal_scale(
                series_list,
                fold.train_end_idx,
                period=config.training.seasonal_scale_period,
            )
            forecast, train_seconds, inference_seconds, parameter_count = _fit_predict_deepar(train_df, config, alias=alias)
            forecast = _forecast_to_quantiles(forecast, alias=alias, quantiles=quantiles)
            forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
            timing_rows.append(
                {
                    "model": config.model.name,
                    "fold": fold.fold,
                    "slice": str(slice_name),
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "train_rows": len(train_df),
                    "eval_series": forecast["unique_id"].nunique(),
                    "device": config.training.device,
                }
            )
            metadata_rows.append(
                {
                    "model": config.model.name,
                    "fold": fold.fold,
                    "slice": str(slice_name),
                    "parameter_count": parameter_count,
                    "implementation": "neuralforecast.models.DeepAR + DistributionLoss",
                    "training_scope": "per_slice",
                    "trained_models": 1,
                    "distribution": config.model.distribution,
                    "quantiles": ",".join(str(q) for q in quantiles),
                }
            )
            for unique_id, pred_sub in forecast.groupby("unique_id", sort=False):
                original = series_map[str(unique_id)]
                target = original.iloc[fold.target_start_idx : fold.target_end_idx + 1]
                pred_sub = _enforce_monotonic_quantiles(pred_sub.sort_values("ds"), quantiles)
                if len(pred_sub) != config.backtest.horizon:
                    raise ValueError(f"Expected {config.backtest.horizon} predictions for {unique_id}, got {len(pred_sub)}")
                for step_idx in range(config.backtest.horizon):
                    row = {
                        "fold": fold.fold,
                        "model": config.model.name,
                        "training_scope": "per_slice",
                        "trained_slice": str(slice_name),
                        "unique_id": str(unique_id),
                        "slice": str(slice_name),
                        "origin_timestamp": fold.train_end,
                        "timestamp": target["ds"].iloc[step_idx],
                        "horizon": step_idx + 1,
                        "y_true": float(target["y"].iloc[step_idx]),
                    }
                    for quantile in quantiles:
                        row[_quantile_column(quantile)] = float(pred_sub[_quantile_column(quantile)].iloc[step_idx])
                    prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    median_predictions = predictions.rename(columns={_quantile_column(0.5): "y_pred"})
    deterministic_metrics_by_fold = _metric_rows(median_predictions, scales)
    deterministic_metrics = _summary_metrics(deterministic_metrics_by_fold)
    deterministic_summary = _benchmark_summary(deterministic_metrics)
    deterministic_summary_by_slice = _benchmark_summary_by_slice(deterministic_metrics)
    probabilistic_metrics_by_fold = _probabilistic_metric_rows(predictions, quantiles)
    probabilistic_metrics = _summary_probabilistic_metrics(probabilistic_metrics_by_fold)
    probabilistic_summary = _probabilistic_summary(probabilistic_metrics)
    probabilistic_summary_by_slice = _probabilistic_summary_by_slice(probabilistic_metrics)
    timing = pd.DataFrame(timing_rows)
    metadata = pd.DataFrame(metadata_rows)

    config_dict = {
        "seed": config.seed,
        "data": config.data.__dict__,
        "backtest": config.backtest.__dict__,
        "model": {
            "name": config.model.name,
            "quantiles": list(config.model.quantiles),
            "distribution": config.model.distribution,
            "max_steps": config.model.max_steps,
            "learning_rate": config.model.learning_rate,
            "batch_size": config.model.batch_size,
            "windows_batch_size": config.model.windows_batch_size,
            "step_size": config.model.step_size,
            "scaler_type": config.model.scaler_type,
            "lstm_n_layers": config.model.lstm_n_layers,
            "lstm_hidden_size": config.model.lstm_hidden_size,
            "lstm_dropout": config.model.lstm_dropout,
            "decoder_hidden_layers": config.model.decoder_hidden_layers,
            "decoder_hidden_size": config.model.decoder_hidden_size,
            "trajectory_samples": config.model.trajectory_samples,
            "num_samples": config.model.num_samples,
            "val_check_steps": config.model.val_check_steps,
            "early_stop_patience_steps": config.model.early_stop_patience_steps,
        },
        "training": {
            "device": config.training.device,
            "slices": list(config.training.slices),
            "seasonal_scale_period": config.training.seasonal_scale_period,
        },
        "output": config.output.__dict__,
    }
    paths = {
        "resolved_config": run_dir / "resolved_config.yaml",
        "run_meta": run_dir / "run_meta.json",
        "folds": run_dir / "folds.csv",
        "leakage_audit": run_dir / "leakage_audit.csv",
        "predictions": run_dir / "predictions_probabilistic.csv",
        "deterministic_metrics_by_fold": run_dir / "metrics_by_fold.csv",
        "deterministic_metrics": run_dir / "metrics.csv",
        "deterministic_summary": run_dir / "benchmark_summary.csv",
        "deterministic_summary_by_slice": run_dir / "benchmark_summary_by_slice.csv",
        "probabilistic_metrics_by_fold": run_dir / "metrics_probabilistic_by_fold.csv",
        "probabilistic_metrics": run_dir / "metrics_probabilistic.csv",
        "probabilistic_summary": run_dir / "benchmark_probabilistic_summary.csv",
        "probabilistic_summary_by_slice": run_dir / "benchmark_probabilistic_summary_by_slice.csv",
        "timing": run_dir / "timing.csv",
        "model_metadata": run_dir / "model_metadata.csv",
    }
    paths["resolved_config"].write_text(yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")
    paths["run_meta"].write_text(
        json.dumps(
            {
                "panel_rows": int(len(panel)),
                "series": int(panel["unique_id"].nunique()),
                "folds": len(folds),
                "horizon": config.backtest.horizon,
                "quantiles": list(quantiles),
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
    deterministic_metrics_by_fold.to_csv(ensure_parent(paths["deterministic_metrics_by_fold"]), index=False)
    deterministic_metrics.to_csv(ensure_parent(paths["deterministic_metrics"]), index=False)
    deterministic_summary.to_csv(ensure_parent(paths["deterministic_summary"]), index=False)
    deterministic_summary_by_slice.to_csv(ensure_parent(paths["deterministic_summary_by_slice"]), index=False)
    probabilistic_metrics_by_fold.to_csv(ensure_parent(paths["probabilistic_metrics_by_fold"]), index=False)
    probabilistic_metrics.to_csv(ensure_parent(paths["probabilistic_metrics"]), index=False)
    probabilistic_summary.to_csv(ensure_parent(paths["probabilistic_summary"]), index=False)
    probabilistic_summary_by_slice.to_csv(ensure_parent(paths["probabilistic_summary_by_slice"]), index=False)
    timing.to_csv(ensure_parent(paths["timing"]), index=False)
    metadata.to_csv(ensure_parent(paths["model_metadata"]), index=False)
    return paths
