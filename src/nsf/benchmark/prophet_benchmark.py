"""Local Prophet benchmark for the subnet/slice panel."""

from __future__ import annotations

import contextlib
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
class ProphetModelConfig:
    daily_seasonality: bool | str = True
    weekly_seasonality: bool | str = True
    yearly_seasonality: bool | str = False
    seasonality_mode: str = "additive"
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    interval_width: float = 0.8
    log_transform: bool = True


@dataclass(frozen=True)
class ProphetTrainingConfig:
    slices: tuple[str, ...] = field(default_factory=tuple)
    max_series_per_slice: int | None = None
    train_tail: int | None = None
    params_path: str | None = None


@dataclass(frozen=True)
class ProphetBenchmarkConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    model: ProphetModelConfig
    training: ProphetTrainingConfig
    output_dir: str


def _seasonality_value(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value
    return bool(value)


def load_prophet_benchmark_config(path: str | Path) -> ProphetBenchmarkConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model_raw = raw.get("model", {})
    training_raw = raw.get("training", {})
    max_series_per_slice = training_raw.get("max_series_per_slice")
    train_tail = training_raw.get("train_tail")
    return ProphetBenchmarkConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        model=ProphetModelConfig(
            daily_seasonality=_seasonality_value(model_raw.get("daily_seasonality", True)),
            weekly_seasonality=_seasonality_value(model_raw.get("weekly_seasonality", True)),
            yearly_seasonality=_seasonality_value(model_raw.get("yearly_seasonality", False)),
            seasonality_mode=str(model_raw.get("seasonality_mode", "additive")),
            changepoint_prior_scale=float(model_raw.get("changepoint_prior_scale", 0.05)),
            seasonality_prior_scale=float(model_raw.get("seasonality_prior_scale", 10.0)),
            interval_width=float(model_raw.get("interval_width", 0.8)),
            log_transform=bool(model_raw.get("log_transform", True)),
        ),
        training=ProphetTrainingConfig(
            slices=tuple(str(value) for value in training_raw.get("slices", [])),
            max_series_per_slice=None if max_series_per_slice in (None, 0) else int(max_series_per_slice),
            train_tail=None if train_tail in (None, 0) else int(train_tail),
            params_path=training_raw.get("params_path"),
        ),
        output_dir=str(raw.get("output", {}).get("output_dir", "experiments/runs/prophet_benchmark")),
    )


def _prophet_kwargs(model: ProphetModelConfig) -> dict[str, Any]:
    return {
        "daily_seasonality": model.daily_seasonality,
        "weekly_seasonality": model.weekly_seasonality,
        "yearly_seasonality": model.yearly_seasonality,
        "seasonality_mode": model.seasonality_mode,
        "changepoint_prior_scale": model.changepoint_prior_scale,
        "seasonality_prior_scale": model.seasonality_prior_scale,
        "interval_width": model.interval_width,
    }


def _load_best_params(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return dict(raw.get("best_params", raw))


def _apply_params(model: ProphetModelConfig, params: dict[str, Any]) -> ProphetModelConfig:
    if not params:
        return model
    return ProphetModelConfig(
        daily_seasonality=_seasonality_value(params.get("daily_seasonality", model.daily_seasonality)),
        weekly_seasonality=_seasonality_value(params.get("weekly_seasonality", model.weekly_seasonality)),
        yearly_seasonality=_seasonality_value(params.get("yearly_seasonality", model.yearly_seasonality)),
        seasonality_mode=str(params.get("seasonality_mode", model.seasonality_mode)),
        changepoint_prior_scale=float(params.get("changepoint_prior_scale", model.changepoint_prior_scale)),
        seasonality_prior_scale=float(params.get("seasonality_prior_scale", model.seasonality_prior_scale)),
        interval_width=float(params.get("interval_width", model.interval_width)),
        log_transform=bool(params.get("log_transform", model.log_transform)),
    )


def _fit_predict_prophet(train: pd.DataFrame, future_ds: pd.Series, config: ProphetModelConfig) -> tuple[np.ndarray, float, float]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    try:
        from prophet import Prophet
    except ModuleNotFoundError as exc:
        raise RuntimeError("Prophet benchmark requires prophet. Install it with: .venv/bin/pip install prophet") from exc

    y = train["y"].to_numpy(dtype=float)
    if config.log_transform:
        y = np.log1p(np.maximum(0.0, y))
    train_df = pd.DataFrame({"ds": pd.to_datetime(train["ds"], utc=True).dt.tz_convert(None), "y": y})
    future = pd.DataFrame({"ds": pd.to_datetime(future_ds, utc=True).dt.tz_convert(None)})

    model = Prophet(**_prophet_kwargs(config))
    train_start = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            model.fit(train_df)
    train_seconds = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    forecast = model.predict(future)
    inference_seconds = time.perf_counter() - infer_start
    y_pred = forecast["yhat"].to_numpy(dtype=float)
    if config.log_transform:
        y_pred = np.expm1(y_pred)
    return np.maximum(0.0, y_pred), train_seconds, inference_seconds


def run_prophet_benchmark(config: ProphetBenchmarkConfig) -> dict[str, Path]:
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
    model_config = _apply_params(config.model, _load_best_params(config.training.params_path))

    for fold in folds:
        for slice_name in slice_names:
            series_list = _slice_series(series_map, str(slice_name))
            if config.training.max_series_per_slice is not None:
                series_list = series_list[: config.training.max_series_per_slice]
            scales[(fold.fold, str(slice_name))] = _seasonal_scale(series_list, fold.train_end_idx, period=144)

            for sub in series_list:
                unique_id = str(sub["unique_id"].iloc[0])
                train = sub.iloc[fold.train_start_idx : fold.train_end_idx + 1]
                if config.training.train_tail is not None:
                    train = train.tail(config.training.train_tail)
                target = sub.iloc[fold.target_start_idx : fold.target_end_idx + 1]
                pred, train_seconds, inference_seconds = _fit_predict_prophet(train, target["ds"], model_config)
                timing_rows.append(
                    {
                        "model": "prophet",
                        "fold": fold.fold,
                        "slice": str(slice_name),
                        "unique_id": unique_id,
                        "train_seconds": train_seconds,
                        "inference_seconds": inference_seconds,
                        "train_rows": len(train),
                        "eval_rows": len(target),
                    }
                )
                metadata_rows.append(
                    {
                        "model": "prophet",
                        "fold": fold.fold,
                        "slice": str(slice_name),
                        "unique_id": unique_id,
                        "parameter_count": None,
                        "implementation": "prophet.Prophet",
                        "training_scope": "local_per_series",
                        "trained_models": 1,
                        "params_source": config.training.params_path or "",
                    }
                )
                for step_idx in range(config.backtest.horizon):
                    prediction_rows.append(
                        {
                            "fold": fold.fold,
                            "model": "prophet",
                            "training_scope": "local_per_series",
                            "trained_slice": str(slice_name),
                            "unique_id": unique_id,
                            "slice": str(slice_name),
                            "origin_timestamp": fold.train_end,
                            "timestamp": target["ds"].iloc[step_idx],
                            "horizon": step_idx + 1,
                            "y_true": float(target["y"].iloc[step_idx]),
                            "y_pred": float(pred[step_idx]),
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
                "model": model_config.__dict__,
                "training": {
                    "slices": list(config.training.slices),
                    "max_series_per_slice": config.training.max_series_per_slice,
                    "train_tail": config.training.train_tail,
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
                "training_scope": "local_per_series",
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
