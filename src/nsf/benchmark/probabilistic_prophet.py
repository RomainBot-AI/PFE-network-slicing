"""Prophet interval benchmark for probabilistic subnet/slice forecasting."""

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
from nsf.benchmark.prophet_benchmark import _apply_params, _load_best_params, _prophet_kwargs, _seasonality_value
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.splitting.panel_folds import folds_to_frame, leakage_audit, make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class ProbabilisticProphetModelConfig:
    name: str = "prophet_interval"
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    daily_seasonality: bool | str = True
    weekly_seasonality: bool | str = True
    yearly_seasonality: bool | str = False
    seasonality_mode: str = "additive"
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    interval_width: float = 0.8
    log_transform: bool = True


@dataclass(frozen=True)
class ProbabilisticProphetTrainingConfig:
    slices: tuple[str, ...] = field(default_factory=tuple)
    max_series_per_slice: int | None = None
    train_tail: int | None = None
    params_path: str | None = None
    seasonal_scale_period: int = 144


@dataclass(frozen=True)
class ProbabilisticProphetConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    model: ProbabilisticProphetModelConfig
    training: ProbabilisticProphetTrainingConfig
    output: BenchmarkOutputConfig


def load_probabilistic_prophet_config(path: str | Path) -> ProbabilisticProphetConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model_raw = raw.get("model", {})
    training_raw = raw.get("training", {})
    max_series_per_slice = training_raw.get("max_series_per_slice")
    train_tail = training_raw.get("train_tail")
    return ProbabilisticProphetConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        model=ProbabilisticProphetModelConfig(
            name=str(model_raw.get("name", "prophet_interval")),
            quantiles=tuple(float(value) for value in model_raw.get("quantiles", [0.1, 0.5, 0.9])),
            daily_seasonality=_seasonality_value(model_raw.get("daily_seasonality", True)),
            weekly_seasonality=_seasonality_value(model_raw.get("weekly_seasonality", True)),
            yearly_seasonality=_seasonality_value(model_raw.get("yearly_seasonality", False)),
            seasonality_mode=str(model_raw.get("seasonality_mode", "additive")),
            changepoint_prior_scale=float(model_raw.get("changepoint_prior_scale", 0.05)),
            seasonality_prior_scale=float(model_raw.get("seasonality_prior_scale", 10.0)),
            interval_width=float(model_raw.get("interval_width", 0.8)),
            log_transform=bool(model_raw.get("log_transform", True)),
        ),
        training=ProbabilisticProphetTrainingConfig(
            slices=tuple(str(value) for value in training_raw.get("slices", [])),
            max_series_per_slice=None if max_series_per_slice in (None, 0) else int(max_series_per_slice),
            train_tail=None if train_tail in (None, 0) else int(train_tail),
            params_path=training_raw.get("params_path"),
            seasonal_scale_period=int(training_raw.get("seasonal_scale_period", 144)),
        ),
        output=BenchmarkOutputConfig(**raw.get("output", {})),
    )


def _as_prophet_model(config: ProbabilisticProphetModelConfig):
    from nsf.benchmark.prophet_benchmark import ProphetModelConfig

    return ProphetModelConfig(
        daily_seasonality=config.daily_seasonality,
        weekly_seasonality=config.weekly_seasonality,
        yearly_seasonality=config.yearly_seasonality,
        seasonality_mode=config.seasonality_mode,
        changepoint_prior_scale=config.changepoint_prior_scale,
        seasonality_prior_scale=config.seasonality_prior_scale,
        interval_width=config.interval_width,
        log_transform=config.log_transform,
    )


def _fit_predict_prophet_interval(
    train: pd.DataFrame,
    future_ds: pd.Series,
    config: ProbabilisticProphetModelConfig,
) -> tuple[pd.DataFrame, float, float]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    try:
        from prophet import Prophet
    except ModuleNotFoundError as exc:
        raise RuntimeError("Prophet interval benchmark requires prophet. Install it with: .venv/bin/pip install prophet") from exc

    y = train["y"].to_numpy(dtype=float)
    if config.log_transform:
        y = np.log1p(np.maximum(0.0, y))
    train_df = pd.DataFrame({"ds": pd.to_datetime(train["ds"], utc=True).dt.tz_convert(None), "y": y})
    future = pd.DataFrame({"ds": pd.to_datetime(future_ds, utc=True).dt.tz_convert(None)})

    prophet_config = _as_prophet_model(config)
    model = Prophet(**_prophet_kwargs(prophet_config))
    train_start = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            model.fit(train_df)
    train_seconds = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    forecast = model.predict(future)
    inference_seconds = time.perf_counter() - infer_start

    out = pd.DataFrame(
        {
            _quantile_column(0.1): forecast["yhat_lower"].to_numpy(dtype=float),
            _quantile_column(0.5): forecast["yhat"].to_numpy(dtype=float),
            _quantile_column(0.9): forecast["yhat_upper"].to_numpy(dtype=float),
        }
    )
    if config.log_transform:
        out = np.expm1(out)
    return pd.DataFrame(np.maximum(0.0, out), columns=out.columns), train_seconds, inference_seconds


def run_probabilistic_prophet_benchmark(config: ProbabilisticProphetConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
    quantiles = _validate_quantiles(config.model.quantiles)
    if quantiles != (0.1, 0.5, 0.9):
        raise ValueError("Prophet native intervals currently map to quantiles [0.1, 0.5, 0.9]")
    expected_width = max(quantiles) - min(quantiles)
    if not np.isclose(config.model.interval_width, expected_width):
        raise ValueError(f"Prophet interval_width must be {expected_width} for quantiles {quantiles}")

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
    tuned_model = _apply_params(_as_prophet_model(config.model), _load_best_params(config.training.params_path))
    model_config = ProbabilisticProphetModelConfig(
        name=config.model.name,
        quantiles=config.model.quantiles,
        daily_seasonality=tuned_model.daily_seasonality,
        weekly_seasonality=tuned_model.weekly_seasonality,
        yearly_seasonality=tuned_model.yearly_seasonality,
        seasonality_mode=tuned_model.seasonality_mode,
        changepoint_prior_scale=tuned_model.changepoint_prior_scale,
        seasonality_prior_scale=tuned_model.seasonality_prior_scale,
        interval_width=tuned_model.interval_width,
        log_transform=tuned_model.log_transform,
    )

    for fold in folds:
        for slice_name in slice_names:
            series_list = _slice_series(series_map, str(slice_name))
            if config.training.max_series_per_slice is not None:
                series_list = series_list[: config.training.max_series_per_slice]
            scales[(fold.fold, str(slice_name))] = _seasonal_scale(
                series_list,
                fold.train_end_idx,
                period=config.training.seasonal_scale_period,
            )
            for sub in series_list:
                unique_id = str(sub["unique_id"].iloc[0])
                train = sub.iloc[fold.train_start_idx : fold.train_end_idx + 1]
                if config.training.train_tail is not None:
                    train = train.tail(config.training.train_tail)
                target = sub.iloc[fold.target_start_idx : fold.target_end_idx + 1]
                pred, train_seconds, inference_seconds = _fit_predict_prophet_interval(train, target["ds"], model_config)
                pred = _enforce_monotonic_quantiles(pred, quantiles)
                timing_rows.append(
                    {
                        "model": config.model.name,
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
                        "model": config.model.name,
                        "fold": fold.fold,
                        "slice": str(slice_name),
                        "unique_id": unique_id,
                        "parameter_count": None,
                        "implementation": "prophet.Prophet native interval",
                        "training_scope": "local_per_series",
                        "trained_models": 1,
                        "quantiles": ",".join(str(q) for q in quantiles),
                        "params_source": config.training.params_path or "",
                    }
                )
                for step_idx in range(config.backtest.horizon):
                    row = {
                        "fold": fold.fold,
                        "model": config.model.name,
                        "training_scope": "local_per_series",
                        "trained_slice": str(slice_name),
                        "unique_id": unique_id,
                        "slice": str(slice_name),
                        "origin_timestamp": fold.train_end,
                        "timestamp": target["ds"].iloc[step_idx],
                        "horizon": step_idx + 1,
                        "y_true": float(target["y"].iloc[step_idx]),
                    }
                    for quantile in quantiles:
                        row[_quantile_column(quantile)] = float(pred[_quantile_column(quantile)].iloc[step_idx])
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
                    "seasonal_scale_period": config.training.seasonal_scale_period,
                },
                "output": config.output.__dict__,
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
                "quantiles": list(quantiles),
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
