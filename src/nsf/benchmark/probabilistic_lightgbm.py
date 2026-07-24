"""LightGBM quantile benchmark for probabilistic subnet/slice forecasting."""

from __future__ import annotations

import json
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
    BenchmarkOutputConfig,
    _align_columns,
    _benchmark_summary,
    _benchmark_summary_by_slice,
    _filter_series_by_slice,
    _load_slice_params,
    _make_lgbm_eval_frame,
    _make_lgbm_train_frame,
    _metric_rows,
    _prepare_panel,
    _seasonal_scales,
    _series_map,
    _summary_metrics,
    _validate_dense,
)
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.evaluation.probabilistic import (
    interval_coverage,
    interval_score,
    interval_width,
    normalized_interval_width,
    pinball_loss,
)
from nsf.splitting.panel_folds import folds_to_frame, leakage_audit, make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class ProbabilisticLightGBMModelConfig:
    name: str = "lightgbm_quantile"
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    params: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProbabilisticLightGBMConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    features: BenchmarkFeatureConfig
    model: ProbabilisticLightGBMModelConfig
    output: BenchmarkOutputConfig


def load_probabilistic_lightgbm_config(path: str | Path) -> ProbabilisticLightGBMConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model_raw = raw.get("model", {})
    return ProbabilisticLightGBMConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        features=BenchmarkFeatureConfig(
            lags=tuple(int(value) for value in raw.get("features", {}).get("lags", [1, 2, 3, 6, 12, 36, 144, 1008])),
            train_origin_stride=int(raw.get("features", {}).get("train_origin_stride", 36)),
            seasonal_scale_period=int(raw.get("features", {}).get("seasonal_scale_period", 144)),
        ),
        model=ProbabilisticLightGBMModelConfig(
            name=str(model_raw.get("name", "lightgbm_quantile")),
            quantiles=tuple(float(value) for value in model_raw.get("quantiles", [0.1, 0.5, 0.9])),
            params=dict(model_raw.get("params", {})),
        ),
        output=BenchmarkOutputConfig(**raw.get("output", {})),
    )


def _validate_quantiles(quantiles: tuple[float, ...]) -> tuple[float, ...]:
    if not quantiles:
        raise ValueError("At least one quantile is required")
    ordered = tuple(sorted(float(q) for q in quantiles))
    if any(q <= 0.0 or q >= 1.0 for q in ordered):
        raise ValueError("Quantiles must be strictly between 0 and 1")
    if 0.5 not in ordered:
        raise ValueError("Quantile benchmark requires q=0.5 for median deterministic metrics")
    return ordered


def _quantile_column(quantile: float) -> str:
    return f"q{int(round(quantile * 100)):02d}"


def _enforce_monotonic_quantiles(frame: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
    cols = [_quantile_column(q) for q in quantiles]
    values = np.maximum.accumulate(frame[cols].to_numpy(dtype=float), axis=1)
    frame = frame.copy()
    frame[cols] = values
    return frame


def _probabilistic_metric_rows(predictions: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
    lower_q = min(quantiles)
    upper_q = max(quantiles)
    lower_col = _quantile_column(lower_q)
    upper_col = _quantile_column(upper_q)
    alpha = 1.0 - (upper_q - lower_q)
    rows = []
    for (model, fold, slice_name, horizon), sub in predictions.groupby(["model", "fold", "slice", "horizon"], sort=False):
        y_true = sub["y_true"].to_numpy(dtype=float)
        row = {
            "model": model,
            "fold": fold,
            "slice": slice_name,
            "horizon": horizon,
            "coverage": interval_coverage(y_true, sub[lower_col], sub[upper_col]),
            "interval_width": interval_width(sub[lower_col], sub[upper_col]),
            "normalized_interval_width": normalized_interval_width(y_true, sub[lower_col], sub[upper_col]),
            "interval_score": interval_score(y_true, sub[lower_col], sub[upper_col], alpha=alpha),
            "n_predictions": len(sub),
        }
        for quantile in quantiles:
            col = _quantile_column(quantile)
            row[f"pinball_{col}"] = pinball_loss(y_true, sub[col], quantile)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "slice", "horizon", "fold"])


def _summary_probabilistic_metrics(metrics_by_fold: pd.DataFrame) -> pd.DataFrame:
    value_cols = [col for col in metrics_by_fold.columns if col not in {"model", "fold", "slice", "horizon", "n_predictions"}]
    agg = {col: (col, "mean") for col in value_cols}
    agg["folds"] = ("fold", "nunique")
    agg["n_predictions"] = ("n_predictions", "sum")
    return (
        metrics_by_fold.groupby(["model", "slice", "horizon"], as_index=False)
        .agg(**agg)
        .sort_values(["model", "slice", "horizon"])
    )


def _probabilistic_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    value_cols = [col for col in metrics.columns if col not in {"model", "slice", "horizon", "folds", "n_predictions"}]
    by_model = metrics.groupby("model", as_index=False).agg(
        **{col: (col, "mean") for col in value_cols},
        horizons=("horizon", "nunique"),
        slices=("slice", "nunique"),
    )
    return by_model.sort_values(["interval_score", "normalized_interval_width"])


def _probabilistic_summary_by_slice(metrics: pd.DataFrame) -> pd.DataFrame:
    value_cols = [col for col in metrics.columns if col not in {"model", "slice", "horizon", "folds", "n_predictions"}]
    by_slice = metrics.groupby(["slice", "model"], as_index=False).agg(
        **{col: (col, "mean") for col in value_cols},
        horizons=("horizon", "nunique"),
        folds=("folds", "max"),
        n_predictions=("n_predictions", "sum"),
    )
    return by_slice.sort_values(["slice", "interval_score", "normalized_interval_width"])


def _lightgbm_quantile_predictions(config: ProbabilisticLightGBMConfig, folds, series_map) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise RuntimeError("LightGBM quantile benchmark requires lightgbm. Install with: .venv/bin/python -m pip install lightgbm") from exc

    quantiles = _validate_quantiles(config.model.quantiles)
    model_params = dict(config.model.params or {})
    training_scope = str(model_params.pop("training_scope", "per_slice"))
    if training_scope not in {"per_slice", "global"}:
        raise ValueError("LightGBM training_scope must be 'per_slice' or 'global'")
    best_params_by_slice = _load_slice_params(model_params.pop("slice_params_path", None))
    slice_scopes = sorted({str(sub["slice"].iloc[0]) for sub in series_map.values()}) if training_scope == "per_slice" else ["__global__"]

    prediction_rows = []
    train_seconds = 0.0
    infer_seconds = 0.0
    model_count = 0
    feature_count = 0
    for fold in folds:
        for horizon in range(1, config.backtest.horizon + 1):
            for slice_scope in slice_scopes:
                scoped_series = series_map if slice_scope == "__global__" else _filter_series_by_slice(series_map, slice_scope)
                train_x, train_y = _make_lgbm_train_frame(
                    series_map=scoped_series,
                    fold=fold,
                    lags=config.features.lags,
                    horizon=horizon,
                    stride=config.features.train_origin_stride,
                )
                eval_x, meta = _make_lgbm_eval_frame(scoped_series, fold=fold, lags=config.features.lags, horizon=horizon)
                train_x, eval_x = _align_columns(train_x, eval_x)
                meta["model"] = config.model.name
                meta["training_scope"] = training_scope
                meta["trained_slice"] = slice_scope

                pred_cols = {}
                for quantile in quantiles:
                    params = {
                        "objective": "quantile",
                        "alpha": quantile,
                        "n_estimators": 200,
                        "learning_rate": 0.05,
                        "num_leaves": 31,
                        "random_state": config.seed,
                        "n_jobs": 1,
                        "verbosity": -1,
                    }
                    params.update(model_params)
                    if slice_scope != "__global__":
                        params.update(best_params_by_slice.get(slice_scope, {}))
                    params["objective"] = "quantile"
                    params["alpha"] = quantile
                    model = LGBMRegressor(**params)
                    start = time.perf_counter()
                    model.fit(train_x, train_y)
                    train_seconds += time.perf_counter() - start
                    start = time.perf_counter()
                    pred_log = model.predict(eval_x)
                    infer_seconds += time.perf_counter() - start
                    pred_cols[_quantile_column(quantile)] = np.maximum(0.0, np.expm1(pred_log))
                    model_count += 1
                    feature_count = train_x.shape[1]
                for col, values in pred_cols.items():
                    meta[col] = values
                meta = _enforce_monotonic_quantiles(meta, quantiles)
                prediction_rows.extend(
                    meta[
                        [
                            "fold",
                            "model",
                            "training_scope",
                            "trained_slice",
                            "unique_id",
                            "slice",
                            "origin_timestamp",
                            "timestamp",
                            "horizon",
                            "y_true",
                            *[_quantile_column(q) for q in quantiles],
                        ]
                    ].to_dict("records")
                )
    timing = pd.DataFrame(
        [{"model": config.model.name, "train_seconds": train_seconds, "inference_seconds": infer_seconds}]
    )
    metadata = pd.DataFrame(
        [
            {
                "model": config.model.name,
                "implementation": "lightgbm.LGBMRegressor(objective=quantile)",
                "training_scope": training_scope,
                "parameter_count": np.nan,
                "trained_models": model_count,
                "features": feature_count,
                "quantiles": ",".join(str(q) for q in quantiles),
            }
        ]
    )
    return pd.DataFrame(prediction_rows), timing, metadata


def run_probabilistic_lightgbm_benchmark(config: ProbabilisticLightGBMConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
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
    scales = {}
    for fold in folds:
        for slice_name, scale in _seasonal_scales(series_map, fold, config.features.seasonal_scale_period).items():
            scales[(fold.fold, slice_name)] = scale

    predictions, timing, metadata = _lightgbm_quantile_predictions(config, folds, series_map)
    median_predictions = predictions.rename(columns={_quantile_column(0.5): "y_pred"})
    deterministic_metrics_by_fold = _metric_rows(median_predictions, scales)
    deterministic_metrics = _summary_metrics(deterministic_metrics_by_fold)
    deterministic_summary = _benchmark_summary(deterministic_metrics)
    deterministic_summary_by_slice = _benchmark_summary_by_slice(deterministic_metrics)
    probabilistic_metrics_by_fold = _probabilistic_metric_rows(predictions, quantiles)
    probabilistic_metrics = _summary_probabilistic_metrics(probabilistic_metrics_by_fold)
    probabilistic_summary = _probabilistic_summary(probabilistic_metrics)
    probabilistic_summary_by_slice = _probabilistic_summary_by_slice(probabilistic_metrics)

    config_dict = {
        "seed": config.seed,
        "data": config.data.__dict__,
        "backtest": config.backtest.__dict__,
        "features": {
            "lags": list(config.features.lags),
            "train_origin_stride": config.features.train_origin_stride,
            "seasonal_scale_period": config.features.seasonal_scale_period,
        },
        "model": {
            "name": config.model.name,
            "quantiles": list(config.model.quantiles),
            "params": dict(config.model.params or {}),
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
