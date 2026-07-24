"""Deterministic forecasting benchmark for the subnet/slice panel."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nsf.config import BacktestConfig, ModelConfig
from nsf.data.loading import read_panel
from nsf.evaluation.deterministic import bias, mae, mase, rmse, smape, under_over_error, wape
from nsf.models.base import ForecastHorizon
from nsf.models.registry import make_forecaster
from nsf.splitting.panel_folds import folds_to_frame, leakage_audit, make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class BenchmarkDataConfig:
    panel_csv: str
    frequency: str = "10min"


@dataclass(frozen=True)
class BenchmarkFeatureConfig:
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 36, 144, 1008)
    train_origin_stride: int = 36
    seasonal_scale_period: int = 144


@dataclass(frozen=True)
class BenchmarkOutputConfig:
    run_dir: str = "experiments/runs/deterministic_benchmark"


@dataclass(frozen=True)
class DeterministicBenchmarkConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    features: BenchmarkFeatureConfig
    models: tuple[ModelConfig, ...]
    output: BenchmarkOutputConfig


def load_benchmark_config(path: str | Path) -> DeterministicBenchmarkConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    models = tuple(ModelConfig(name=item["name"], params=dict(item.get("params", {}))) for item in raw["models"])
    return DeterministicBenchmarkConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        features=BenchmarkFeatureConfig(
            lags=tuple(int(value) for value in raw.get("features", {}).get("lags", [1, 2, 3, 6, 12, 36, 144, 1008])),
            train_origin_stride=int(raw.get("features", {}).get("train_origin_stride", 36)),
            seasonal_scale_period=int(raw.get("features", {}).get("seasonal_scale_period", 144)),
        ),
        models=models,
        output=BenchmarkOutputConfig(**raw.get("output", {})),
    )


def _load_slice_params(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    params_path = Path(path)
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    raw = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    return {str(slice_name): dict(values or {}) for slice_name, values in raw.get("best_params_by_slice", raw).items()}


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["unique_id", "ds", "y", "slice"]
    optional = [col for col in ["id_institution", "id_institution_subnet"] if col in panel.columns]
    panel = panel[cols + optional].copy()
    panel["unique_id"] = panel["unique_id"].astype(str)
    panel["slice"] = panel["slice"].astype(str)
    panel["y"] = pd.to_numeric(panel["y"], errors="coerce")
    return panel.dropna(subset=["unique_id", "ds", "y", "slice"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _series_map(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(unique_id): sub.sort_values("ds").reset_index(drop=True) for unique_id, sub in panel.groupby("unique_id", sort=False)}


def _validate_dense(series_map: dict[str, pd.DataFrame], timestamps: pd.DatetimeIndex) -> None:
    for unique_id, sub in series_map.items():
        if len(sub) != len(timestamps):
            raise ValueError(f"Series {unique_id} is not dense")
        if not pd.DatetimeIndex(sub["ds"]).equals(timestamps):
            raise ValueError(f"Series {unique_id} does not match common timestamps")


def _seasonal_scales(series_map: dict[str, pd.DataFrame], fold, period: int) -> dict[str, float]:
    by_slice: dict[str, list[float]] = {}
    for sub in series_map.values():
        values = sub["y"].iloc[fold.train_start_idx : fold.train_end_idx + 1].to_numpy(dtype=float)
        slice_name = str(sub["slice"].iloc[0])
        if len(values) > period:
            diffs = np.abs(values[period:] - values[:-period])
            by_slice.setdefault(slice_name, []).extend(diffs[np.isfinite(diffs)].tolist())
    return {slice_name: float(np.mean(values)) if values else 1.0 for slice_name, values in by_slice.items()}


def _metric_rows(predictions: pd.DataFrame, scales: dict[tuple[int, str], float]) -> pd.DataFrame:
    rows = []
    for (model, fold, slice_name, horizon), sub in predictions.groupby(["model", "fold", "slice", "horizon"], sort=False):
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_pred"].to_numpy(dtype=float)
        under, over = under_over_error(y_true, y_pred)
        rows.append(
            {
                "model": model,
                "fold": fold,
                "slice": slice_name,
                "horizon": horizon,
                "MAE": mae(y_true, y_pred),
                "RMSE": rmse(y_true, y_pred),
                "WAPE": wape(y_true, y_pred),
                "MASE": mase(y_true, y_pred, scales.get((fold, slice_name), 1.0)),
                "sMAPE": smape(y_true, y_pred),
                "bias": bias(y_true, y_pred),
                "under_prediction_error": under,
                "over_prediction_error": over,
                "n_predictions": len(sub),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "slice", "horizon", "fold"])


def _summary_metrics(metrics_by_fold: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_by_fold.groupby(["model", "slice", "horizon"], as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            WAPE=("WAPE", "mean"),
            MASE=("MASE", "mean"),
            sMAPE=("sMAPE", "mean"),
            bias=("bias", "mean"),
            under_prediction_error=("under_prediction_error", "mean"),
            over_prediction_error=("over_prediction_error", "mean"),
            folds=("fold", "nunique"),
            n_predictions=("n_predictions", "sum"),
        )
        .sort_values(["model", "slice", "horizon"])
    )


def _benchmark_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    by_model = (
        metrics.groupby("model", as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            WAPE=("WAPE", "mean"),
            MASE=("MASE", "mean"),
            horizons=("horizon", "nunique"),
            slices=("slice", "nunique"),
        )
        .sort_values("RMSE")
    )
    by_model["rank_rmse"] = by_model["RMSE"].rank(method="dense")
    return by_model


def _benchmark_summary_by_slice(metrics: pd.DataFrame) -> pd.DataFrame:
    by_slice = (
        metrics.groupby(["slice", "model"], as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            WAPE=("WAPE", "mean"),
            MASE=("MASE", "mean"),
            sMAPE=("sMAPE", "mean"),
            bias=("bias", "mean"),
            horizons=("horizon", "nunique"),
            folds=("folds", "max"),
            n_predictions=("n_predictions", "sum"),
        )
        .sort_values(["slice", "RMSE"])
    )
    by_slice["rank_rmse_within_slice"] = by_slice.groupby("slice")["RMSE"].rank(method="dense")
    return by_slice


def _baseline_predictions(config, folds, series_map) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    timing_rows = []
    metadata_rows = []
    baseline_models = [model for model in config.models if model.name != "lightgbm"]
    for model_config in baseline_models:
        train_seconds = 0.0
        infer_seconds = 0.0
        for fold in folds:
            for unique_id, sub in series_map.items():
                train = sub.iloc[fold.train_start_idx : fold.train_end_idx + 1]
                target = sub.iloc[fold.target_start_idx : fold.target_end_idx + 1]
                start = time.perf_counter()
                forecaster = make_forecaster(model_config.name, **model_config.params).fit(train["y"])
                train_seconds += time.perf_counter() - start
                start = time.perf_counter()
                forecast = forecaster.predict(ForecastHorizon(steps=config.backtest.horizon, freq=config.data.frequency))
                infer_seconds += time.perf_counter() - start
                for step_idx in range(config.backtest.horizon):
                    prediction_rows.append(
                        {
                            "fold": fold.fold,
                            "model": model_config.name,
                            "unique_id": unique_id,
                            "slice": str(sub["slice"].iloc[0]),
                            "origin_timestamp": fold.train_end,
                            "timestamp": target["ds"].iloc[step_idx],
                            "horizon": step_idx + 1,
                            "y_true": float(target["y"].iloc[step_idx]),
                            "y_pred": max(0.0, float(forecast.iloc[step_idx])),
                        }
                    )
        timing_rows.append({"model": model_config.name, "train_seconds": train_seconds, "inference_seconds": infer_seconds})
        metadata_rows.append({"model": model_config.name, "parameter_count": 0, "implementation": "nsf baseline"})
    return pd.DataFrame(prediction_rows), pd.DataFrame(timing_rows), pd.DataFrame(metadata_rows)


def _feature_row(sub: pd.DataFrame, origin_idx: int, lags: tuple[int, ...], horizon: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "horizon": horizon,
        "origin_hour": int(sub["ds"].iloc[origin_idx].hour),
        "origin_dayofweek": int(sub["ds"].iloc[origin_idx].dayofweek),
        "origin_is_weekend": int(sub["ds"].iloc[origin_idx].dayofweek >= 5),
        "slice": str(sub["slice"].iloc[0]),
    }
    if "id_institution" in sub.columns:
        row["id_institution"] = int(sub["id_institution"].iloc[0])
    if "id_institution_subnet" in sub.columns:
        row["id_institution_subnet"] = int(sub["id_institution_subnet"].iloc[0])
    for lag in lags:
        row[f"lag_{lag}"] = float(sub["y"].iloc[origin_idx - lag + 1])
        row[f"log_lag_{lag}"] = float(np.log1p(max(0.0, row[f"lag_{lag}"])))
    return row


def _make_lgbm_train_frame(series_map, fold, lags: tuple[int, ...], horizon: int, stride: int) -> tuple[pd.DataFrame, pd.Series]:
    max_lag = max(lags)
    rows = []
    target = []
    max_origin = fold.train_end_idx - horizon
    min_origin = fold.train_start_idx + max_lag - 1
    for sub in series_map.values():
        for origin_idx in range(min_origin, max_origin + 1, stride):
            rows.append(_feature_row(sub, origin_idx, lags, horizon))
            target.append(float(np.log1p(max(0.0, sub["y"].iloc[origin_idx + horizon]))))
    x = pd.DataFrame(rows)
    y = pd.Series(target, name="target_log")
    return _encode_features(x), y


def _make_lgbm_eval_frame(series_map, fold, lags: tuple[int, ...], horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta = []
    origin_idx = fold.train_end_idx
    target_idx = fold.target_start_idx + horizon - 1
    for unique_id, sub in series_map.items():
        rows.append(_feature_row(sub, origin_idx, lags, horizon))
        meta.append(
            {
                "fold": fold.fold,
                "unique_id": unique_id,
                "slice": str(sub["slice"].iloc[0]),
                "origin_timestamp": fold.train_end,
                "timestamp": sub["ds"].iloc[target_idx],
                "horizon": horizon,
                "y_true": float(sub["y"].iloc[target_idx]),
            }
        )
    return _encode_features(pd.DataFrame(rows)), pd.DataFrame(meta)


def _filter_series_by_slice(series_map: dict[str, pd.DataFrame], slice_name: str) -> dict[str, pd.DataFrame]:
    return {
        unique_id: sub
        for unique_id, sub in series_map.items()
        if str(sub["slice"].iloc[0]) == slice_name
    }


def _encode_features(x: pd.DataFrame) -> pd.DataFrame:
    x = pd.get_dummies(x, columns=["slice"], prefix="slice", dtype=float)
    return x.astype(float)


def _align_columns(train_x: pd.DataFrame, eval_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_x = eval_x.reindex(columns=train_x.columns, fill_value=0.0)
    return train_x, eval_x


def _lightgbm_predictions(config, folds, series_map) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lgbm_configs = [model for model in config.models if model.name == "lightgbm"]
    if not lgbm_configs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise RuntimeError("Model 'lightgbm' requires lightgbm. Install with: .venv/bin/python -m pip install lightgbm") from exc

    prediction_rows = []
    timing_rows = []
    metadata_rows = []
    for model_config in lgbm_configs:
        training_scope = str(model_config.params.get("training_scope", "per_slice"))
        if training_scope not in {"per_slice", "global"}:
            raise ValueError("LightGBM training_scope must be 'per_slice' or 'global'")
        slice_params_path = model_config.params.get("slice_params_path")
        best_params_by_slice = _load_slice_params(slice_params_path)
        model_params = {
            key: value
            for key, value in model_config.params.items()
            if key not in {"training_scope", "slice_params_path"}
        }
        slice_scopes = sorted({str(sub["slice"].iloc[0]) for sub in series_map.values()}) if training_scope == "per_slice" else ["__global__"]
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
                    eval_x, meta = _make_lgbm_eval_frame(series_map=scoped_series, fold=fold, lags=config.features.lags, horizon=horizon)
                    train_x, eval_x = _align_columns(train_x, eval_x)
                    params = {
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
                    model = LGBMRegressor(**params)
                    start = time.perf_counter()
                    model.fit(train_x, train_y)
                    train_seconds += time.perf_counter() - start
                    start = time.perf_counter()
                    pred_log = model.predict(eval_x)
                    infer_seconds += time.perf_counter() - start
                    meta["model"] = model_config.name
                    meta["training_scope"] = training_scope
                    meta["trained_slice"] = slice_scope
                    meta["y_pred"] = np.maximum(0.0, np.expm1(pred_log))
                    prediction_rows.extend(meta[["fold", "model", "training_scope", "trained_slice", "unique_id", "slice", "origin_timestamp", "timestamp", "horizon", "y_true", "y_pred"]].to_dict("records"))
                    model_count += 1
                    feature_count = train_x.shape[1]
        timing_rows.append({"model": model_config.name, "train_seconds": train_seconds, "inference_seconds": infer_seconds})
        metadata_rows.append({"model": model_config.name, "training_scope": training_scope, "parameter_count": np.nan, "implementation": "lightgbm.LGBMRegressor", "trained_models": model_count, "features": feature_count})
    return pd.DataFrame(prediction_rows), pd.DataFrame(timing_rows), pd.DataFrame(metadata_rows)


def run_deterministic_benchmark(config: DeterministicBenchmarkConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
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
        fold_scales = _seasonal_scales(series_map, fold, config.features.seasonal_scale_period)
        for slice_name, scale in fold_scales.items():
            scales[(fold.fold, slice_name)] = scale

    baseline_preds, baseline_timing, baseline_meta = _baseline_predictions(config, folds, series_map)
    lgbm_preds, lgbm_timing, lgbm_meta = _lightgbm_predictions(config, folds, series_map)
    predictions = pd.concat([frame for frame in [baseline_preds, lgbm_preds] if not frame.empty], ignore_index=True)
    timing = pd.concat([frame for frame in [baseline_timing, lgbm_timing] if not frame.empty], ignore_index=True)
    model_metadata = pd.concat([frame for frame in [baseline_meta, lgbm_meta] if not frame.empty], ignore_index=True)
    metrics_by_fold = _metric_rows(predictions, scales)
    metrics = _summary_metrics(metrics_by_fold)
    summary = _benchmark_summary(metrics)
    summary_by_slice = _benchmark_summary_by_slice(metrics)

    paths = {
        "resolved_config": run_dir / "resolved_config.yaml",
        "run_meta": run_dir / "run_meta.json",
        "folds": run_dir / "folds.csv",
        "leakage_audit": run_dir / "leakage_audit.csv",
        "predictions": run_dir / "predictions.csv",
        "metrics_by_fold": run_dir / "metrics_by_fold.csv",
        "metrics": run_dir / "metrics.csv",
        "benchmark_summary": run_dir / "benchmark_summary.csv",
        "benchmark_summary_by_slice": run_dir / "benchmark_summary_by_slice.csv",
        "timing": run_dir / "timing.csv",
        "model_metadata": run_dir / "model_metadata.csv",
    }
    config_dict = {
        "seed": config.seed,
        "data": config.data.__dict__,
        "backtest": config.backtest.__dict__,
        "features": {
            "lags": list(config.features.lags),
            "train_origin_stride": config.features.train_origin_stride,
            "seasonal_scale_period": config.features.seasonal_scale_period,
        },
        "models": [{"name": model.name, "params": model.params} for model in config.models],
        "output": config.output.__dict__,
    }
    paths["resolved_config"].write_text(yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")
    paths["run_meta"].write_text(
        json.dumps(
            {
                "panel_rows": int(len(panel)),
                "series": int(panel["unique_id"].nunique()),
                "folds": len(folds),
                "horizon": config.backtest.horizon,
                "predictions": int(len(predictions)),
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
    model_metadata.to_csv(ensure_parent(paths["model_metadata"]), index=False)
    return paths
