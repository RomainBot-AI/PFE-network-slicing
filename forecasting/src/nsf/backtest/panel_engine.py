"""Configurable panel backtesting engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from nsf.config import ExperimentConfig, resolved_config_dict
from nsf.data.loading import read_panel
from nsf.evaluation.deterministic import bias, mae, rmse, smape, under_over_error, wape
from nsf.models.base import ForecastHorizon
from nsf.models.registry import make_forecaster
from nsf.splitting.panel_folds import folds_to_frame, leakage_audit, make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel[["unique_id", "ds", "y", "slice"]].copy()
    panel["unique_id"] = panel["unique_id"].astype(str)
    panel["slice"] = panel["slice"].astype(str)
    return panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _series_by_id(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(unique_id): sub.sort_values("ds").reset_index(drop=True) for unique_id, sub in panel.groupby("unique_id")}


def _validate_regular_panel(series_map: dict[str, pd.DataFrame], timestamps: pd.DatetimeIndex) -> None:
    expected = len(timestamps)
    for unique_id, sub in series_map.items():
        if len(sub) != expected:
            raise ValueError(f"Series {unique_id} has {len(sub)} points, expected dense panel length {expected}")
        if not pd.DatetimeIndex(sub["ds"]).equals(timestamps):
            raise ValueError(f"Series {unique_id} does not match the common timestamp grid")


def _metric_row(model: str, slice_name: str, horizon: int, fold: int, y_true: list[float], y_pred: list[float]) -> dict:
    under, over = under_over_error(y_true, y_pred)
    return {
        "model": model,
        "slice": slice_name,
        "horizon": horizon,
        "fold": fold,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "under_prediction_error": under,
        "over_prediction_error": over,
        "n_predictions": len(y_true),
    }


def run_panel_backtest(config: ExperimentConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    panel = _prepare_panel(read_panel(config.data.panel_csv))
    timestamps = pd.DatetimeIndex(sorted(panel["ds"].unique()))
    series_map = _series_by_id(panel)
    _validate_regular_panel(series_map, timestamps)

    folds = make_panel_folds(
        timestamps=timestamps,
        input_size=config.backtest.input_size,
        horizon=config.backtest.horizon,
        n_folds=config.backtest.n_folds,
        fold_stride=config.backtest.fold_stride,
        expanding=config.backtest.expanding,
    )

    prediction_rows = []
    metric_inputs: dict[tuple[str, str, int, int], dict[str, list[float]]] = {}

    for fold in folds:
        for model_config in config.models:
            for unique_id, sub in series_map.items():
                train = sub.iloc[fold.train_start_idx : fold.train_end_idx + 1]
                target = sub.iloc[fold.target_start_idx : fold.target_end_idx + 1]
                forecaster = make_forecaster(model_config.name, **model_config.params).fit(train["y"])
                forecast = forecaster.predict(ForecastHorizon(steps=config.backtest.horizon, freq=config.data.frequency))
                slice_name = str(sub["slice"].iloc[0])

                for step_idx in range(config.backtest.horizon):
                    horizon = step_idx + 1
                    y_true = float(target["y"].iloc[step_idx])
                    y_pred = float(forecast.iloc[step_idx])
                    prediction_rows.append(
                        {
                            "fold": fold.fold,
                            "model": model_config.name,
                            "unique_id": unique_id,
                            "slice": slice_name,
                            "origin_timestamp": fold.train_end,
                            "timestamp": target["ds"].iloc[step_idx],
                            "horizon": horizon,
                            "y_true": y_true,
                            "y_pred": y_pred,
                        }
                    )
                    key = (model_config.name, slice_name, horizon, fold.fold)
                    bucket = metric_inputs.setdefault(key, {"y_true": [], "y_pred": []})
                    bucket["y_true"].append(y_true)
                    bucket["y_pred"].append(y_pred)

    metric_rows = []
    for (model, slice_name, horizon, fold), values in metric_inputs.items():
        metric_rows.append(_metric_row(model, slice_name, horizon, fold, values["y_true"], values["y_pred"]))

    predictions = pd.DataFrame(prediction_rows)
    metrics_by_fold = pd.DataFrame(metric_rows).sort_values(["model", "slice", "horizon", "fold"])
    metrics = (
        metrics_by_fold.groupby(["model", "slice", "horizon"], as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            WAPE=("WAPE", "mean"),
            sMAPE=("sMAPE", "mean"),
            bias=("bias", "mean"),
            under_prediction_error=("under_prediction_error", "mean"),
            over_prediction_error=("over_prediction_error", "mean"),
            folds=("fold", "nunique"),
            n_predictions=("n_predictions", "sum"),
        )
        .sort_values(["model", "slice", "horizon"])
    )

    paths = {
        "resolved_config": run_dir / "resolved_config.yaml",
        "run_meta": run_dir / "run_meta.json",
        "folds": run_dir / "folds.csv",
        "leakage_audit": run_dir / "leakage_audit.csv",
        "predictions": run_dir / "predictions.csv",
        "metrics_by_fold": run_dir / "metrics_by_fold.csv",
        "metrics": run_dir / "metrics.csv",
    }
    paths["resolved_config"].write_text(yaml.safe_dump(resolved_config_dict(config), sort_keys=False), encoding="utf-8")
    paths["run_meta"].write_text(
        json.dumps(
            {
                "panel_rows": int(len(panel)),
                "series": int(panel["unique_id"].nunique()),
                "slices": int(panel["slice"].nunique()),
                "timestamps": int(len(timestamps)),
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
    return paths
