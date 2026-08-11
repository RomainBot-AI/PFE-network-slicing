"""Fold-aware panel feature dataset preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nsf.config import BacktestConfig, DataConfig
from nsf.data.loading import read_panel
from nsf.preprocessing.scaler import fit_log_zscore, transform_log_zscore
from nsf.splitting.panel_folds import folds_to_frame, leakage_audit, make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class FeatureConfig:
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 36, 144, 1008)
    calendar: bool = True
    log_target: bool = True
    scale_per_series: bool = True


@dataclass(frozen=True)
class PreprocessConfig:
    seed: int
    data: DataConfig
    backtest: BacktestConfig
    features: FeatureConfig
    output_dir: str


def load_preprocess_config(path: str | Path) -> PreprocessConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return PreprocessConfig(
        seed=int(raw.get("seed", 42)),
        data=DataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        features=FeatureConfig(
            lags=tuple(int(value) for value in raw.get("features", {}).get("lags", [1, 2, 3, 6, 12, 36, 144, 1008])),
            calendar=bool(raw.get("features", {}).get("calendar", True)),
            log_target=bool(raw.get("features", {}).get("log_target", True)),
            scale_per_series=bool(raw.get("features", {}).get("scale_per_series", True)),
        ),
        output_dir=str(raw.get("output", {}).get("output_dir", "data/processed/subnet_slice_baseline")),
    )


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = ["unique_id", "ds", "y", "slice"]
    optional = [col for col in ["id_institution", "id_institution_subnet"] if col in panel.columns]
    panel = panel[required + optional].copy()
    panel["unique_id"] = panel["unique_id"].astype(str)
    panel["slice"] = panel["slice"].astype(str)
    panel["y"] = pd.to_numeric(panel["y"], errors="coerce")
    return panel.dropna(subset=["unique_id", "ds", "y", "slice"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _series_by_id(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(unique_id): sub.sort_values("ds").reset_index(drop=True) for unique_id, sub in panel.groupby("unique_id", sort=False)}


def _calendar_features(timestamp: pd.Timestamp, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_hour": int(timestamp.hour),
        f"{prefix}_dayofweek": int(timestamp.dayofweek),
        f"{prefix}_is_weekend": int(timestamp.dayofweek >= 5),
    }


def _fit_params(train_values: pd.Series, use_log: bool):
    if use_log:
        return fit_log_zscore(train_values)
    mean = float(train_values.mean())
    std = float(train_values.std())
    if std == 0.0 or np.isnan(std):
        std = 1.0
    return {"mean": mean, "std": std}


def _transform_values(values: pd.Series | np.ndarray, params, use_log: bool) -> np.ndarray:
    if use_log:
        return transform_log_zscore(values, params)
    raw = np.asarray(values, dtype=float)
    return (raw - params["mean"]) / params["std"]


def _params_to_dict(params, use_log: bool) -> dict[str, float]:
    if use_log:
        return {"mean": float(params.mean), "std": float(params.std), "transform": "log1p_zscore"}
    return {"mean": float(params["mean"]), "std": float(params["std"]), "transform": "zscore"}


def prepare_panel_feature_dataset(config: PreprocessConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _prepare_panel(read_panel(config.data.panel_csv))
    timestamps = pd.DatetimeIndex(sorted(panel["ds"].unique()))
    folds = make_panel_folds(
        timestamps=timestamps,
        input_size=config.backtest.input_size,
        horizon=config.backtest.horizon,
        n_folds=config.backtest.n_folds,
        fold_stride=config.backtest.fold_stride,
        expanding=config.backtest.expanding,
    )
    series_map = _series_by_id(panel)
    max_lag = max(config.features.lags) if config.features.lags else 0
    use_log_scale = config.features.log_target and config.features.scale_per_series

    rows = []
    scaler_rows = []
    audit_rows = []

    for fold in folds:
        if fold.train_end_idx - max_lag + 1 < fold.train_start_idx:
            raise ValueError(f"Fold {fold.fold} does not contain enough training history for max lag {max_lag}")
        for unique_id, sub in series_map.items():
            train = sub.iloc[fold.train_start_idx : fold.train_end_idx + 1].copy()
            target = sub.iloc[fold.target_start_idx : fold.target_end_idx + 1].copy()
            if len(train) != fold.train_end_idx - fold.train_start_idx + 1:
                raise ValueError(f"Series {unique_id} has incomplete train window for fold {fold.fold}")
            if len(target) != config.backtest.horizon:
                raise ValueError(f"Series {unique_id} has incomplete target window for fold {fold.fold}")

            params = _fit_params(train["y"], use_log=use_log_scale)
            scaler_row = {
                "fold": fold.fold,
                "unique_id": unique_id,
                "slice": str(sub["slice"].iloc[0]),
                "train_start": fold.train_start,
                "train_end": fold.train_end,
            }
            scaler_row.update(_params_to_dict(params, use_log=use_log_scale))
            scaler_rows.append(scaler_row)

            lag_values = {}
            lag_scaled = {}
            for lag in config.features.lags:
                lag_idx = fold.train_end_idx - lag + 1
                if lag_idx < fold.train_start_idx:
                    raise ValueError(f"Leakage/lag error: lag {lag} is outside fold {fold.fold} train window")
                raw_value = float(sub["y"].iloc[lag_idx])
                lag_values[f"lag_{lag}"] = raw_value
                lag_scaled[f"lag_{lag}_scaled"] = float(_transform_values(np.asarray([raw_value]), params, use_log=use_log_scale)[0])

            for step_idx in range(config.backtest.horizon):
                horizon = step_idx + 1
                target_timestamp = target["ds"].iloc[step_idx]
                y_raw = float(target["y"].iloc[step_idx])
                y_scaled = float(_transform_values(np.asarray([y_raw]), params, use_log=use_log_scale)[0])
                row = {
                    "fold": fold.fold,
                    "unique_id": unique_id,
                    "slice": str(sub["slice"].iloc[0]),
                    "origin_timestamp": fold.train_end,
                    "target_timestamp": target_timestamp,
                    "horizon": horizon,
                    "y": y_raw,
                    "y_scaled": y_scaled,
                }
                if "id_institution" in sub.columns:
                    row["id_institution"] = int(sub["id_institution"].iloc[0])
                if "id_institution_subnet" in sub.columns:
                    row["id_institution_subnet"] = int(sub["id_institution_subnet"].iloc[0])
                if config.features.calendar:
                    row.update(_calendar_features(fold.train_end, "origin"))
                    row.update(_calendar_features(target_timestamp, "target"))
                row.update(lag_values)
                row.update(lag_scaled)
                rows.append(row)

            audit_rows.append(
                {
                    "fold": fold.fold,
                    "unique_id": unique_id,
                    "train_end_before_target": fold.train_end < fold.target_start,
                    "max_lag_available_in_train": max_lag <= len(train),
                    "target_rows": len(target),
                    "train_rows": len(train),
                }
            )

    features = pd.DataFrame(rows)
    scalers = pd.DataFrame(scaler_rows)
    audit = pd.DataFrame(audit_rows)

    paths = {
        "resolved_config": output_dir / "resolved_config.yaml",
        "features": output_dir / "features.csv",
        "scalers": output_dir / "scalers.csv",
        "folds": output_dir / "folds.csv",
        "leakage_audit": output_dir / "leakage_audit.csv",
        "feature_audit": output_dir / "feature_audit.csv",
        "run_meta": output_dir / "run_meta.json",
    }
    paths["resolved_config"].write_text(
        yaml.safe_dump(
            {
                "seed": config.seed,
                "data": {"panel_csv": config.data.panel_csv, "frequency": config.data.frequency},
                "backtest": config.backtest.__dict__,
                "features": {
                    "lags": list(config.features.lags),
                    "calendar": config.features.calendar,
                    "log_target": config.features.log_target,
                    "scale_per_series": config.features.scale_per_series,
                },
                "output": {"output_dir": config.output_dir},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    features.to_csv(ensure_parent(paths["features"]), index=False)
    scalers.to_csv(ensure_parent(paths["scalers"]), index=False)
    folds_to_frame(folds).to_csv(ensure_parent(paths["folds"]), index=False)
    leakage_audit(folds).to_csv(ensure_parent(paths["leakage_audit"]), index=False)
    audit.to_csv(ensure_parent(paths["feature_audit"]), index=False)
    paths["run_meta"].write_text(
        json.dumps(
            {
                "rows": int(len(features)),
                "series": int(features["unique_id"].nunique()),
                "folds": int(features["fold"].nunique()),
                "horizon": int(config.backtest.horizon),
                "max_lag": int(max_lag),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths
