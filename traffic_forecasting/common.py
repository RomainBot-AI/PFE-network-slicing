"""Shared helpers for traffic forecasting scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SLICES = ["URLLC", "URLLC_eMBB_MIX", "eMBB", "mMTC"]


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def read_slice_series(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    missing = [name for name in SLICES if name not in df.columns]
    if missing:
        raise ValueError(f"Missing slice columns in {path}: {missing}")
    return df[SLICES].astype(float)


def chronological_split(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if len(df) <= test_size:
        raise ValueError(f"Series length {len(df)} must be greater than test_size {test_size}")
    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()


def normalize_log_zscore(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    train_norm = train.copy()
    test_norm = test.copy()
    params: dict[str, tuple[float, float]] = {}

    for slice_name in SLICES:
        train_log = np.log1p(train[slice_name].to_numpy(dtype=float))
        mu = float(train_log.mean())
        sigma = float(train_log.std())
        if sigma == 0:
            sigma = 1.0
        train_norm[slice_name] = (train_log - mu) / sigma
        test_norm[slice_name] = (np.log1p(test[slice_name].to_numpy(dtype=float)) - mu) / sigma
        params[slice_name] = (mu, sigma)

    return train_norm, test_norm, params


def denormalize_log_zscore(values: Iterable[float], slice_name: str, params: dict[str, tuple[float, float]]) -> np.ndarray:
    mu, sigma = params[slice_name]
    y_log = np.asarray(values, dtype=float) * sigma + mu
    return np.expm1(y_log)


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    return float(np.mean(np.abs(true[mask] - pred[mask])))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    return float(np.sqrt(np.mean((true[mask] - pred[mask]) ** 2)))


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(true) + np.abs(pred)
    mask = np.isfinite(true) & np.isfinite(pred) & (denom > 0)
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(pred[mask] - true[mask]) / denom[mask]))


def bias(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    return float(np.mean(pred[mask] - true[mask]))


def under_over_error(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[float, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    diff = pred[mask] - true[mask]
    under = np.maximum(-diff, 0.0)
    over = np.maximum(diff, 0.0)
    return float(under.mean()), float(over.mean())
