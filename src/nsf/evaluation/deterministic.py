"""Deterministic forecasting metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _finite_arrays(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true, pred = _finite_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true, pred = _finite_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def wape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true, pred = _finite_arrays(y_true, y_pred)
    denom = float(np.sum(np.abs(true)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(np.abs(true - pred)) / denom)


def mase(y_true: Iterable[float], y_pred: Iterable[float], scale: float) -> float:
    true, pred = _finite_arrays(y_true, y_pred)
    if scale <= 0.0 or not np.isfinite(scale):
        return 0.0
    return float(np.mean(np.abs(true - pred)) / scale)


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true, pred = _finite_arrays(y_true, y_pred)
    denom = np.abs(true) + np.abs(pred)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(pred[mask] - true[mask]) / denom[mask]))


def bias(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true, pred = _finite_arrays(y_true, y_pred)
    return float(np.mean(pred - true))


def under_over_error(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[float, float]:
    true, pred = _finite_arrays(y_true, y_pred)
    diff = pred - true
    return float(np.maximum(-diff, 0.0).mean()), float(np.maximum(diff, 0.0).mean())
