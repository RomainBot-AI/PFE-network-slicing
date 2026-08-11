"""Probabilistic forecasting metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _finite_arrays(*arrays: Iterable[float]) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(array, dtype=float) for array in arrays)
    mask = np.ones_like(converted[0], dtype=bool)
    for array in converted:
        mask &= np.isfinite(array)
    return tuple(array[mask] for array in converted)


def pinball_loss(y_true: Iterable[float], y_quantile: Iterable[float], quantile: float) -> float:
    true, pred = _finite_arrays(y_true, y_quantile)
    diff = true - pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def interval_coverage(y_true: Iterable[float], lower: Iterable[float], upper: Iterable[float]) -> float:
    true, low, high = _finite_arrays(y_true, lower, upper)
    if len(true) == 0:
        return 0.0
    return float(np.mean((true >= low) & (true <= high)))


def interval_width(lower: Iterable[float], upper: Iterable[float]) -> float:
    low, high = _finite_arrays(lower, upper)
    return float(np.mean(np.maximum(0.0, high - low)))


def normalized_interval_width(y_true: Iterable[float], lower: Iterable[float], upper: Iterable[float]) -> float:
    true, low, high = _finite_arrays(y_true, lower, upper)
    denom = float(np.mean(np.abs(true)))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    return float(np.mean(np.maximum(0.0, high - low)) / denom)


def interval_score(y_true: Iterable[float], lower: Iterable[float], upper: Iterable[float], alpha: float) -> float:
    true, low, high = _finite_arrays(y_true, lower, upper)
    width = np.maximum(0.0, high - low)
    lower_penalty = (2.0 / alpha) * np.maximum(low - true, 0.0)
    upper_penalty = (2.0 / alpha) * np.maximum(true - high, 0.0)
    return float(np.mean(width + lower_penalty + upper_penalty))
