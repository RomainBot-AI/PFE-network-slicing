"""Train-only scaling helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LogZScoreParams:
    mean: float
    std: float


def fit_log_zscore(train: pd.Series) -> LogZScoreParams:
    values = np.log1p(train.to_numpy(dtype=float))
    mean = float(values.mean())
    std = float(values.std())
    if std == 0.0:
        std = 1.0
    return LogZScoreParams(mean=mean, std=std)


def transform_log_zscore(values: pd.Series | np.ndarray, params: LogZScoreParams) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    return (np.log1p(raw) - params.mean) / params.std


def inverse_log_zscore(values: pd.Series | np.ndarray, params: LogZScoreParams) -> np.ndarray:
    scaled = np.asarray(values, dtype=float)
    return np.expm1(scaled * params.std + params.mean)
