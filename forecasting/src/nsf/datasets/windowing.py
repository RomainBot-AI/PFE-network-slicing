"""Model-independent windowing helpers."""

from __future__ import annotations

import numpy as np


def make_supervised_windows(values: np.ndarray, input_size: int, horizon: int, step_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if min(input_size, horizon, step_size) <= 0:
        raise ValueError("input_size, horizon, and step_size must be positive")
    x_rows = []
    y_rows = []
    for start in range(0, len(values) - input_size - horizon + 1, step_size):
        x_rows.append(values[start : start + input_size])
        y_rows.append(values[start + input_size : start + input_size + horizon])
    return np.asarray(x_rows), np.asarray(y_rows)
