"""Rolling-origin split helpers.

This module keeps the split protocol independent of any model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WindowBounds:
    split: str
    window_pos: int
    input_start_idx: int
    input_end_idx: int
    target_start_idx: int
    target_end_idx: int


def chronological_split(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if len(df) <= test_size:
        raise ValueError(f"Series length {len(df)} must be greater than test_size {test_size}")
    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()


def make_rolling_windows(
    total_points: int,
    input_size: int,
    horizon: int,
    step_size: int,
) -> list[WindowBounds]:
    if min(total_points, input_size, horizon, step_size) <= 0:
        raise ValueError("total_points, input_size, horizon, and step_size must be positive")
    span = input_size + horizon
    if total_points < span:
        return []

    windows = []
    for window_pos, start_idx in enumerate(range(0, total_points - span + 1, step_size)):
        windows.append(
            WindowBounds(
                split="unspecified",
                window_pos=window_pos,
                input_start_idx=start_idx,
                input_end_idx=start_idx + input_size - 1,
                target_start_idx=start_idx + input_size,
                target_end_idx=start_idx + input_size + horizon - 1,
            )
        )
    return windows
