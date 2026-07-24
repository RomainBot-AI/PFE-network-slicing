"""Panel rolling-origin fold generation and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PanelFold:
    fold: int
    train_start_idx: int
    train_end_idx: int
    target_start_idx: int
    target_end_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    target_start: pd.Timestamp
    target_end: pd.Timestamp


def make_panel_folds(
    timestamps: pd.DatetimeIndex,
    input_size: int,
    horizon: int,
    n_folds: int,
    fold_stride: int,
    expanding: bool = False,
) -> list[PanelFold]:
    """Create final rolling-origin folds shared by every series.

    Folds are placed near the end of the series and ordered chronologically.
    Each target block starts strictly after its training context.
    """

    if min(len(timestamps), input_size, horizon, n_folds, fold_stride) <= 0:
        raise ValueError("timestamps, input_size, horizon, n_folds, and fold_stride must be positive")
    max_target_start = len(timestamps) - horizon
    first_target_start = max_target_start - (n_folds - 1) * fold_stride
    if first_target_start < input_size:
        raise ValueError(
            "Not enough timestamps for requested folds: "
            f"need first_target_start >= input_size, got {first_target_start} < {input_size}"
        )

    folds = []
    for fold_idx in range(n_folds):
        target_start_idx = first_target_start + fold_idx * fold_stride
        target_end_idx = target_start_idx + horizon - 1
        train_end_idx = target_start_idx - 1
        train_start_idx = 0 if expanding else target_start_idx - input_size
        folds.append(
            PanelFold(
                fold=fold_idx,
                train_start_idx=train_start_idx,
                train_end_idx=train_end_idx,
                target_start_idx=target_start_idx,
                target_end_idx=target_end_idx,
                train_start=timestamps[train_start_idx],
                train_end=timestamps[train_end_idx],
                target_start=timestamps[target_start_idx],
                target_end=timestamps[target_end_idx],
            )
        )
    return folds


def folds_to_frame(folds: list[PanelFold]) -> pd.DataFrame:
    return pd.DataFrame([fold.__dict__ for fold in folds])


def leakage_audit(folds: list[PanelFold]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        train_positions = set(range(fold.train_start_idx, fold.train_end_idx + 1))
        target_positions = set(range(fold.target_start_idx, fold.target_end_idx + 1))
        rows.append(
            {
                "fold": fold.fold,
                "train_ends_before_target": fold.train_end_idx < fold.target_start_idx,
                "train_target_overlap_points": len(train_positions.intersection(target_positions)),
                "input_size": fold.train_end_idx - fold.train_start_idx + 1,
                "horizon": fold.target_end_idx - fold.target_start_idx + 1,
            }
        )
    return pd.DataFrame(rows)
