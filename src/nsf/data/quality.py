"""Quality checks for regular traffic time series."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesQualityReport:
    rows: int
    duplicate_timestamps: int
    missing_timestamps: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None


def audit_regular_index(index: pd.DatetimeIndex, freq: str = "10min") -> TimeSeriesQualityReport:
    """Audit duplicate and missing timestamps without changing the data."""

    clean_index = pd.DatetimeIndex(index).dropna().sort_values()
    if clean_index.empty:
        return TimeSeriesQualityReport(0, 0, 0, None, None)

    duplicate_timestamps = int(clean_index.duplicated().sum())
    unique_index = pd.DatetimeIndex(clean_index.unique()).sort_values()
    expected = pd.date_range(unique_index.min(), unique_index.max(), freq=freq, tz=unique_index.tz)
    missing_timestamps = int(len(expected.difference(unique_index)))
    return TimeSeriesQualityReport(
        rows=len(index),
        duplicate_timestamps=duplicate_timestamps,
        missing_timestamps=missing_timestamps,
        start=unique_index.min(),
        end=unique_index.max(),
    )
