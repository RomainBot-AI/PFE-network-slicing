"""Loading functions for traffic forecasting datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_panel(path: str | Path) -> pd.DataFrame:
    """Read a long panel with unique_id, ds, y, and slice columns."""

    df = pd.read_csv(path, dtype={"unique_id": str, "slice": str})
    required = {"unique_id", "ds", "y", "slice"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True).dt.tz_convert(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["unique_id", "ds", "y", "slice"])
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
