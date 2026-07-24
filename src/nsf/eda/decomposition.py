"""EDA placeholders for seasonality/decomposition functions."""

from __future__ import annotations

import pandas as pd


def describe_by_slice(series: pd.DataFrame) -> pd.DataFrame:
    return series.describe().T
