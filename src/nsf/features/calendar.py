"""Calendar feature generation."""

from __future__ import annotations

import pandas as pd


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    return pd.DataFrame(
        {
            "hour": idx.hour,
            "dayofweek": idx.dayofweek,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
        },
        index=idx,
    )
