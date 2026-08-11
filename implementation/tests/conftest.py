"""Shared fixtures: small synthetic panels with the real schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SLICES = ["URLLC", "URLLC_eMBB_MIX", "eMBB", "mMTC"]
_SCALE = {"URLLC": 2e6, "URLLC_eMBB_MIX": 3e6, "eMBB": 2e7, "mMTC": 1e6}


def _make_long(n_steps: int = 240, subnets=(0, 1), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_steps, freq="10min")
    daily = 1 + 0.6 * np.sin(np.arange(n_steps) * 2 * np.pi / 144)
    rows = []
    for sub in subnets:
        for s in SLICES:
            y = np.clip(_SCALE[s] * (1 + 0.5 * sub) * daily * rng.uniform(0.7, 1.3, n_steps), 0, None)
            rows.extend((t, sub, s, float(v)) for t, v in zip(ts, y))
    return pd.DataFrame(rows, columns=["ds", "id_institution_subnet", "slice", "y"])


@pytest.fixture
def synthetic_pivoted() -> pd.DataFrame:
    """Dense pivoted panel (one column per slice) for a single station."""
    long_df = _make_long(subnets=(0,))
    return (
        long_df.pivot_table(index=["ds", "id_institution_subnet"], columns="slice", values="y", fill_value=0.0)
        .reset_index()
        .sort_values("ds")
        .reset_index(drop=True)
    )


@pytest.fixture
def synthetic_csv(tmp_path) -> str:
    """Path to a small raw long-format panel CSV with two subnets."""
    path = tmp_path / "panel.csv"
    _make_long(subnets=(0, 1)).to_csv(path, index=False)
    return str(path)
