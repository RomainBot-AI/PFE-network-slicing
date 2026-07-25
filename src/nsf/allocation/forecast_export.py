"""Export probabilistic subnet forecasts into slice-level simulation inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SECONDS_PER_10MIN = 600.0
BITS_PER_BYTE = 8.0
MBIT = 1_000_000.0


def bytes_per_step_to_mbps(values: pd.Series, step_seconds: float = SECONDS_PER_10MIN) -> pd.Series:
    """Convert bytes observed over one forecast step to megabits per second."""
    return values.astype(float) * BITS_PER_BYTE / step_seconds / MBIT


def export_slice_forecast(
    predictions_csv: str | Path,
    output_csv: str | Path,
    step_seconds: float = SECONDS_PER_10MIN,
) -> Path:
    """Aggregate subnet/slice probabilistic predictions to slice-level demand."""
    predictions = pd.read_csv(predictions_csv)
    required = {"timestamp", "slice", "q10", "q50", "q90"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns in {predictions_csv}: {sorted(missing)}")

    grouped = (
        predictions.groupby(["timestamp", "slice"], as_index=False)[["q10", "q50", "q90"]]
        .sum()
        .sort_values(["timestamp", "slice"])
    )
    grouped = grouped.rename(columns={"q10": "q10_bytes", "q50": "q50_bytes", "q90": "q90_bytes"})
    for quantile in ("q10", "q50", "q90"):
        grouped[f"{quantile}_mbps"] = bytes_per_step_to_mbps(grouped[f"{quantile}_bytes"], step_seconds=step_seconds)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)
    return output_path
