from __future__ import annotations

import pandas as pd

from nsf.allocation.forecast_export import bytes_per_step_to_mbps, export_slice_forecast


def test_bytes_per_step_to_mbps() -> None:
    values = pd.Series([75_000_000.0])

    converted = bytes_per_step_to_mbps(values, step_seconds=600.0)

    assert converted.iloc[0] == 1.0


def test_export_slice_forecast_aggregates_subnets(tmp_path) -> None:
    predictions = pd.DataFrame(
        [
            {"timestamp": "2024-01-01 00:00:00", "slice": "URLLC", "q10": 10.0, "q50": 20.0, "q90": 30.0},
            {"timestamp": "2024-01-01 00:00:00", "slice": "URLLC", "q10": 1.0, "q50": 2.0, "q90": 3.0},
            {"timestamp": "2024-01-01 00:00:00", "slice": "eMBB", "q10": 75_000_000.0, "q50": 75_000_000.0, "q90": 75_000_000.0},
        ]
    )
    input_csv = tmp_path / "predictions.csv"
    output_csv = tmp_path / "forecast.csv"
    predictions.to_csv(input_csv, index=False)

    export_slice_forecast(input_csv, output_csv)

    exported = pd.read_csv(output_csv)
    urllc = exported[exported["slice"] == "URLLC"].iloc[0]
    embb = exported[exported["slice"] == "eMBB"].iloc[0]
    assert urllc["q10_bytes"] == 11.0
    assert urllc["q50_bytes"] == 22.0
    assert urllc["q90_bytes"] == 33.0
    assert embb["q90_mbps"] == 1.0
