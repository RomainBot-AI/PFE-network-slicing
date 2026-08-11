#!/usr/bin/env python3
"""Export probabilistic benchmark predictions to a PPO simulation input CSV."""

from __future__ import annotations

import argparse

from nsf.allocation.forecast_export import SECONDS_PER_10MIN, export_slice_forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-csv",
        default="forecasting/experiments/runs/probabilistic_lightgbm_14d/predictions_probabilistic.csv",
        help="Input subnet/slice probabilistic predictions CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="simulation/forecast_inputs/slice_demand_forecast_lightgbm_q90.csv",
        help="Output slice-level simulation forecast CSV.",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=SECONDS_PER_10MIN,
        help="Forecast step duration used to convert bytes per step to Mbps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_slice_forecast(args.predictions_csv, args.output_csv, step_seconds=args.step_seconds)
    print(f"simulation forecast: {output}")


if __name__ == "__main__":
    main()
