#!/usr/bin/env python3
"""Build a compact traffic time series per slice from the sliced CESNET CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from traffic_forecasting.common import SLICES, ensure_parent
except ModuleNotFoundError:
    from common import SLICES, ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="simulation/mininet/cesnet_points_clustered_4slices.csv",
        help="Sliced dataset produced by Dataset Preparing/cluster_4_slices.py.",
    )
    parser.add_argument(
        "--output-csv",
        default="traffic_forecasting/data/slice_traffic_10min.csv",
        help="Compact wide time-series CSV to create.",
    )
    parser.add_argument("--freq", default="10min", help="Pandas resampling frequency.")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug limit.")
    return parser.parse_args()


def build_slice_series(
    input_csv: str | Path,
    output_csv: str | Path,
    freq: str,
    chunksize: int,
    max_rows: int | None,
) -> pd.DataFrame:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    required = ["timestamp", "slice", "n_bytes"]
    totals = []
    rows_seen = 0

    for chunk in pd.read_csv(input_path, usecols=required, chunksize=chunksize):
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]

        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
        chunk["n_bytes"] = pd.to_numeric(chunk["n_bytes"], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp", "slice", "n_bytes"])
        grouped = chunk.groupby(["timestamp", "slice"], as_index=False)["n_bytes"].sum()
        totals.append(grouped)
        rows_seen += len(chunk)

    if not totals:
        raise ValueError("No valid rows found in input CSV")

    aggregated = pd.concat(totals, ignore_index=True)
    aggregated = aggregated.groupby(["timestamp", "slice"], as_index=False)["n_bytes"].sum()

    wide = (
        aggregated.pivot(index="timestamp", columns="slice", values="n_bytes")
        .sort_index()
        .fillna(0.0)
    )
    for slice_name in SLICES:
        if slice_name not in wide.columns:
            wide[slice_name] = 0.0

    wide = wide[SLICES]
    wide = wide.resample(freq).sum().fillna(0.0)
    wide = wide.reset_index()

    output_path = ensure_parent(output_csv)
    wide.to_csv(output_path, index=False)
    return wide


def main() -> None:
    args = parse_args()
    df = build_slice_series(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        freq=args.freq,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )
    print(f"Wrote {args.output_csv}")
    print(f"rows={len(df)} columns={list(df.columns)}")
    print(f"period={df['timestamp'].min()} -> {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
