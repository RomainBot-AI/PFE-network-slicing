#!/usr/bin/env python3
"""Build the subnet/slice forecasting panel from the clustered dataset.

Reproduces the benchmark panel. For the dense 14-day reference panel used by the
benchmarks, run with ``--dense`` and an output named
``subnet_slice_traffic_min2016_dense.csv``.
"""

from __future__ import annotations

import argparse

from nsf.preprocessing.subnet_panel import build_subnet_slice_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="simulation/mininet/cesnet_points_clustered_4slices.csv",
        help="Clustered dataset produced by dataset-prep/cluster_4_slices.py.",
    )
    parser.add_argument(
        "--relationship-csv",
        default="data/reference/ids_relationship.csv",
        help="CESNET mapping with id_ip, id_institution, id_institution_subnet.",
    )
    parser.add_argument(
        "--output-csv",
        default="forecasting/data/subnet_slice_traffic_10min_long.csv",
        help="Long subnet/slice panel CSV to create.",
    )
    parser.add_argument(
        "--slice-output-csv",
        default="forecasting/data/slice_traffic_from_subnet_10min.csv",
        help="Wide per-slice aggregate rebuilt from the subnet/slice series.",
    )
    parser.add_argument("--target", default="n_bytes")
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug row limit.")
    parser.add_argument(
        "--min-total-points",
        type=int,
        default=0,
        help="Keep only series with at least N observed timestamps (0 = keep all).",
    )
    parser.add_argument(
        "--tail-timestamps",
        type=int,
        default=0,
        help="Keep only the last N timestamps after aggregation (0 = full period).",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Reindex series on the regular grid and fill missing points with zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel, slice_wide, summary = build_subnet_slice_series(
        input_csv=args.input_csv,
        relationship_csv=args.relationship_csv,
        output_csv=args.output_csv,
        slice_output_csv=args.slice_output_csv,
        target=args.target,
        freq=args.freq,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        min_total_points=args.min_total_points,
        tail_timestamps=args.tail_timestamps,
        dense=args.dense,
    )
    print(f"Wrote {args.output_csv}")
    print(f"panel_rows={len(panel)} series={panel['unique_id'].nunique()}")
    print(f"period={panel['ds'].min()} -> {panel['ds'].max()}")
    print(f"Wrote {args.slice_output_csv}")
    print(f"slice_rows={len(slice_wide)} columns={list(slice_wide.columns)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
