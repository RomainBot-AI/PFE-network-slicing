#!/usr/bin/env python3
"""Generate a reproducible EDA report for the subnet/slice panel."""

from __future__ import annotations

import argparse

from nsf.eda.panel import run_panel_eda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv",
        help="Dense subnet/slice panel with unique_id, ds, y, slice, id_institution, id_institution_subnet.",
    )
    parser.add_argument("--output-dir", default="traffic_forecasting/reports")
    parser.add_argument("--report-md", default="traffic_forecasting/reports/subnet_slice_eda.md")
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--autocorr-lags", default="1,6,12,36,144,1008")
    parser.add_argument("--autocorr-top-n-per-slice", type=int, default=10)
    return parser.parse_args()


def parse_lags(value: str) -> tuple[int, ...]:
    lags = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not lags:
        raise ValueError("Expected at least one autocorrelation lag")
    return lags


def main() -> None:
    args = parse_args()
    result = run_panel_eda(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        report_md=args.report_md,
        freq=args.freq,
        autocorr_lags=parse_lags(args.autocorr_lags),
        autocorr_top_n_per_slice=args.autocorr_top_n_per_slice,
    )
    print(f"Wrote {args.report_md}")
    print(
        "Panel EDA: "
        f"rows={result.overview['rows']} "
        f"series={result.overview['series']} "
        f"subnets={result.overview['subnets']} "
        f"timestamps={result.overview['timestamps']}"
    )


if __name__ == "__main__":
    main()
