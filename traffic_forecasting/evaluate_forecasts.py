#!/usr/bin/env python3
"""Summarize exported forecasting metrics."""

from __future__ import annotations

import argparse

import pandas as pd

try:
    from traffic_forecasting.common import ensure_parent
except ModuleNotFoundError:
    from common import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        nargs="+",
        default=["traffic_forecasting/reports/metrics.csv"],
        help="One or more metrics CSV files to rank together.",
    )
    parser.add_argument("--output-csv", default="traffic_forecasting/reports/metrics_ranked.csv")
    parser.add_argument(
        "--group-by-evaluation",
        action="store_true",
        help="Rank separately per evaluation label. By default, rank all comparable metrics by slice/horizon.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for metrics_csv in args.metrics_csv:
        frame = pd.read_csv(metrics_csv)
        frame["source_file"] = metrics_csv
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    group_cols = ["slice", "horizon"]
    sort_cols = ["slice", "horizon", "RMSE"]
    if args.group_by_evaluation and "evaluation" in metrics.columns:
        group_cols.append("evaluation")
        sort_cols = ["slice", "horizon", "evaluation", "RMSE"]
    ranked = metrics.sort_values(sort_cols).copy()
    ranked["rank_by_rmse"] = ranked.groupby(group_cols)["RMSE"].rank(method="dense")
    output_path = ensure_parent(args.output_csv)
    ranked.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    print(ranked.to_string(index=False))


if __name__ == "__main__":
    main()
