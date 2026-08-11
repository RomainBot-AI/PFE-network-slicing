#!/usr/bin/env python3
"""Build a standalone HTML EDA report from generated subnet/slice EDA tables."""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="forecasting/reports")
    parser.add_argument("--output-html", default="forecasting/forecasting/reports/subnet_slice_eda.html")
    parser.add_argument("--panel-path", default="forecasting/data/subnet_slice_traffic_min2016_dense.csv")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    from nsf.visualization.eda_html import build_html_from_report_tables

    args = parse_args()
    output = build_html_from_report_tables(
        report_dir=args.report_dir,
        output_html=args.output_html,
        panel_path=args.panel_path,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
