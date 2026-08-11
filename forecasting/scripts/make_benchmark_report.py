#!/usr/bin/env python3
"""Build a standalone HTML report for a deterministic benchmark run."""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="forecasting/experiments/runs/deterministic_benchmark")
    parser.add_argument("--output-html", default=None)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    from nsf.visualization.benchmark_html import build_benchmark_html

    args = parse_args()
    output = build_benchmark_html(args.run_dir, args.output_html)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
