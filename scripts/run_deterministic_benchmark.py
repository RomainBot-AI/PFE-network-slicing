#!/usr/bin/env python3
"""Run the deterministic subnet/slice forecasting benchmark."""

from __future__ import annotations

import argparse

from nsf.benchmark.deterministic import load_benchmark_config, run_deterministic_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment/deterministic_benchmark.yaml",
        help="YAML benchmark configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_deterministic_benchmark(load_benchmark_config(args.config))
    print("Benchmark outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
