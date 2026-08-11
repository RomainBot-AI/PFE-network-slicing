#!/usr/bin/env python3
"""Run the local Prophet benchmark."""

from __future__ import annotations

import argparse

from nsf.benchmark.prophet_benchmark import load_prophet_benchmark_config, run_prophet_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="forecasting/configs/experiment/prophet_benchmark.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_prophet_benchmark(load_prophet_benchmark_config(args.config))
    print("Prophet benchmark outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
