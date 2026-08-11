#!/usr/bin/env python3
"""Run the per-slice N-HiTS benchmark."""

from __future__ import annotations

import argparse

from nsf.benchmark.nhits_benchmark import load_nhits_benchmark_config, run_nhits_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="forecasting/configs/experiment/nhits_benchmark.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_nhits_benchmark(load_nhits_benchmark_config(args.config))
    print("N-HiTS benchmark outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
