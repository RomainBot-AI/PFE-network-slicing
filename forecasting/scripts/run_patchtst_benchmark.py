#!/usr/bin/env python3
"""Run the per-slice PatchTST benchmark."""

from __future__ import annotations

import argparse

from nsf.benchmark.patchtst_benchmark import load_patchtst_benchmark_config, run_patchtst_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="forecasting/configs/experiment/patchtst_benchmark.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_patchtst_benchmark(load_patchtst_benchmark_config(args.config))
    print("PatchTST benchmark outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
