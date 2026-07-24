#!/usr/bin/env python3
"""Run the per-slice LSTM benchmark with tuned parameters."""

from __future__ import annotations

import argparse

from nsf.benchmark.lstm_benchmark import load_lstm_benchmark_config, run_lstm_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/lstm_benchmark.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_lstm_benchmark(load_lstm_benchmark_config(args.config))
    print("LSTM benchmark outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
