#!/usr/bin/env python3
"""Tune per-slice LSTM hyperparameters with Optuna."""

from __future__ import annotations

import argparse

from nsf.benchmark.lstm_tuning import load_lstm_tuning_config, tune_lstm_by_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="forecasting/configs/experiment/lstm_tuning.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = tune_lstm_by_slice(load_lstm_tuning_config(args.config))
    print("LSTM tuning outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
