#!/usr/bin/env python3
"""Run a configurable rolling-origin backtest on the subnet/slice panel."""

from __future__ import annotations

import argparse

from nsf.backtest.panel_engine import run_panel_backtest
from nsf.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="forecasting/configs/experiment/subnet_slice_baseline_backtest.yaml",
        help="YAML or JSON experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_panel_backtest(load_experiment_config(args.config))
    print("Backtest outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
