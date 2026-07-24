#!/usr/bin/env python3
"""Tune Prophet on a small explicit grid."""

from __future__ import annotations

import argparse

from nsf.benchmark.prophet_tuning import load_prophet_tuning_config, tune_prophet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/prophet_tuning.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = tune_prophet(load_prophet_tuning_config(args.config))
    print("Prophet tuning outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
