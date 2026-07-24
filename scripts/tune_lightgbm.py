#!/usr/bin/env python3
"""Tune LightGBM hyperparameters by slice with bounded random search."""

from __future__ import annotations

import argparse

from nsf.benchmark.lightgbm_tuning import load_lightgbm_tuning_config, tune_lightgbm_by_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/lightgbm_tuning.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = tune_lightgbm_by_slice(load_lightgbm_tuning_config(args.config))
    print("LightGBM tuning outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
