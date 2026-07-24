#!/usr/bin/env python3
"""Tune per-slice N-HiTS hyperparameters with Optuna."""

from __future__ import annotations

import argparse

from nsf.benchmark.nhits_tuning import load_nhits_tuning_config, tune_nhits_by_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/nhits_tuning.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = tune_nhits_by_slice(load_nhits_tuning_config(args.config))
    print("N-HiTS tuning outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
