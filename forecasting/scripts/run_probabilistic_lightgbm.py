#!/usr/bin/env python3
"""Run the LightGBM quantile probabilistic benchmark."""

from __future__ import annotations

import argparse

from nsf.benchmark.probabilistic_lightgbm import (
    load_probabilistic_lightgbm_config,
    run_probabilistic_lightgbm_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="forecasting/configs/experiment/probabilistic_lightgbm_14d.yaml",
        help="YAML benchmark configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_probabilistic_lightgbm_benchmark(load_probabilistic_lightgbm_config(args.config))
    print("Probabilistic LightGBM outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
