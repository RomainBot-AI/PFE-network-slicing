#!/usr/bin/env python3
"""Prepare fold-aware feature datasets for subnet/slice forecasting."""

from __future__ import annotations

import argparse

from nsf.preprocessing.panel_dataset import load_preprocess_config, prepare_panel_feature_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="forecasting/configs/experiment/subnet_slice_preprocess.yaml",
        help="YAML preprocessing configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = prepare_panel_feature_dataset(load_preprocess_config(args.config))
    print("Preprocessing outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
