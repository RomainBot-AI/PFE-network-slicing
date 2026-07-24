#!/usr/bin/env python3
"""Tune PatchTST on a small explicit candidate set."""

from __future__ import annotations

import argparse

from nsf.benchmark.patchtst_tuning import load_patchtst_tuning_config, tune_patchtst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/patchtst_tuning.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = tune_patchtst(load_patchtst_tuning_config(args.config))
    print("PatchTST tuning outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
