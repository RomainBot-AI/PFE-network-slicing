#!/usr/bin/env python3
"""Merge benchmark summaries across retained model runs."""

from __future__ import annotations

import argparse

from nsf.visualization.model_comparison import build_model_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_model_comparison(args.output_dir)
    print("Model comparison outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
