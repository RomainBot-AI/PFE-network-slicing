#!/usr/bin/env python3
"""Merge benchmark summaries for input-history sensitivity runs."""

from __future__ import annotations

import argparse

from nsf.visualization.history_sensitivity import build_history_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_history_sensitivity(args.output_dir)
    print("History sensitivity outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
