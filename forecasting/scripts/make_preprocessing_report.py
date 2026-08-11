#!/usr/bin/env python3
"""Build the preprocessing HTML report."""

from __future__ import annotations

import argparse

from nsf.visualization.preprocessing_html import build_preprocessing_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed/subnet_slice_preprocess")
    parser.add_argument("--output-html", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_preprocessing_html(args.processed_dir, args.output_html)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
