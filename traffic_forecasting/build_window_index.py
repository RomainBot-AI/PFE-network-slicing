#!/usr/bin/env python3
"""Build explicit train/validation/test window indices for panel forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from traffic_forecasting.common import ensure_parent
except ModuleNotFoundError:
    from common import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-csv",
        default="traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv",
        help="Dense panel CSV with unique_id, ds, y, slice.",
    )
    parser.add_argument(
        "--output-csv",
        default="traffic_forecasting/reports/subnet_slice_window_index_2016_36_stride144.csv",
        help="Window index CSV to write.",
    )
    parser.add_argument("--input-size", type=int, default=2016, help="History length in time steps.")
    parser.add_argument("--horizon", type=int, default=36, help="Forecast horizon length in time steps.")
    parser.add_argument("--stride", type=int, default=144, help="Stride between window starts, in time steps.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of eligible chronological windows assigned to validation.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of eligible chronological windows assigned to test.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("target_chronological", "disjoint_blocks", "rotating_blocks"),
        default="target_chronological",
        help=(
            "target_chronological splits eligible windows by target chronology; "
            "disjoint_blocks first splits timestamps into train/val/test blocks and "
            "keeps only windows fully contained in one block; rotating_blocks splits "
            "time into complete blocks and rotates train/val/test assignment by group."
        ),
    )
    parser.add_argument(
        "--split-block-size",
        type=int,
        default=0,
        help="Block size in time steps for rotating_blocks. Use 0 for input_size+horizon.",
    )
    parser.add_argument(
        "--block-pattern",
        default="train,train,val,train,train,test",
        help="Comma-separated split pattern for rotating_blocks.",
    )
    parser.add_argument(
        "--rotate-by",
        default="id_institution_subnet",
        help="Metadata column used to rotate block assignment in rotating_blocks.",
    )
    return parser.parse_args()


def read_panel_metadata(path: str | Path) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    df = pd.read_csv(path, usecols=lambda col: col != "y")
    required = {"unique_id", "ds", "slice"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True).dt.tz_convert(None)
    df["unique_id"] = df["unique_id"].astype(str)
    df["slice"] = df["slice"].astype(str)
    df = df.dropna(subset=["unique_id", "ds", "slice"])
    timestamps = pd.DatetimeIndex(sorted(df["ds"].unique()))
    metadata_cols = [col for col in df.columns if col != "ds"]
    metadata = df[metadata_cols].drop_duplicates("unique_id").sort_values("unique_id")
    return metadata, timestamps


def split_for_position(position: int, total: int, val_fraction: float, test_fraction: float) -> str:
    test_count = int(round(total * test_fraction))
    val_count = int(round(total * val_fraction))
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError("Split fractions leave no training windows")
    if position < train_count:
        return "train"
    if position < train_count + val_count:
        return "val"
    return "test"


def timestamp_blocks(total_points: int, val_fraction: float, test_fraction: float) -> dict[str, tuple[int, int]]:
    test_points = int(round(total_points * test_fraction))
    val_points = int(round(total_points * val_fraction))
    train_points = total_points - val_points - test_points
    if train_points <= 0 or val_points <= 0 or test_points <= 0:
        raise ValueError("Split fractions must leave non-empty train/val/test blocks")
    return {
        "train": (0, train_points - 1),
        "val": (train_points, train_points + val_points - 1),
        "test": (train_points + val_points, total_points - 1),
    }


def split_for_complete_window(
    input_start_idx: int,
    target_end_idx: int,
    blocks: dict[str, tuple[int, int]],
) -> str | None:
    for split, (start, end) in blocks.items():
        if input_start_idx >= start and target_end_idx <= end:
            return split
    return None


def parse_block_pattern(value: str) -> list[str]:
    pattern = [part.strip() for part in value.split(",") if part.strip()]
    valid = {"train", "val", "test"}
    if not pattern or any(part not in valid for part in pattern):
        raise ValueError(f"--block-pattern must contain only {sorted(valid)}")
    if "train" not in pattern or "val" not in pattern or "test" not in pattern:
        raise ValueError("--block-pattern must include train, val, and test")
    return pattern


def group_offsets(metadata: pd.DataFrame, rotate_by: str, pattern_len: int) -> dict[str, int]:
    if rotate_by not in metadata.columns:
        raise ValueError(f"--rotate-by column '{rotate_by}' is missing from panel metadata")
    groups = sorted(metadata[rotate_by].astype(str).dropna().unique())
    return {group: idx % pattern_len for idx, group in enumerate(groups)}


def append_window_rows(
    rows: list[dict],
    metadata: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    split: str,
    window_pos: int,
    input_start_idx: int,
    input_size: int,
    horizon: int,
    stride: int,
    split_mode: str,
) -> None:
    input_end_idx = input_start_idx + input_size - 1
    target_start_idx = input_start_idx + input_size
    target_end_idx = input_start_idx + input_size + horizon - 1

    for meta in metadata.itertuples(index=False):
        row = meta._asdict()
        row.update(
            {
                "split": split,
                "window_pos": window_pos,
                "input_start_idx": input_start_idx,
                "input_end_idx": input_end_idx,
                "target_start_idx": target_start_idx,
                "target_end_idx": target_end_idx,
                "input_start": timestamps[input_start_idx],
                "input_end": timestamps[input_end_idx],
                "target_start": timestamps[target_start_idx],
                "target_end": timestamps[target_end_idx],
                "input_size": input_size,
                "horizon": horizon,
                "stride": stride,
                "split_mode": split_mode,
            }
        )
        rows.append(row)


def build_window_index(
    panel_csv: str | Path,
    output_csv: str | Path,
    input_size: int,
    horizon: int,
    stride: int,
    val_fraction: float,
    test_fraction: float,
    split_mode: str,
    split_block_size: int,
    block_pattern: str,
    rotate_by: str,
) -> pd.DataFrame:
    metadata, timestamps = read_panel_metadata(panel_csv)
    total_points = len(timestamps)
    window_span = input_size + horizon
    if total_points < window_span:
        raise ValueError(f"Need at least {window_span} timestamps, found {total_points}")

    starts = list(range(0, total_points - window_span + 1, stride))
    blocks = timestamp_blocks(total_points, val_fraction, test_fraction)
    rows = []
    window_pos = 0

    if split_mode in {"target_chronological", "disjoint_blocks"}:
        for start_idx in starts:
            target_end_idx = start_idx + input_size + horizon - 1
            if split_mode == "target_chronological":
                split = split_for_position(window_pos, len(starts), val_fraction, test_fraction)
            else:
                split = split_for_complete_window(start_idx, target_end_idx, blocks)
                if split is None:
                    window_pos += 1
                    continue
            append_window_rows(
                rows=rows,
                metadata=metadata,
                timestamps=timestamps,
                split=split,
                window_pos=window_pos,
                input_start_idx=start_idx,
                input_size=input_size,
                horizon=horizon,
                stride=stride,
                split_mode=split_mode,
            )
            window_pos += 1

    else:
        pattern = parse_block_pattern(block_pattern)
        block_size = split_block_size if split_block_size > 0 else window_span
        if block_size < window_span:
            raise ValueError("--split-block-size must be at least input_size+horizon")
        offsets = group_offsets(metadata, rotate_by, len(pattern))
        for block_idx, block_start in enumerate(range(0, total_points - window_span + 1, block_size)):
            block_end = min(block_start + block_size - 1, total_points - 1)
            max_start = block_end - window_span + 1
            if max_start < block_start:
                continue
            block_starts = range(block_start, max_start + 1, stride)
            for start_idx in block_starts:
                for group_value, group_meta in metadata.groupby(metadata[rotate_by].astype(str), sort=False):
                    split = pattern[(block_idx + offsets[str(group_value)]) % len(pattern)]
                    append_window_rows(
                        rows=rows,
                        metadata=group_meta,
                        timestamps=timestamps,
                        split=split,
                        window_pos=window_pos,
                        input_start_idx=start_idx,
                        input_size=input_size,
                        horizon=horizon,
                        stride=stride,
                        split_mode=split_mode,
                    )
                window_pos += 1

    windows = pd.DataFrame(rows)
    windows.to_csv(ensure_parent(output_csv), index=False)
    return windows


def summarize(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_summary = (
        windows.groupby("split")
        .agg(
            windows=("window_pos", "nunique"),
            examples=("window_pos", "size"),
            series=("unique_id", "nunique"),
            target_start_min=("target_start", "min"),
            target_end_max=("target_end", "max"),
        )
        .reset_index()
    )
    slice_summary = (
        windows.groupby(["split", "slice"])
        .agg(
            windows=("window_pos", "nunique"),
            examples=("window_pos", "size"),
            series=("unique_id", "nunique"),
        )
        .reset_index()
    )
    return split_summary, slice_summary


def main() -> None:
    args = parse_args()
    windows = build_window_index(
        panel_csv=args.panel_csv,
        output_csv=args.output_csv,
        input_size=args.input_size,
        horizon=args.horizon,
        stride=args.stride,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_mode=args.split_mode,
        split_block_size=args.split_block_size,
        block_pattern=args.block_pattern,
        rotate_by=args.rotate_by,
    )
    split_summary, slice_summary = summarize(windows)
    print(f"Wrote {args.output_csv}")
    print(f"rows={len(windows)} series={windows['unique_id'].nunique()} windows={windows['window_pos'].nunique()}")
    print("\nSplit summary")
    print(split_summary.to_string(index=False))
    print("\nSlice summary")
    print(slice_summary.to_string(index=False))


if __name__ == "__main__":
    main()
