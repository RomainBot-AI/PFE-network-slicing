#!/usr/bin/env python3
"""Audit the sliced dataset before training forecasting models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from traffic_forecasting.common import SLICES, ensure_parent
except ModuleNotFoundError:
    from common import SLICES, ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        default="simulation/mininet/cesnet_points_clustered_4slices.csv",
        help="Sliced source CSV.",
    )
    parser.add_argument(
        "--slice-csv",
        default="traffic_forecasting/data/slice_traffic_10min.csv",
        help="Full slice aggregate CSV.",
    )
    parser.add_argument(
        "--panel-csv",
        default="traffic_forecasting/data/ip_slice_traffic_min2016_real_full_dense.csv",
        help="Optional dense real ip/slice panel CSV to audit.",
    )
    parser.add_argument(
        "--panel-slice-csv",
        default="traffic_forecasting/data/slice_traffic_min2016_real_full_from_ip.csv",
        help="Optional slice aggregate rebuilt from the audited panel.",
    )
    parser.add_argument(
        "--output-md",
        default="traffic_forecasting/reports/data_audit.md",
        help="Markdown report to write.",
    )
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--min-history-points", type=int, default=2016)
    parser.add_argument("--test-size", type=int, default=144)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=2016)
    parser.add_argument("--stride", type=int, default=144)
    return parser.parse_args()


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_\n"
    text_df = df.astype(str)
    headers = list(text_df.columns)
    rows = text_df.to_numpy().tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def audit_source(path: str | Path, freq: str, chunksize: int) -> dict[str, object]:
    required = ["timestamp", "slice", "ip_id", "n_bytes"]
    rows = 0
    valid_rows = 0
    duplicate_rows_after_floor = 0
    timestamp_min = None
    timestamp_max = None
    slice_counts = []
    slice_bytes = []
    series_counts = []
    timestamp_values = set()

    for chunk in pd.read_csv(path, usecols=required, chunksize=chunksize):
        rows += len(chunk)
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
        chunk["ip_id"] = chunk["ip_id"].astype(str)
        chunk["n_bytes"] = pd.to_numeric(chunk["n_bytes"], errors="coerce")
        chunk = chunk.dropna(subset=required)
        chunk = chunk[chunk["slice"].isin(SLICES)].copy()
        valid_rows += len(chunk)
        if chunk.empty:
            continue

        timestamp_min = chunk["timestamp"].min() if timestamp_min is None else min(timestamp_min, chunk["timestamp"].min())
        timestamp_max = chunk["timestamp"].max() if timestamp_max is None else max(timestamp_max, chunk["timestamp"].max())
        chunk["timestamp_floor"] = chunk["timestamp"].dt.floor(freq)
        duplicate_rows_after_floor += int(chunk.duplicated(["timestamp_floor", "slice", "ip_id"]).sum())
        timestamp_values.update(chunk["timestamp_floor"].dropna().unique())

        slice_counts.append(chunk.groupby("slice").size().reset_index(name="rows"))
        slice_bytes.append(chunk.groupby("slice")["n_bytes"].sum().reset_index(name="n_bytes"))
        series_counts.append(
            chunk.drop_duplicates(["timestamp_floor", "slice", "ip_id"])
            .groupby(["slice", "ip_id"])
            .size()
            .reset_index(name="observed_points")
        )

    if not series_counts:
        raise ValueError(f"No valid source rows found in {path}")

    slice_count_df = pd.concat(slice_counts).groupby("slice", as_index=False)["rows"].sum()
    slice_bytes_df = pd.concat(slice_bytes).groupby("slice", as_index=False)["n_bytes"].sum()
    series_df = pd.concat(series_counts).groupby(["slice", "ip_id"], as_index=False)["observed_points"].sum()
    timestamps = pd.DatetimeIndex(sorted(timestamp_values))
    expected_grid = pd.date_range(timestamps.min(), timestamps.max(), freq=freq)
    missing_timestamps = len(expected_grid.difference(timestamps))

    return {
        "rows": rows,
        "valid_rows": valid_rows,
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "grid_start": timestamps.min(),
        "grid_end": timestamps.max(),
        "grid_points": len(timestamps),
        "expected_grid_points": len(expected_grid),
        "missing_timestamps": missing_timestamps,
        "duplicate_rows_after_floor": duplicate_rows_after_floor,
        "slice_counts": slice_count_df,
        "slice_bytes": slice_bytes_df,
        "series": series_df,
    }


def summarize_series(series: pd.DataFrame, thresholds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    for threshold in thresholds:
        overall_rows.append(
            {
                "min_observed_points": threshold,
                "series_count": int((series["observed_points"] >= threshold).sum()),
            }
        )

    per_slice = (
        series.groupby("slice")["observed_points"]
        .agg(["count", "min", "median", "mean", "max"])
        .reset_index()
        .round(2)
    )
    return pd.DataFrame(overall_rows), per_slice


def audit_slice_coverage(full_slice_csv: str | Path, panel_slice_csv: str | Path) -> pd.DataFrame:
    full_path = Path(full_slice_csv)
    panel_path = Path(panel_slice_csv)
    if not full_path.exists() or not panel_path.exists():
        return pd.DataFrame()

    full = pd.read_csv(full_path)
    panel = pd.read_csv(panel_path)
    merged = full.merge(panel, on="timestamp", suffixes=("_full", "_panel"))
    rows = []
    for slice_name in SLICES:
        total = float(merged[f"{slice_name}_full"].sum())
        covered = float(merged[f"{slice_name}_panel"].sum())
        coverage = covered / total * 100.0 if total else 0.0
        max_abs_diff = float((merged[f"{slice_name}_full"] - merged[f"{slice_name}_panel"]).abs().max())
        rows.append(
            {
                "slice": slice_name,
                "covered_pct": round(coverage, 2),
                "missing_pct": round(100.0 - coverage, 2),
                "max_abs_point_diff": max_abs_diff,
            }
        )
    return pd.DataFrame(rows)


def audit_panel(path: str | Path, test_size: int, window_size: int, horizon: int, stride: int) -> dict[str, object] | None:
    panel_path = Path(path)
    if not panel_path.exists():
        return None

    df = pd.read_csv(panel_path, usecols=["unique_id", "slice", "ip_id", "ds", "y"], dtype={"unique_id": str, "slice": str, "ip_id": str})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True).dt.tz_convert(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["unique_id", "slice", "ip_id", "ds", "y"])
    timestamps = pd.DatetimeIndex(sorted(df["ds"].unique()))
    train_ts = timestamps[:-test_size]
    test_start = timestamps[-test_size] if len(timestamps) > test_size else pd.NaT
    train = df[df["ds"].isin(train_ts)]

    counts = train.groupby(["unique_id", "slice"], as_index=False)["ds"].nunique().rename(columns={"ds": "train_points"})
    usable = counts["train_points"] - window_size - horizon + 1
    counts["train_windows"] = ((usable - 1) // stride + 1).where(usable > 0, 0).astype(int)
    per_slice_windows = counts.groupby("slice")["train_windows"].agg(["count", "sum", "min", "max"]).reset_index()
    zero_share = (
        df.assign(is_zero=df["y"] == 0)
        .groupby("slice")["is_zero"]
        .mean()
        .mul(100)
        .reset_index(name="zero_pct")
        .round(2)
    )

    return {
        "rows": len(df),
        "series": df["unique_id"].nunique(),
        "ip_ids": df["ip_id"].nunique(),
        "has_other": bool((df["ip_id"] == "OTHER").any()),
        "timestamps": len(timestamps),
        "period_start": timestamps.min(),
        "period_end": timestamps.max(),
        "test_start": test_start,
        "total_train_windows": int(counts["train_windows"].sum()),
        "per_slice_series": df.groupby("slice")["unique_id"].nunique().reset_index(name="series"),
        "per_slice_windows": per_slice_windows,
        "zero_share": zero_share,
    }


def render_report(args: argparse.Namespace, source: dict[str, object], panel: dict[str, object] | None, coverage: pd.DataFrame) -> str:
    thresholds = [1, 144, 1008, 2016, 5000, 10000, 20000, 30000, 40000]
    threshold_df, per_slice_series = summarize_series(source["series"], thresholds)

    lines = [
        "# Forecasting Data Audit",
        "",
        "## Source Dataset",
        "",
        f"- source: `{args.source_csv}`",
        f"- rows: {source['rows']:,}",
        f"- valid rows: {source['valid_rows']:,}",
        f"- timestamp period raw: {source['timestamp_min']} -> {source['timestamp_max']}",
        f"- floored {args.freq} grid: {source['grid_start']} -> {source['grid_end']}",
        f"- observed grid points: {source['grid_points']:,}",
        f"- expected regular grid points: {source['expected_grid_points']:,}",
        f"- missing timestamps on grid: {source['missing_timestamps']:,}",
        f"- duplicate rows after `{args.freq}` floor by `(timestamp, slice, ip_id)`: {source['duplicate_rows_after_floor']:,}",
        "",
        "### Rows By Slice",
        "",
        md_table(source["slice_counts"]),
        "### Bytes By Slice",
        "",
        md_table(source["slice_bytes"]),
        "### Series Count By Minimum Observed Points",
        "",
        md_table(threshold_df),
        "### Observed Points Per `slice/ip_id`",
        "",
        md_table(per_slice_series),
    ]

    if panel is not None:
        lines.extend(
            [
                "## Dense Panel",
                "",
                f"- panel: `{args.panel_csv}`",
                f"- rows: {panel['rows']:,}",
                f"- real/synthetic series: {panel['series']:,}",
                f"- distinct `ip_id`: {panel['ip_ids']:,}",
                f"- contains synthetic `OTHER`: {panel['has_other']}",
                f"- timestamps: {panel['timestamps']:,}",
                f"- period: {panel['period_start']} -> {panel['period_end']}",
                f"- test start with `test_size={args.test_size}`: {panel['test_start']}",
                f"- training windows with `window_size={args.window_size}`, `horizon={args.horizon}`, `stride={args.stride}`: {panel['total_train_windows']:,}",
                "",
                "### Panel Series By Slice",
                "",
                md_table(panel["per_slice_series"]),
                "### Training Windows By Slice",
                "",
                md_table(panel["per_slice_windows"]),
                "### Zero Share After Densification",
                "",
                md_table(panel["zero_share"]),
            ]
        )

    if not coverage.empty:
        lines.extend(
            [
                "## Coverage Against Full Slice Aggregate",
                "",
                f"- full aggregate: `{args.slice_csv}`",
                f"- panel aggregate: `{args.panel_slice_csv}`",
                "",
                md_table(coverage),
            ]
        )

    lines.extend(
        [
            "## Audit Conclusions",
            "",
            "- Train forecasting models only on real `slice/ip_id` series unless a synthetic residual is explicitly part of a separate experiment.",
            "- Compare panel forecasts against baselines computed on the same retained `ip_id` coverage.",
            "- Keep chronological splits; no generated training window target should overlap the final test interval.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source = audit_source(args.source_csv, args.freq, args.chunksize)
    panel = audit_panel(args.panel_csv, args.test_size, args.window_size, args.horizon, args.stride)
    coverage = audit_slice_coverage(args.slice_csv, args.panel_slice_csv)
    report = render_report(args, source, panel, coverage)
    output_path = ensure_parent(args.output_md)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")
    print(report)


if __name__ == "__main__":
    main()
