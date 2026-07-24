#!/usr/bin/env python3
"""Build a multi-series traffic panel by slice and ip_id."""

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
        "--input-csv",
        default="simulation/mininet/cesnet_points_clustered_4slices.csv",
        help="Sliced dataset produced by Dataset Preparing/cluster_4_slices.py.",
    )
    parser.add_argument(
        "--output-csv",
        default="traffic_forecasting/data/ip_slice_traffic_10min_long.csv",
        help="Long panel CSV to create.",
    )
    parser.add_argument(
        "--slice-output-csv",
        default="traffic_forecasting/data/slice_traffic_10min_from_ip.csv",
        help="Wide slice aggregate rebuilt from the same grouped data.",
    )
    parser.add_argument("--target", default="n_bytes")
    parser.add_argument("--freq", default="10min", help="Pandas resampling frequency.")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug limit.")
    parser.add_argument(
        "--top-k-per-slice",
        type=int,
        default=0,
        help="Keep only the top K ip_id values per slice by total target. Use 0 to keep all observed series.",
    )
    parser.add_argument(
        "--active-tail-min-points",
        type=int,
        default=0,
        help="Keep series with at least N observed timestamps in the retained tail window. Requires --tail-timestamps.",
    )
    parser.add_argument(
        "--min-total-points",
        type=int,
        default=0,
        help="Keep series with at least N observed timestamps across the full aggregated period.",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Reindex selected series on the full timestamp grid and fill missing points with zero.",
    )
    parser.add_argument(
        "--add-other-series",
        action="store_true",
        help="Deprecated for model training. Add one synthetic residual OTHER series per slice so selected aggregates match total slice traffic.",
    )
    parser.add_argument(
        "--tail-timestamps",
        type=int,
        default=0,
        help="Keep only the last N timestamps after aggregation. Use 0 to keep the full period.",
    )
    return parser.parse_args()


def _normalize_chunk(chunk: pd.DataFrame, target: str) -> pd.DataFrame:
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
    chunk[target] = pd.to_numeric(chunk[target], errors="coerce")
    chunk["ip_id"] = chunk["ip_id"].astype(str)
    chunk = chunk.dropna(subset=["timestamp", "slice", "ip_id", target])
    return chunk


def _unique_id(slice_values: pd.Series, ip_values: pd.Series) -> pd.Series:
    return slice_values.astype(str) + "__ip_" + ip_values.astype(str)


def build_ip_slice_series(
    input_csv: str | Path,
    output_csv: str | Path,
    slice_output_csv: str | Path,
    target: str,
    freq: str,
    chunksize: int,
    max_rows: int | None,
    top_k_per_slice: int,
    active_tail_min_points: int,
    min_total_points: int,
    dense: bool,
    add_other_series: bool,
    tail_timestamps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    required = ["timestamp", "slice", "ip_id", target]
    grouped_chunks = []
    rows_seen = 0

    for chunk in pd.read_csv(input_path, usecols=required, chunksize=chunksize):
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]

        chunk = _normalize_chunk(chunk, target)
        grouped = chunk.groupby(["timestamp", "slice", "ip_id"], as_index=False)[target].sum()
        grouped_chunks.append(grouped)
        rows_seen += len(chunk)

    if not grouped_chunks:
        raise ValueError("No valid rows found in input CSV")

    panel = pd.concat(grouped_chunks, ignore_index=True)
    panel = panel.groupby(["timestamp", "slice", "ip_id"], as_index=False)[target].sum()
    panel = panel[panel["slice"].isin(SLICES)].copy()
    panel["timestamp"] = panel["timestamp"].dt.floor(freq)
    panel = panel.groupby(["timestamp", "slice", "ip_id"], as_index=False)[target].sum()

    selected_ids = None
    if top_k_per_slice > 0:
        totals = panel.groupby(["slice", "ip_id"], as_index=False)[target].sum()
        selected_ids = (
            totals.sort_values(["slice", target], ascending=[True, False])
            .groupby("slice", group_keys=False)
            .head(top_k_per_slice)[["slice", "ip_id"]]
        )

    if min_total_points > 0:
        total_ids = (
            panel.drop_duplicates(["timestamp", "slice", "ip_id"])
            .groupby(["slice", "ip_id"])
            .size()
            .reset_index(name="observed_points")
        )
        total_ids = total_ids[total_ids["observed_points"] >= min_total_points][["slice", "ip_id"]]
        selected_ids = total_ids if selected_ids is None else selected_ids.merge(total_ids, on=["slice", "ip_id"], how="inner")

    if active_tail_min_points > 0:
        if tail_timestamps <= 0:
            raise ValueError("--active-tail-min-points requires --tail-timestamps")
        full_index = pd.date_range(panel["timestamp"].min(), panel["timestamp"].max(), freq=freq)
        tail_index = full_index[-tail_timestamps:]
        active_ids = (
            panel[panel["timestamp"].isin(tail_index)]
            .drop_duplicates(["timestamp", "slice", "ip_id"])
            .groupby(["slice", "ip_id"])
            .size()
            .reset_index(name="observed_points")
        )
        active_ids = active_ids[active_ids["observed_points"] >= active_tail_min_points][["slice", "ip_id"]]
        selected_ids = active_ids if selected_ids is None else selected_ids.merge(active_ids, on=["slice", "ip_id"], how="inner")

    if selected_ids is not None:
        selected = panel.merge(selected_ids, on=["slice", "ip_id"], how="inner")
        if add_other_series:
            full_slice = panel.groupby(["timestamp", "slice"], as_index=False)[target].sum()
            selected_slice = selected.groupby(["timestamp", "slice"], as_index=False)[target].sum()
            residual = full_slice.merge(
                selected_slice,
                on=["timestamp", "slice"],
                how="left",
                suffixes=("_full", "_selected"),
            )
            residual[f"{target}_selected"] = residual[f"{target}_selected"].fillna(0.0)
            residual[target] = residual[f"{target}_full"] - residual[f"{target}_selected"]
            residual[target] = residual[target].clip(lower=0.0)
            residual["ip_id"] = "OTHER"
            residual = residual[["timestamp", "slice", "ip_id", target]]
            panel = pd.concat([selected, residual], ignore_index=True)
        else:
            panel = selected
    elif add_other_series:
        raise ValueError("--add-other-series requires a selection filter such as --top-k-per-slice or --active-tail-min-points")

    panel["unique_id"] = _unique_id(panel["slice"], panel["ip_id"])
    panel = panel.rename(columns={"timestamp": "ds", target: "y"})
    panel = panel[["unique_id", "ds", "y", "slice", "ip_id"]].sort_values(["unique_id", "ds"])

    if tail_timestamps > 0 and not dense:
        timestamps = pd.DatetimeIndex(sorted(panel["ds"].unique()))
        if len(timestamps) > tail_timestamps:
            keep = timestamps[-tail_timestamps:]
            panel = panel[panel["ds"].isin(keep)].copy()

    if dense:
        if top_k_per_slice <= 0 and active_tail_min_points <= 0 and min_total_points <= 0 and tail_timestamps <= 0:
            raise ValueError("--dense without selection should use --tail-timestamps to avoid creating a huge dense grid")
        panel = densify_panel(panel, freq, tail_timestamps)

    slice_wide = (
        panel.groupby(["ds", "slice"], as_index=False)["y"].sum()
        .pivot(index="ds", columns="slice", values="y")
        .sort_index()
        .fillna(0.0)
    )
    for slice_name in SLICES:
        if slice_name not in slice_wide.columns:
            slice_wide[slice_name] = 0.0
    slice_wide = slice_wide[SLICES].reset_index().rename(columns={"ds": "timestamp"})

    output_path = ensure_parent(output_csv)
    slice_output_path = ensure_parent(slice_output_csv)
    panel.to_csv(output_path, index=False)
    slice_wide.to_csv(slice_output_path, index=False)
    return panel, slice_wide


def densify_panel(panel: pd.DataFrame, freq: str, tail_timestamps: int = 0) -> pd.DataFrame:
    ids = panel[["unique_id", "slice", "ip_id"]].drop_duplicates().sort_values("unique_id")
    full_index = pd.date_range(panel["ds"].min(), panel["ds"].max(), freq=freq)
    if tail_timestamps > 0 and len(full_index) > tail_timestamps:
        full_index = full_index[-tail_timestamps:]
    grid = pd.MultiIndex.from_product(
        [ids["unique_id"].to_numpy(), full_index],
        names=["unique_id", "ds"],
    ).to_frame(index=False)
    values = panel[panel["ds"].isin(full_index)][["unique_id", "ds", "y"]]
    dense = grid.merge(values, on=["unique_id", "ds"], how="left")
    dense["y"] = dense["y"].fillna(0.0)
    dense = dense.merge(ids, on="unique_id", how="left")
    return dense[["unique_id", "ds", "y", "slice", "ip_id"]]


def main() -> None:
    args = parse_args()
    panel, slice_wide = build_ip_slice_series(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        slice_output_csv=args.slice_output_csv,
        target=args.target,
        freq=args.freq,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        top_k_per_slice=args.top_k_per_slice,
        active_tail_min_points=args.active_tail_min_points,
        min_total_points=args.min_total_points,
        dense=args.dense,
        add_other_series=args.add_other_series,
        tail_timestamps=args.tail_timestamps,
    )
    print(f"Wrote {args.output_csv}")
    print(f"panel_rows={len(panel)} series={panel['unique_id'].nunique()}")
    print(f"period={panel['ds'].min()} -> {panel['ds'].max()}")
    print(f"Wrote {args.slice_output_csv}")
    print(f"slice_rows={len(slice_wide)} columns={list(slice_wide.columns)}")


if __name__ == "__main__":
    main()
