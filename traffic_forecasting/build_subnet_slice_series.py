#!/usr/bin/env python3
"""Build forecasting series by institutional subnet and slice."""

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
        "--relationship-csv",
        default="ForecastingDoc/ids_relationship.csv",
        help="CESNET mapping with id_ip, id_institution, id_institution_subnet.",
    )
    parser.add_argument(
        "--output-csv",
        default="traffic_forecasting/data/subnet_slice_traffic_10min_long.csv",
        help="Long subnet/slice panel CSV to create.",
    )
    parser.add_argument(
        "--slice-output-csv",
        default="traffic_forecasting/data/slice_traffic_from_subnet_10min.csv",
        help="Wide slice aggregate rebuilt from subnet/slice series.",
    )
    parser.add_argument("--target", default="n_bytes")
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug limit.")
    parser.add_argument(
        "--min-total-points",
        type=int,
        default=0,
        help="Keep subnet/slice series with at least N observed timestamps. Use 0 to keep all.",
    )
    parser.add_argument(
        "--tail-timestamps",
        type=int,
        default=0,
        help="Keep only the last N timestamps after aggregation. Use 0 to keep full period.",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Reindex selected subnet/slice series on the regular timestamp grid and fill missing points with zero.",
    )
    return parser.parse_args()


def read_relationship(path: str | Path) -> pd.DataFrame:
    rel = pd.read_csv(path)
    required = {"id_ip", "id_institution", "id_institution_subnet"}
    missing = sorted(required - set(rel.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    rel = rel[list(required)].copy()
    rel = rel.rename(columns={"id_ip": "ip_id"})
    rel["ip_id"] = pd.to_numeric(rel["ip_id"], errors="coerce").astype("Int64")
    rel["id_institution"] = pd.to_numeric(rel["id_institution"], errors="coerce").astype("Int64")
    rel["id_institution_subnet"] = pd.to_numeric(rel["id_institution_subnet"], errors="coerce").astype("Int64")
    rel = rel.dropna(subset=["ip_id", "id_institution", "id_institution_subnet"])
    return rel.drop_duplicates("ip_id")


def unique_id(df: pd.DataFrame) -> pd.Series:
    return (
        "subnet_"
        + df["id_institution_subnet"].astype(str)
        + "__"
        + df["slice"].astype(str)
    )


def densify_panel(panel: pd.DataFrame, freq: str, tail_timestamps: int = 0) -> pd.DataFrame:
    ids = panel[
        ["unique_id", "slice", "id_institution", "id_institution_subnet"]
    ].drop_duplicates().sort_values("unique_id")
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
    return dense[["unique_id", "ds", "y", "slice", "id_institution", "id_institution_subnet"]]


def build_subnet_slice_series(
    input_csv: str | Path,
    relationship_csv: str | Path,
    output_csv: str | Path,
    slice_output_csv: str | Path,
    target: str,
    freq: str,
    chunksize: int,
    max_rows: int | None,
    min_total_points: int,
    tail_timestamps: int,
    dense: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    rel = read_relationship(relationship_csv)
    required = ["timestamp", "slice", "ip_id", target]
    chunks = []
    rows_seen = 0
    missing_mapping_rows = 0

    for chunk in pd.read_csv(input_path, usecols=required, chunksize=chunksize):
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]

        rows_seen += len(chunk)
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
        chunk["ip_id"] = pd.to_numeric(chunk["ip_id"], errors="coerce").astype("Int64")
        chunk[target] = pd.to_numeric(chunk[target], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp", "slice", "ip_id", target])
        chunk = chunk[chunk["slice"].isin(SLICES)].copy()
        chunk = chunk.merge(rel, on="ip_id", how="left")
        missing_mapping_rows += int(chunk["id_institution_subnet"].isna().sum())
        chunk = chunk.dropna(subset=["id_institution", "id_institution_subnet"])
        if chunk.empty:
            continue

        chunk["timestamp"] = chunk["timestamp"].dt.floor(freq)
        grouped = chunk.groupby(
            ["timestamp", "id_institution", "id_institution_subnet", "slice"],
            as_index=False,
        )[target].sum()
        chunks.append(grouped)

    if not chunks:
        raise ValueError("No valid mapped rows found")

    panel = pd.concat(chunks, ignore_index=True)
    panel = panel.groupby(
        ["timestamp", "id_institution", "id_institution_subnet", "slice"],
        as_index=False,
    )[target].sum()

    if min_total_points > 0:
        keep = (
            panel.drop_duplicates(["timestamp", "id_institution_subnet", "slice"])
            .groupby(["id_institution_subnet", "slice"])
            .size()
            .reset_index(name="observed_points")
        )
        keep = keep[keep["observed_points"] >= min_total_points][["id_institution_subnet", "slice"]]
        panel = panel.merge(keep, on=["id_institution_subnet", "slice"], how="inner")

    panel["unique_id"] = unique_id(panel)
    panel = panel.rename(columns={"timestamp": "ds", target: "y"})
    panel = panel[
        ["unique_id", "ds", "y", "slice", "id_institution", "id_institution_subnet"]
    ].sort_values(["unique_id", "ds"])

    if tail_timestamps > 0 and not dense:
        timestamps = pd.DatetimeIndex(sorted(panel["ds"].unique()))
        if len(timestamps) > tail_timestamps:
            keep = timestamps[-tail_timestamps:]
            panel = panel[panel["ds"].isin(keep)].copy()

    if dense:
        panel = densify_panel(panel, freq=freq, tail_timestamps=tail_timestamps)

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

    summary = pd.DataFrame(
        [
            {
                "input_rows_seen": rows_seen,
                "missing_mapping_rows": missing_mapping_rows,
                "panel_rows": len(panel),
                "series": panel["unique_id"].nunique(),
                "institutions": panel["id_institution"].nunique(),
                "subnets": panel["id_institution_subnet"].nunique(),
                "timestamps": panel["ds"].nunique(),
            }
        ]
    )

    panel.to_csv(ensure_parent(output_csv), index=False)
    slice_wide.to_csv(ensure_parent(slice_output_csv), index=False)
    return panel, slice_wide, summary


def main() -> None:
    args = parse_args()
    panel, slice_wide, summary = build_subnet_slice_series(
        input_csv=args.input_csv,
        relationship_csv=args.relationship_csv,
        output_csv=args.output_csv,
        slice_output_csv=args.slice_output_csv,
        target=args.target,
        freq=args.freq,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        min_total_points=args.min_total_points,
        tail_timestamps=args.tail_timestamps,
        dense=args.dense,
    )
    print(f"Wrote {args.output_csv}")
    print(f"panel_rows={len(panel)} series={panel['unique_id'].nunique()}")
    print(f"period={panel['ds'].min()} -> {panel['ds'].max()}")
    print(f"Wrote {args.slice_output_csv}")
    print(f"slice_rows={len(slice_wide)} columns={list(slice_wide.columns)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
