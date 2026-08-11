"""Build the subnet/slice forecasting panel from the clustered dataset.

Turns the clustered per-IP dataset produced by
``Dataset Preparing/cluster_4_slices.py`` into the long subnet/slice panel used
by the forecasting benchmarks. Each IP is mapped to its institution/subnet via
``ids_relationship.csv``, then traffic is summed per
(timestamp, subnet, slice). With ``dense=True`` the selected series are
reindexed onto the regular timestamp grid with missing points filled to zero --
this is how ``subnet_slice_traffic_min2016_dense.csv`` is produced.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nsf.constants import SLICES
from nsf.utils.io import ensure_parent


def read_relationship(path: str | Path) -> pd.DataFrame:
    """Load the id_ip -> institution/subnet mapping, one row per ip."""
    rel = pd.read_csv(path)
    required = {"id_ip", "id_institution", "id_institution_subnet"}
    missing = sorted(required - set(rel.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    rel = rel[list(required)].rename(columns={"id_ip": "ip_id"})
    for col in ("ip_id", "id_institution", "id_institution_subnet"):
        rel[col] = pd.to_numeric(rel[col], errors="coerce").astype("Int64")
    rel = rel.dropna(subset=["ip_id", "id_institution", "id_institution_subnet"])
    return rel.drop_duplicates("ip_id")


def _unique_id(df: pd.DataFrame) -> pd.Series:
    return "subnet_" + df["id_institution_subnet"].astype(str) + "__" + df["slice"].astype(str)


def densify_panel(panel: pd.DataFrame, freq: str, tail_timestamps: int = 0) -> pd.DataFrame:
    """Reindex every series onto the full timestamp grid, filling gaps with zero."""
    ids = (
        panel[["unique_id", "slice", "id_institution", "id_institution_subnet"]]
        .drop_duplicates()
        .sort_values("unique_id")
    )
    full_index = pd.date_range(panel["ds"].min(), panel["ds"].max(), freq=freq)
    if tail_timestamps > 0 and len(full_index) > tail_timestamps:
        full_index = full_index[-tail_timestamps:]

    grid = pd.MultiIndex.from_product(
        [ids["unique_id"].to_numpy(), full_index], names=["unique_id", "ds"]
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
    target: str = "n_bytes",
    freq: str = "10min",
    chunksize: int = 500_000,
    max_rows: int | None = None,
    min_total_points: int = 0,
    tail_timestamps: int = 0,
    dense: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the long subnet/slice panel and a wide per-slice aggregate.

    Returns ``(panel, slice_wide, summary)`` and writes both CSVs.
    """
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
            ["timestamp", "id_institution", "id_institution_subnet", "slice"], as_index=False
        )[target].sum()
        chunks.append(grouped)

    if not chunks:
        raise ValueError("No valid mapped rows found")

    panel = pd.concat(chunks, ignore_index=True)
    panel = panel.groupby(
        ["timestamp", "id_institution", "id_institution_subnet", "slice"], as_index=False
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

    panel["unique_id"] = _unique_id(panel)
    panel = panel.rename(columns={"timestamp": "ds", target: "y"})
    panel = panel[["unique_id", "ds", "y", "slice", "id_institution", "id_institution_subnet"]].sort_values(
        ["unique_id", "ds"]
    )

    if tail_timestamps > 0 and not dense:
        timestamps = pd.DatetimeIndex(sorted(panel["ds"].unique()))
        if len(timestamps) > tail_timestamps:
            panel = panel[panel["ds"].isin(timestamps[-tail_timestamps:])].copy()

    if dense:
        panel = densify_panel(panel, freq=freq, tail_timestamps=tail_timestamps)

    slice_wide = (
        panel.groupby(["ds", "slice"], as_index=False)["y"]
        .sum()
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
