"""Exploratory analysis for subnet/slice forecasting panels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nsf.constants import SLICES
from nsf.utils.io import ensure_parent


@dataclass(frozen=True)
class PanelEDAResult:
    overview: dict[str, object]
    slice_summary: pd.DataFrame
    series_summary: pd.DataFrame
    concentration: pd.DataFrame
    hourly_profile: pd.DataFrame
    weekday_profile: pd.DataFrame
    autocorrelation: pd.DataFrame


def read_subnet_slice_panel(path: str | Path) -> pd.DataFrame:
    """Read the subnet/slice panel with memory-friendly dtypes."""

    usecols = ["unique_id", "ds", "y", "slice", "id_institution", "id_institution_subnet"]
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype={
            "unique_id": "string",
            "slice": "category",
            "id_institution": "int32",
            "id_institution_subnet": "int32",
            "y": "float64",
        },
    )
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["unique_id", "ds", "y", "slice"])
    df = df[df["slice"].astype(str).isin(SLICES)].copy()
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def overview(df: pd.DataFrame, freq: str) -> dict[str, object]:
    timestamps = pd.DatetimeIndex(sorted(df["ds"].unique()))
    expected = pd.date_range(timestamps.min(), timestamps.max(), freq=freq) if len(timestamps) else pd.DatetimeIndex([])
    duplicates = int(df.duplicated(["unique_id", "ds"]).sum())
    return {
        "rows": int(len(df)),
        "series": int(df["unique_id"].nunique()),
        "slices": int(df["slice"].nunique()),
        "subnets": int(df["id_institution_subnet"].nunique()),
        "institutions": int(df["id_institution"].nunique()),
        "timestamps": int(len(timestamps)),
        "expected_timestamps": int(len(expected)),
        "missing_global_timestamps": int(len(expected.difference(timestamps))),
        "duplicate_unique_id_timestamps": duplicates,
        "start": timestamps.min() if len(timestamps) else None,
        "end": timestamps.max() if len(timestamps) else None,
        "freq": freq,
    }


def summarize_by_series(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    expected_points = int(pd.date_range(df["ds"].min(), df["ds"].max(), freq=freq).size)
    grouped = df.groupby(["unique_id", "slice", "id_institution", "id_institution_subnet"], observed=True)
    summary = grouped["y"].agg(
        points="size",
        total_bytes="sum",
        mean_bytes="mean",
        median_bytes="median",
        max_bytes="max",
        p95_bytes=lambda s: float(s.quantile(0.95)),
        p99_bytes=lambda s: float(s.quantile(0.99)),
    ).reset_index()
    zeros = grouped["y"].apply(lambda s: float((s == 0).mean())).rename("zero_share").reset_index()
    nonzero = grouped["y"].apply(lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0).rename("nonzero_mean_bytes").reset_index()
    summary = summary.merge(zeros, on=["unique_id", "slice", "id_institution", "id_institution_subnet"])
    summary = summary.merge(nonzero, on=["unique_id", "slice", "id_institution", "id_institution_subnet"])
    summary["missing_points"] = expected_points - summary["points"]
    summary["coverage_share"] = summary["points"] / expected_points if expected_points else 0.0
    summary["cv_bytes"] = grouped["y"].std().to_numpy() / summary["mean_bytes"].replace(0, np.nan).to_numpy()
    summary["cv_bytes"] = summary["cv_bytes"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return summary.sort_values("total_bytes", ascending=False).reset_index(drop=True)


def summarize_by_slice(series_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slice_name, sub in series_summary.groupby("slice", observed=True):
        total = float(sub["total_bytes"].sum())
        rows.append(
            {
                "slice": str(slice_name),
                "series": int(sub["unique_id"].nunique()),
                "subnets": int(sub["id_institution_subnet"].nunique()),
                "total_bytes": total,
                "traffic_share": np.nan,
                "median_series_total_bytes": float(sub["total_bytes"].median()),
                "mean_zero_share": float(sub["zero_share"].mean()),
                "median_zero_share": float(sub["zero_share"].median()),
                "max_series_share_within_slice": float(sub["total_bytes"].max() / total) if total > 0 else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    grand_total = float(out["total_bytes"].sum())
    out["traffic_share"] = out["total_bytes"] / grand_total if grand_total > 0 else 0.0
    return out.sort_values("slice").reset_index(drop=True)


def concentration_by_slice(series_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slice_name, sub in series_summary.groupby("slice", observed=True):
        totals = sub.sort_values("total_bytes", ascending=False)["total_bytes"].to_numpy(dtype=float)
        total = float(totals.sum())
        if total <= 0:
            continue
        for k in [1, 3, 5, 10]:
            rows.append({"slice": str(slice_name), "top_k_series": k, "traffic_share": float(totals[:k].sum() / total)})
    return pd.DataFrame(rows)


def calendar_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[["ds", "slice", "y"]].copy()
    work["hour"] = work["ds"].dt.hour
    work["dayofweek"] = work["ds"].dt.dayofweek
    hourly = (
        work.groupby(["slice", "hour"], observed=True)["y"]
        .agg(mean_bytes="mean", median_bytes="median", p95_bytes=lambda s: float(s.quantile(0.95)))
        .reset_index()
    )
    weekday = (
        work.groupby(["slice", "dayofweek"], observed=True)["y"]
        .agg(mean_bytes="mean", median_bytes="median", p95_bytes=lambda s: float(s.quantile(0.95)))
        .reset_index()
    )
    return hourly, weekday


def autocorrelation_summary(df: pd.DataFrame, lags: tuple[int, ...], top_n_per_slice: int) -> pd.DataFrame:
    rows = []
    totals = df.groupby(["slice", "unique_id"], observed=True)["y"].sum().reset_index()
    selected_ids = (
        totals.sort_values(["slice", "y"], ascending=[True, False])
        .groupby("slice", observed=True)
        .head(top_n_per_slice)["unique_id"]
        .astype(str)
        .tolist()
    )
    selected = df[df["unique_id"].astype(str).isin(selected_ids)]
    for (slice_name, unique_id), sub in selected.groupby(["slice", "unique_id"], observed=True):
        values = sub.sort_values("ds")["y"].to_numpy(dtype=float)
        for lag in lags:
            if len(values) <= lag or np.std(values[:-lag]) == 0 or np.std(values[lag:]) == 0:
                corr = np.nan
            else:
                corr = float(np.corrcoef(values[:-lag], values[lag:])[0, 1])
            rows.append({"slice": str(slice_name), "unique_id": str(unique_id), "lag": lag, "autocorrelation": corr})
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    return (
        raw.groupby(["slice", "lag"], observed=True)["autocorrelation"]
        .agg(series="count", mean_autocorrelation="mean", median_autocorrelation="median")
        .reset_index()
    )


def run_panel_eda(
    input_csv: str | Path,
    output_dir: str | Path,
    report_md: str | Path,
    freq: str = "10min",
    autocorr_lags: tuple[int, ...] = (1, 6, 12, 36, 144, 1008),
    autocorr_top_n_per_slice: int = 10,
) -> PanelEDAResult:
    df = read_subnet_slice_panel(input_csv)
    result_overview = overview(df, freq=freq)
    series_summary = summarize_by_series(df, freq=freq)
    slice_summary = summarize_by_slice(series_summary)
    concentration = concentration_by_slice(series_summary)
    hourly_profile, weekday_profile = calendar_profiles(df)
    autocorrelation = autocorrelation_summary(df, lags=autocorr_lags, top_n_per_slice=autocorr_top_n_per_slice)

    output_path = Path(output_dir)
    ensure_parent(output_path / "placeholder")
    series_summary.to_csv(output_path / "subnet_slice_eda_series_summary.csv", index=False)
    slice_summary.to_csv(output_path / "subnet_slice_eda_slice_summary.csv", index=False)
    concentration.to_csv(output_path / "subnet_slice_eda_concentration.csv", index=False)
    hourly_profile.to_csv(output_path / "subnet_slice_eda_hourly_profile.csv", index=False)
    weekday_profile.to_csv(output_path / "subnet_slice_eda_weekday_profile.csv", index=False)
    autocorrelation.to_csv(output_path / "subnet_slice_eda_autocorrelation.csv", index=False)

    write_markdown_report(
        report_md=report_md,
        input_csv=input_csv,
        result=PanelEDAResult(
            overview=result_overview,
            slice_summary=slice_summary,
            series_summary=series_summary,
            concentration=concentration,
            hourly_profile=hourly_profile,
            weekday_profile=weekday_profile,
            autocorrelation=autocorrelation,
        ),
    )
    return PanelEDAResult(
        overview=result_overview,
        slice_summary=slice_summary,
        series_summary=series_summary,
        concentration=concentration,
        hourly_profile=hourly_profile,
        weekday_profile=weekday_profile,
        autocorrelation=autocorrelation,
    )


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    table = table.astype(object).where(pd.notna(table), "")
    headers = [str(col) for col in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown_report(report_md: str | Path, input_csv: str | Path, result: PanelEDAResult) -> None:
    report_path = ensure_parent(report_md)
    overview_items = "\n".join(f"- {key}: `{value}`" for key, value in result.overview.items())
    top_series = result.series_summary.head(10).copy()
    top_series["zero_share"] = top_series["zero_share"].map(_fmt_pct)
    top_series["traffic_share_global"] = top_series["total_bytes"] / result.series_summary["total_bytes"].sum()
    top_series["traffic_share_global"] = top_series["traffic_share_global"].map(_fmt_pct)

    slice_table = result.slice_summary.copy()
    for col in ["traffic_share", "mean_zero_share", "median_zero_share", "max_series_share_within_slice"]:
        slice_table[col] = slice_table[col].map(_fmt_pct)

    concentration_table = result.concentration.copy()
    if not concentration_table.empty:
        concentration_table["traffic_share"] = concentration_table["traffic_share"].map(_fmt_pct)

    lines = [
        "# Subnet/Slice EDA Report",
        "",
        "This report uses the subnet/slice panel as the forecasting dataset. The 4-slice aggregate is not used as the analysis unit.",
        "",
        "## Input",
        "",
        f"- panel: `{input_csv}`",
        "- granularity: `unique_id = subnet + slice`",
        "- target: `y` traffic bytes per 10-minute step",
        "",
        "## Dataset Overview",
        "",
        overview_items,
        "",
        "## Slice Summary",
        "",
        _markdown_table(slice_table),
        "",
        "## Traffic Concentration",
        "",
        _markdown_table(concentration_table),
        "",
        "## Top 10 Subnet/Slice Series By Traffic",
        "",
        _markdown_table(
            top_series[
                [
                    "unique_id",
                    "slice",
                    "id_institution_subnet",
                    "total_bytes",
                    "traffic_share_global",
                    "zero_share",
                    "mean_bytes",
                    "p95_bytes",
                    "p99_bytes",
                ]
            ]
        ),
        "",
        "## Autocorrelation Summary",
        "",
        _markdown_table(result.autocorrelation) if not result.autocorrelation.empty else "No autocorrelation values computed.",
        "",
        "## Methodological Implications",
        "",
        "- The forecasting benchmark should remain panel-based at subnet/slice granularity.",
        "- Zero share and traffic concentration must be reported because they affect model choice and error interpretation.",
        "- Daily lag `144` and weekly lag `1008` should remain explicit seasonal baselines/features.",
        "- Metrics must be reported by horizon because short-term and six-hour behavior can differ.",
        "- Scaling and feature statistics must be fit per training fold only, never on the full panel.",
        "",
        "## Reproducible Outputs",
        "",
        "- `subnet_slice_eda_series_summary.csv`",
        "- `subnet_slice_eda_slice_summary.csv`",
        "- `subnet_slice_eda_concentration.csv`",
        "- `subnet_slice_eda_hourly_profile.csv`",
        "- `subnet_slice_eda_weekday_profile.csv`",
        "- `subnet_slice_eda_autocorrelation.csv`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
