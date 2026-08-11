"""Consolidate benchmark summaries across model run directories."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nsf.utils.io import ensure_parent


DEFAULT_RUNS = {
    "lightgbm_tuned": "forecasting/experiments/runs/deterministic_benchmark_lightgbm_tuned",
    "lstm_5000w": "forecasting/experiments/runs/lstm_benchmark_5000w",
    "prophet_tuned": "forecasting/experiments/runs/prophet_benchmark_tuned",
    "nhits_tuned": "forecasting/experiments/runs/nhits_benchmark_tuned",
    "nhits_robust_tuned": "forecasting/experiments/runs/nhits_benchmark_robust_tuned",
    "patchtst_tuned": "forecasting/experiments/runs/patchtst_benchmark_tuned",
    "patchtst_fast": "forecasting/experiments/runs/patchtst_benchmark_fast",
    "patchtst_balanced": "forecasting/experiments/runs/patchtst_benchmark_balanced",
    "patchtst_heavy1h": "forecasting/experiments/runs/patchtst_benchmark_heavy1h",
}


def _read_run(label: str, run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_path = Path(run_dir)
    summary = pd.read_csv(run_path / "benchmark_summary.csv")
    by_slice = pd.read_csv(run_path / "benchmark_summary_by_slice.csv")
    summary.insert(0, "run", label)
    by_slice.insert(0, "run", label)
    return summary, by_slice


def build_model_comparison(output_dir: str | Path, runs: dict[str, str] | None = None) -> dict[str, Path]:
    output_path = Path(output_dir)
    run_map = runs or DEFAULT_RUNS
    summaries = []
    summaries_by_slice = []
    missing = []
    for label, run_dir in run_map.items():
        run_path = Path(run_dir)
        if not (run_path / "benchmark_summary.csv").exists():
            missing.append(str(run_path))
            continue
        summary, by_slice = _read_run(label, run_path)
        summaries.append(summary)
        summaries_by_slice.append(by_slice)
    if not summaries:
        raise FileNotFoundError(f"No benchmark summaries found. Missing: {missing}")

    global_df = pd.concat(summaries, ignore_index=True).sort_values(["RMSE", "WAPE", "MASE"])
    global_df["rank_rmse_overall"] = global_df["RMSE"].rank(method="dense")
    global_df["rank_wape_overall"] = global_df["WAPE"].rank(method="dense")
    global_df["rank_mase_overall"] = global_df["MASE"].rank(method="dense")

    by_slice_df = pd.concat(summaries_by_slice, ignore_index=True).sort_values(["slice", "RMSE", "WAPE", "MASE"])
    by_slice_df["rank_rmse_across_runs"] = by_slice_df.groupby("slice")["RMSE"].rank(method="dense")
    by_slice_df["rank_wape_across_runs"] = by_slice_df.groupby("slice")["WAPE"].rank(method="dense")
    by_slice_df["rank_mase_across_runs"] = by_slice_df.groupby("slice")["MASE"].rank(method="dense")

    bias_cols = ["run", "model", "slice", "bias", "under_prediction_error", "over_prediction_error"]
    bias_df = by_slice_df[[col for col in bias_cols if col in by_slice_df.columns]].copy()

    paths = {
        "global": output_path / "model_comparison_global.csv",
        "by_slice": output_path / "model_comparison_by_slice.csv",
        "bias": output_path / "model_comparison_bias.csv",
    }
    global_df.to_csv(ensure_parent(paths["global"]), index=False)
    by_slice_df.to_csv(ensure_parent(paths["by_slice"]), index=False)
    bias_df.to_csv(ensure_parent(paths["bias"]), index=False)
    return paths
