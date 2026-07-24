"""Consolidate benchmark summaries across input-history sensitivity runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nsf.utils.io import ensure_parent


DEFAULT_RUNS = {
    ("14d", "prophet_tuned"): "experiments/runs/prophet_benchmark_tuned",
    ("7d", "prophet_tuned"): "experiments/runs/prophet_benchmark_tuned_hist1w",
    ("1d", "prophet_tuned"): "experiments/runs/prophet_benchmark_tuned_hist1d",
    ("14d", "lightgbm_tuned"): "experiments/runs/deterministic_benchmark_lightgbm_tuned",
    ("7d", "lightgbm_tuned"): "experiments/runs/deterministic_benchmark_lightgbm_tuned_hist1w",
    ("1d", "lightgbm_tuned"): "experiments/runs/deterministic_benchmark_lightgbm_tuned_hist1d",
    ("14d", "lstm_5000w"): "experiments/runs/lstm_benchmark_5000w",
    ("7d", "lstm_5000w"): "experiments/runs/lstm_benchmark_5000w_hist1w",
    ("1d", "lstm_5000w"): "experiments/runs/lstm_benchmark_5000w_hist1d",
    ("14d", "patchtst_tuned"): "experiments/runs/patchtst_benchmark_tuned",
    ("7d", "patchtst_tuned"): "experiments/runs/patchtst_benchmark_tuned_hist1w",
    ("1d", "patchtst_tuned"): "experiments/runs/patchtst_benchmark_tuned_hist1d",
    ("14d", "nhits_tuned"): "experiments/runs/nhits_benchmark_tuned",
    ("7d", "nhits_tuned"): "experiments/runs/nhits_benchmark_tuned_hist1w",
    ("1d", "nhits_tuned"): "experiments/runs/nhits_benchmark_tuned_hist1d",
}


def _read_run(history: str, run: str, run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_path = Path(run_dir)
    summary = pd.read_csv(run_path / "benchmark_summary.csv")
    by_slice = pd.read_csv(run_path / "benchmark_summary_by_slice.csv")
    summary.insert(0, "history", history)
    summary.insert(1, "run", run)
    by_slice.insert(0, "history", history)
    by_slice.insert(1, "run", run)
    return summary, by_slice


def build_history_sensitivity(output_dir: str | Path, runs: dict[tuple[str, str], str] | None = None) -> dict[str, Path]:
    output_path = Path(output_dir)
    run_map = runs or DEFAULT_RUNS
    summaries = []
    summaries_by_slice = []
    for (history, run), run_dir in run_map.items():
        run_path = Path(run_dir)
        if not (run_path / "benchmark_summary.csv").exists():
            continue
        summary, by_slice = _read_run(history, run, run_path)
        summaries.append(summary)
        summaries_by_slice.append(by_slice)
    if not summaries:
        raise FileNotFoundError("No history sensitivity summaries found")

    global_df = pd.concat(summaries, ignore_index=True).sort_values(["run", "history"])
    by_slice_df = pd.concat(summaries_by_slice, ignore_index=True).sort_values(["run", "slice", "history"])
    paths = {
        "global": output_path / "history_sensitivity_global.csv",
        "by_slice": output_path / "history_sensitivity_by_slice.csv",
    }
    global_df.to_csv(ensure_parent(paths["global"]), index=False)
    by_slice_df.to_csv(ensure_parent(paths["by_slice"]), index=False)
    return paths
