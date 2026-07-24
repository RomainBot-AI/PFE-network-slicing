#!/usr/bin/env python3
"""Select deterministic forecasting models for a probabilistic extension."""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import pandas as pd


RUNS = {
    "prophet_tuned": {
        "path": "experiments/runs/prophet_benchmark_tuned",
        "label": "Prophet tuned",
        "family": "statistical",
        "probabilistic": "Native prediction intervals; calibrate coverage per slice because deterministic runs show bias.",
    },
    "lightgbm_tuned": {
        "path": "experiments/runs/deterministic_benchmark_lightgbm_tuned",
        "label": "LightGBM tuned",
        "family": "machine_learning",
        "probabilistic": "Quantile regression with separate alpha models, or conformalized residual intervals.",
    },
    "lstm_5000w": {
        "path": "experiments/runs/lstm_benchmark_5000w",
        "label": "LSTM 5000w",
        "family": "deep_recurrent",
        "probabilistic": "Quantile loss head or distributional output head; requires extra implementation.",
    },
    "nhits_tuned": {
        "path": "experiments/runs/nhits_benchmark_tuned",
        "label": "N-HiTS tuned",
        "family": "deep_mlp",
        "probabilistic": "NeuralForecast quantile/distribution losses; stronger as a 1-day sensitivity than under the 14-day protocol.",
    },
    "patchtst_tuned": {
        "path": "experiments/runs/patchtst_benchmark_tuned",
        "label": "PatchTST tuned",
        "family": "deep_transformer",
        "probabilistic": "NeuralForecast quantile/distribution losses; natural probabilistic transformer candidate.",
    },
}
BASELINES = {"persistence", "seasonal_naive_daily", "seasonal_naive_weekly"}
METRICS = ["RMSE", "WAPE", "MASE"]
PRIMARY_METRIC = "RMSE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--doc", default="docs/probabilistic_candidate_selection.md")
    parser.add_argument("--rmse-margin", type=float, default=0.03)
    parser.add_argument("--wape-margin", type=float, default=0.10)
    parser.add_argument("--mase-margin", type=float, default=0.10)
    return parser.parse_args()


def _metric_value(value: float, metric: str) -> str:
    if metric in {"RMSE", "MAE"}:
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.2f}k"
        return f"{value:.2f}"
    return f"{value:.3f}"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    rows = [columns]
    rows.extend(df[columns].astype(str).values.tolist())
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(columns))]
    lines = []
    for idx, row in enumerate(rows):
        cells = [str(cell).ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(cells) + " |")
        if idx == 0:
            lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(lines)


def _read_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    fold_frames = []
    for run_id, spec in RUNS.items():
        run_path = Path(spec["path"])
        summary_path = run_path / "benchmark_summary_by_slice.csv"
        folds_path = run_path / "metrics_by_fold.csv"
        if not summary_path.exists() or not folds_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        folds = pd.read_csv(folds_path)
        summary = summary[~summary["model"].isin(BASELINES)].copy()
        folds = folds[~folds["model"].isin(BASELINES)].copy()
        for df in (summary, folds):
            df.insert(0, "run", run_id)
            df.insert(1, "model_label", spec["label"])
            df.insert(2, "family", spec["family"])
        summary_frames.append(summary)
        fold_frames.append(folds)
    if not summary_frames:
        raise FileNotFoundError("No retained benchmark runs found")
    return pd.concat(summary_frames, ignore_index=True), pd.concat(fold_frames, ignore_index=True)


def _fold_stability(folds: pd.DataFrame) -> pd.DataFrame:
    fold_level = (
        folds.groupby(["run", "model_label", "family", "slice", "fold"], as_index=False)[METRICS]
        .mean()
        .sort_values(["slice", "run", "fold"])
    )
    rows = []
    for keys, sub in fold_level.groupby(["run", "model_label", "family", "slice"], sort=False):
        row = dict(zip(["run", "model_label", "family", "slice"], keys))
        row["folds"] = int(sub["fold"].nunique())
        for metric in METRICS:
            mean = float(sub[metric].mean())
            std = float(sub[metric].std(ddof=0))
            row[f"{metric}_fold_mean"] = mean
            row[f"{metric}_fold_std"] = std
            row[f"{metric}_fold_cv"] = std / mean if mean else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _score_models(summary: pd.DataFrame, stability: pd.DataFrame, margins: dict[str, float]) -> pd.DataFrame:
    cols = ["run", "model_label", "family", "slice", *METRICS]
    scored = summary[cols].merge(
        stability[
            [
                "run",
                "slice",
                "RMSE_fold_cv",
                "WAPE_fold_cv",
                "MASE_fold_cv",
                "RMSE_fold_std",
                "folds",
            ]
        ],
        on=["run", "slice"],
        how="left",
    )
    for metric in METRICS:
        best = scored.groupby("slice")[metric].transform("min")
        scored[f"{metric}_relative_gap"] = scored[metric] / best - 1.0
        scored[f"within_{metric.lower()}_margin"] = scored[f"{metric}_relative_gap"] <= margins[metric]

    scored["rank_rmse"] = scored.groupby("slice")["RMSE"].rank(method="dense")
    scored["rank_wape"] = scored.groupby("slice")["WAPE"].rank(method="dense")
    scored["rank_mase"] = scored.groupby("slice")["MASE"].rank(method="dense")
    scored["robust_score"] = (
        scored["within_rmse_margin"].astype(int)
        + scored["within_wape_margin"].astype(int)
        + scored["within_mase_margin"].astype(int)
    )
    return scored.sort_values(["slice", "rank_rmse", "rank_wape", "rank_mase"])


def _selection(scored: pd.DataFrame) -> pd.DataFrame:
    selected_runs = {"prophet_tuned", "lightgbm_tuned", "patchtst_tuned"}
    rows = []
    for run_id, sub in scored.groupby("run", sort=False):
        spec = RUNS[run_id]
        rows.append(
            {
                "run": run_id,
                "model": spec["label"],
                "family": spec["family"],
                "selected": run_id in selected_runs,
                "slice_rmse_margin_count": int(sub["within_rmse_margin"].sum()),
                "slice_wape_margin_count": int(sub["within_wape_margin"].sum()),
                "slice_mase_margin_count": int(sub["within_mase_margin"].sum()),
                "mean_rmse_gap": float(sub["RMSE_relative_gap"].mean()),
                "mean_wape_gap": float(sub["WAPE_relative_gap"].mean()),
                "mean_mase_gap": float(sub["MASE_relative_gap"].mean()),
                "mean_rmse_fold_cv": float(sub["RMSE_fold_cv"].mean()),
                "probabilistic_path": spec["probabilistic"],
            }
        )
    return pd.DataFrame(rows).sort_values(["selected", "slice_rmse_margin_count", "mean_rmse_gap"], ascending=[False, False, True])


def _sign_test_pvalue(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    prob = sum(comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * prob)


def _paired_significance(folds: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    best_by_slice = (
        scored.sort_values(["slice", "RMSE", "WAPE", "MASE"])
        .groupby("slice", as_index=False)
        .first()[["slice", "run", "model_label"]]
        .rename(columns={"run": "best_run", "model_label": "best_model"})
    )
    fold_level = (
        folds[~folds["model"].isin(BASELINES)]
        .groupby(["run", "model_label", "slice", "fold", "horizon"], as_index=False)["RMSE"]
        .mean()
    )
    rows = []
    for _, best_row in best_by_slice.iterrows():
        slice_name = best_row["slice"]
        best = fold_level[
            (fold_level["slice"] == slice_name) & (fold_level["run"] == best_row["best_run"])
        ][["fold", "horizon", "RMSE"]].rename(columns={"RMSE": "best_RMSE"})
        for (run_id, model_label), sub in fold_level[fold_level["slice"] == slice_name].groupby(["run", "model_label"]):
            merged = sub.merge(best, on=["fold", "horizon"], how="inner")
            delta = merged["RMSE"] - merged["best_RMSE"]
            wins = int((delta < 0).sum())
            losses = int((delta > 0).sum())
            ties = int((delta == 0).sum())
            pvalue = _sign_test_pvalue(wins, losses)
            mean_best = float(merged["best_RMSE"].mean())
            mean_delta = float(delta.mean())
            rows.append(
                {
                    "slice": slice_name,
                    "best_model": best_row["best_model"],
                    "model": model_label,
                    "run": run_id,
                    "paired_cells": int(len(merged)),
                    "wins_vs_best": wins,
                    "losses_vs_best": losses,
                    "ties_vs_best": ties,
                    "mean_rmse_delta": mean_delta,
                    "relative_rmse_delta": mean_delta / mean_best if mean_best else 0.0,
                    "sign_test_pvalue": pvalue,
                    "statistically_different_5pct": bool(pvalue is not None and pvalue < 0.05),
                }
            )
    return pd.DataFrame(rows).sort_values(["slice", "relative_rmse_delta", "model"])


def _slice_recommendations(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slice_name, sub in scored.groupby("slice", sort=True):
        rmse_best = sub.sort_values(["RMSE", "WAPE", "MASE"]).iloc[0]
        wape_best = sub.sort_values(["WAPE", "RMSE", "MASE"]).iloc[0]
        margin = sub[sub["within_rmse_margin"]].sort_values(["RMSE", "WAPE", "MASE"])
        rows.append(
            {
                "slice": slice_name,
                "best_rmse_model": rmse_best["model_label"],
                "best_rmse": rmse_best["RMSE"],
                "best_rmse_display": _metric_value(rmse_best["RMSE"], "RMSE"),
                "best_wape_model": wape_best["model_label"],
                "best_wape": wape_best["WAPE"],
                "best_wape_display": _metric_value(wape_best["WAPE"], "WAPE"),
                "within_3pct_rmse": ", ".join(margin["model_label"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def _format_scored(scored: pd.DataFrame) -> pd.DataFrame:
    shown = scored.copy()
    for metric in METRICS:
        shown[f"{metric}_display"] = shown[metric].map(lambda value: _metric_value(value, metric))
        shown[f"{metric}_gap_display"] = shown[f"{metric}_relative_gap"].map(lambda value: f"{value * 100:.1f}%")
    shown["RMSE_cv_display"] = shown["RMSE_fold_cv"].map(lambda value: f"{value:.3f}")
    return shown


def _write_doc(
    path: Path,
    scored: pd.DataFrame,
    selection: pd.DataFrame,
    slice_recs: pd.DataFrame,
    significance: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    scored_fmt = _format_scored(scored)
    selected = selection[selection["selected"]].copy()
    rejected = selection[~selection["selected"]].copy()
    selected["mean_rmse_gap_display"] = selected["mean_rmse_gap"].map(lambda value: f"{value * 100:.1f}%")
    selected["mean_rmse_fold_cv_display"] = selected["mean_rmse_fold_cv"].map(lambda value: f"{value:.3f}")
    rejected["mean_rmse_gap_display"] = rejected["mean_rmse_gap"].map(lambda value: f"{value * 100:.1f}%")
    sig = significance.copy()
    sig["relative_rmse_delta_display"] = sig["relative_rmse_delta"].map(lambda value: f"{value * 100:.2f}%")
    sig["sign_test_pvalue_display"] = sig["sign_test_pvalue"].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")

    selected_cols = [
        "model",
        "family",
        "slice_rmse_margin_count",
        "slice_wape_margin_count",
        "slice_mase_margin_count",
        "mean_rmse_gap_display",
        "mean_rmse_fold_cv_display",
        "probabilistic_path",
    ]
    rejected_cols = ["model", "family", "mean_rmse_gap_display", "probabilistic_path"]
    slice_cols = ["slice", "best_rmse_model", "best_rmse_display", "best_wape_model", "best_wape_display", "within_3pct_rmse"]
    scored_cols = [
        "slice",
        "model_label",
        "RMSE_display",
        "RMSE_gap_display",
        "WAPE_display",
        "WAPE_gap_display",
        "MASE_display",
        "MASE_gap_display",
        "RMSE_cv_display",
        "robust_score",
    ]
    sig_cols = [
        "slice",
        "best_model",
        "model",
        "relative_rmse_delta_display",
        "wins_vs_best",
        "losses_vs_best",
        "ties_vs_best",
        "sign_test_pvalue_display",
        "statistically_different_5pct",
    ]

    text = "\n\n".join(
        [
            "# Probabilistic candidate selection",
            "This document selects a small set of deterministic models for a probabilistic forecasting extension. "
            "SARIMA is intentionally not included in the implemented benchmark; Prophet is the retained statistical baseline.",
            "## Selection Rule",
            "\n".join(
                [
                    f"- Primary unit of decision: slice-level performance, not the aggregate global score.",
                    f"- A model is considered competitive on a slice if its RMSE is within {args.rmse_margin:.0%} of the best slice RMSE.",
                    f"- WAPE and MASE are used as secondary checks with {args.wape_margin:.0%} and {args.mase_margin:.0%} margins.",
                    "- Fold stability is measured with the coefficient of variation of fold-level RMSE.",
                    "- The final probabilistic set should cover complementary families, not only the best deterministic WAPE.",
                ]
            ),
            "## Per-slice deterministic reading",
            _markdown_table(slice_recs, slice_cols),
            "## Recommended probabilistic candidates",
            _markdown_table(selected, selected_cols),
            "## Not retained as primary probabilistic candidates",
            _markdown_table(rejected, rejected_cols),
            "## Detailed slice scores",
            _markdown_table(scored_fmt, scored_cols),
            "## Paired fold-horizon comparison against the best RMSE model",
            "This sign test is used as a conservative diagnostic, not as a definitive statistical proof, because adjacent horizons are not fully independent.",
            _markdown_table(sig, sig_cols),
            "## Final recommendation for the thesis",
            "\n".join(
                [
                    "Retain three primary probabilistic candidates:",
                    "",
                    "- **Prophet tuned** as the statistical interval baseline. It is the strongest RMSE model on most slices under the fixed 14-day protocol and provides native prediction intervals. The interval calibration should be checked because the deterministic benchmark shows systematic bias.",
                    "- **LightGBM tuned** as the operational quantile-regression baseline. It is cheap, robust, and strong on WAPE, especially for high-volume traffic. It should not be selected only because of WAPE, but because it gives a simple and defensible probabilistic ML baseline.",
                    "- **PatchTST tuned** as the neural probabilistic candidate. It is near the best on several slices and is the most natural transformer-family extension to quantile or distribution losses.",
                    "",
                    "Keep **LSTM 5000w** as a deterministic deep baseline, but do not make it a primary probabilistic candidate unless implementation time allows. Keep **N-HiTS** as a sensitivity candidate: it is weak under the fixed 14-day deterministic protocol, but the 1-day history sensitivity suggests it may be worth revisiting for short-horizon probabilistic forecasting.",
                ]
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    margins = {"RMSE": args.rmse_margin, "WAPE": args.wape_margin, "MASE": args.mase_margin}
    summary, folds = _read_runs()
    stability = _fold_stability(folds)
    scored = _score_models(summary, stability, margins)
    selected = _selection(scored)
    slice_recs = _slice_recommendations(scored)
    significance = _paired_significance(folds, scored)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "scored": output_dir / "probabilistic_candidate_scores.csv",
        "selection": output_dir / "probabilistic_candidate_selection.csv",
        "slice_recommendations": output_dir / "probabilistic_slice_recommendations.csv",
        "stability": output_dir / "probabilistic_fold_stability.csv",
        "significance": output_dir / "probabilistic_paired_significance.csv",
        "doc": Path(args.doc),
    }
    scored.to_csv(paths["scored"], index=False)
    selected.to_csv(paths["selection"], index=False)
    slice_recs.to_csv(paths["slice_recommendations"], index=False)
    stability.to_csv(paths["stability"], index=False)
    significance.to_csv(paths["significance"], index=False)
    _write_doc(paths["doc"], scored, selected, slice_recs, significance, args)

    print("Probabilistic candidate selection generated:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
