#!/usr/bin/env python3
"""Create final forecasting tables and figures for the thesis report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"


DETERMINISTIC_LABELS = {
    "prophet_tuned": "Prophet 14d",
    "lightgbm_tuned": "LightGBM 14d",
    "lstm_5000w": "LSTM 14d",
    "nhits_tuned": "N-HiTS 14d",
    "patchtst_tuned": "PatchTST 14d",
}

DETERMINISTIC_MODELS = {
    "prophet_tuned": "prophet",
    "lightgbm_tuned": "lightgbm",
    "lstm_5000w": "lstm",
    "nhits_tuned": "nhits",
    "patchtst_tuned": "patchtst",
}


def _fmt_large(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}k"
    return f"{value:.2f}"


def _fmt_float(value: float) -> str:
    return f"{float(value):.3f}"


def _save_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(df[x], df[y], color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_scatter(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(df["interval_width"], df["coverage"], s=55, color="#2f6f73")
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['model_label']} {row['scenario']}",
            (row["interval_width"], row["coverage"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    ax.axhline(0.8, color="#9a3412", linestyle="--", linewidth=1, label="Nominal 80%")
    ax.set_xscale("log")
    ax.set_xlabel("Interval width, log scale")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Probabilistic coverage versus interval width")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def deterministic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    global_df = pd.read_csv(REPORT_DIR / "model_comparison_global.csv")
    by_slice = pd.read_csv(REPORT_DIR / "model_comparison_by_slice.csv")

    mask = global_df.apply(lambda row: DETERMINISTIC_MODELS.get(str(row["run"])) == str(row["model"]), axis=1)
    retained = global_df[mask].copy()
    retained["model_label"] = retained["run"].map(DETERMINISTIC_LABELS)
    retained = retained.sort_values(["RMSE", "WAPE", "MASE"])
    retained_out = retained[["model_label", "RMSE", "WAPE", "MASE"]].copy()
    retained_out["RMSE"] = retained_out["RMSE"].map(_fmt_large)
    retained_out["WAPE"] = retained_out["WAPE"].map(_fmt_float)
    retained_out["MASE"] = retained_out["MASE"].map(_fmt_float)

    slice_rows = []
    mask = by_slice.apply(lambda row: DETERMINISTIC_MODELS.get(str(row["run"])) == str(row["model"]), axis=1)
    for slice_name, sub in by_slice[mask].groupby("slice"):
        best_rmse = sub.loc[sub["RMSE"].idxmin()]
        best_wape = sub.loc[sub["WAPE"].idxmin()]
        best_mase = sub.loc[sub["MASE"].idxmin()]
        slice_rows.append(
            {
                "slice": slice_name,
                "best_rmse": DETERMINISTIC_LABELS.get(str(best_rmse["run"]), str(best_rmse["run"])),
                "RMSE": _fmt_large(best_rmse["RMSE"]),
                "best_wape": DETERMINISTIC_LABELS.get(str(best_wape["run"]), str(best_wape["run"])),
                "WAPE": _fmt_float(best_wape["WAPE"]),
                "best_mase": DETERMINISTIC_LABELS.get(str(best_mase["run"]), str(best_mase["run"])),
                "MASE": _fmt_float(best_mase["MASE"]),
            }
        )
    slice_out = pd.DataFrame(slice_rows).sort_values("slice")
    return retained_out, slice_out


def probabilistic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    global_df = pd.read_csv(REPORT_DIR / "probabilistic_model_comparison_global.csv")
    by_slice = pd.read_csv(REPORT_DIR / "probabilistic_model_comparison_by_slice.csv")

    global_out = global_df[
        ["model_label", "scenario", "RMSE", "WAPE", "MASE", "coverage", "interval_width", "interval_score"]
    ].copy()
    global_out["RMSE"] = global_out["RMSE"].map(_fmt_large)
    global_out["WAPE"] = global_out["WAPE"].map(_fmt_float)
    global_out["MASE"] = global_out["MASE"].map(_fmt_float)
    global_out["coverage"] = global_out["coverage"].map(_fmt_float)
    global_out["interval_width"] = global_out["interval_width"].map(_fmt_large)
    global_out["interval_score"] = global_out["interval_score"].map(_fmt_large)

    slice_rows = []
    for slice_name, sub in by_slice.groupby("slice"):
        best_interval = sub.loc[sub["interval_score"].idxmin()]
        best_rmse = sub.loc[sub["RMSE"].idxmin()]
        best_width = sub.loc[sub["interval_width"].idxmin()]
        slice_rows.append(
            {
                "slice": slice_name,
                "best_interval_score": f"{best_interval['model_label']} {best_interval['scenario']}",
                "interval_score": _fmt_large(best_interval["interval_score"]),
                "coverage": _fmt_float(best_interval["coverage"]),
                "best_median_rmse": f"{best_rmse['model_label']} {best_rmse['scenario']}",
                "RMSE": _fmt_large(best_rmse["RMSE"]),
                "sharpest_interval": f"{best_width['model_label']} {best_width['scenario']}",
                "interval_width": _fmt_large(best_width["interval_width"]),
            }
        )
    slice_out = pd.DataFrame(slice_rows).sort_values("slice")
    return global_out, slice_out


def write_markdown(
    deterministic_global: pd.DataFrame,
    deterministic_slice: pd.DataFrame,
    probabilistic_global: pd.DataFrame,
    probabilistic_slice: pd.DataFrame,
) -> Path:
    def md_table(frame: pd.DataFrame) -> str:
        cols = [str(col) for col in frame.columns]
        rows = [[str(value) for value in row] for row in frame.to_numpy()]
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    path = REPORT_DIR / "final_forecasting_tables.md"
    text = "\n\n".join(
        [
            "# Final forecasting tables",
            "## Deterministic global comparison",
            md_table(deterministic_global),
            "## Deterministic per-slice winners",
            md_table(deterministic_slice),
            "## Probabilistic global comparison",
            md_table(probabilistic_global),
            "## Probabilistic per-slice winners",
            md_table(probabilistic_slice),
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")
    return path


def write_figures() -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(REPORT_DIR / "model_comparison_global.csv")
    mask = det.apply(lambda row: DETERMINISTIC_MODELS.get(str(row["run"])) == str(row["model"]), axis=1)
    det = det[mask].copy()
    det["model_label"] = det["run"].map(DETERMINISTIC_LABELS)
    det = det.sort_values("RMSE")

    prob = pd.read_csv(REPORT_DIR / "probabilistic_model_comparison_global.csv").copy()
    prob = prob.sort_values("interval_score")
    prob["label"] = prob["model_label"] + " " + prob["scenario"]

    paths = [
        FIGURE_DIR / "deterministic_global_rmse.png",
        FIGURE_DIR / "probabilistic_interval_score.png",
        FIGURE_DIR / "probabilistic_coverage_width.png",
    ]
    _save_bar(det, "model_label", "RMSE", "Deterministic global RMSE", "RMSE", paths[0], "#3867a6")
    _save_bar(prob, "label", "interval_score", "Probabilistic interval score", "Interval score", paths[1], "#7a5c2e")
    _save_scatter(prob, paths[2])
    return paths


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    deterministic_global, deterministic_slice = deterministic_tables()
    probabilistic_global, probabilistic_slice = probabilistic_tables()
    table_path = write_markdown(deterministic_global, deterministic_slice, probabilistic_global, probabilistic_slice)
    figure_paths = write_figures()
    print(f"tables: {table_path}")
    for path in figure_paths:
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
