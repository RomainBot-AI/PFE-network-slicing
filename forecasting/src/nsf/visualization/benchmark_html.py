"""HTML report for deterministic benchmark outputs."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nsf.utils.io import ensure_parent


def _fig_to_base64(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _table(df: pd.DataFrame, max_rows: int = 30) -> str:
    return df.head(max_rows).to_html(index=False, classes="data-table", border=0)


def _plot_summary(summary: pd.DataFrame, metric: str) -> str:
    data = summary.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(data["model"], data[metric], color="#2f6f9f")
    ax.set_title(f"{metric} moyen par modele")
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_base64(fig)


def _plot_by_horizon(metrics: pd.DataFrame, metric: str) -> str:
    data = metrics.groupby(["model", "horizon"], as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model, sub in data.groupby("model"):
        ax.plot(sub["horizon"], sub[metric], label=model, linewidth=1.8)
    ax.set_title(f"{metric} moyen par horizon")
    ax.set_xlabel("horizon en pas de 10 minutes")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def _plot_by_slice(metrics: pd.DataFrame, metric: str) -> str:
    data = metrics.groupby(["model", "slice"], as_index=False)[metric].mean()
    pivot = data.pivot(index="slice", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"{metric} moyen par slice")
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def build_benchmark_html(run_dir: str | Path, output_html: str | Path | None = None) -> Path:
    run_path = Path(run_dir)
    output_path = Path(output_html) if output_html else run_path / "benchmark_report.html"
    summary = pd.read_csv(run_path / "benchmark_summary.csv")
    summary_by_slice_path = run_path / "benchmark_summary_by_slice.csv"
    summary_by_slice = pd.read_csv(summary_by_slice_path) if summary_by_slice_path.exists() else pd.DataFrame()
    metrics = pd.read_csv(run_path / "metrics.csv")
    timing = pd.read_csv(run_path / "timing.csv")
    metadata = pd.read_csv(run_path / "model_metadata.csv")
    leakage = pd.read_csv(run_path / "leakage_audit.csv")

    figures = {
        "rmse_summary": _plot_summary(summary, "RMSE"),
        "mase_summary": _plot_summary(summary, "MASE"),
        "rmse_horizon": _plot_by_horizon(metrics, "RMSE"),
        "mase_horizon": _plot_by_horizon(metrics, "MASE"),
        "rmse_slice": _plot_by_slice(metrics, "RMSE"),
    }
    best = summary.sort_values("RMSE").iloc[0]
    leakage_ok = bool(leakage["train_ends_before_target"].all() and (leakage["train_target_overlap_points"] == 0).all())
    html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark Forecasting Deterministe</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f7f8fa; color: #17202a; line-height: 1.45; }}
    header {{ background: #fff; border-bottom: 1px solid #d9dee4; padding: 28px 40px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 32px 0 14px; font-size: 21px; }}
    .muted {{ color: #5d6a75; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .figure, .panel {{ background: #fff; border: 1px solid #d9dee4; border-radius: 6px; padding: 14px; overflow-x: auto; }}
    .figure img {{ width: 100%; display: block; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #d9dee4; padding: 7px 8px; text-align: right; }}
    .data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
    .note {{ background: #eef4f8; border-left: 4px solid #2f6f9f; padding: 14px 16px; border-radius: 4px; }}
    @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} header {{ padding: 22px 20px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Benchmark Forecasting Deterministe</h1>
    <p class="muted">Run: <code>{run_path}</code></p>
  </header>
  <main>
    <section class="note">
      <p><strong>Modele le mieux classe par RMSE moyen :</strong> {best['model']}.</p>
      <p><strong>Audit anti-leakage :</strong> {'OK' if leakage_ok else 'A verifier'}.</p>
      <p>Ce classement moyen doit etre lu avec prudence : les performances doivent aussi etre comparees par slice et par horizon.</p>
    </section>

    <h2>Classement Global</h2>
    <div class="grid">
      <div class="figure"><img alt="RMSE moyen" src="data:image/png;base64,{figures['rmse_summary']}"></div>
      <div class="figure"><img alt="MASE moyen" src="data:image/png;base64,{figures['mase_summary']}"></div>
    </div>

    <h2>Performance Par Horizon</h2>
    <div class="grid">
      <div class="figure"><img alt="RMSE par horizon" src="data:image/png;base64,{figures['rmse_horizon']}"></div>
      <div class="figure"><img alt="MASE par horizon" src="data:image/png;base64,{figures['mase_horizon']}"></div>
    </div>

    <h2>Performance Par Slice</h2>
    <div class="figure"><img alt="RMSE par slice" src="data:image/png;base64,{figures['rmse_slice']}"></div>

    <h2>Tableaux Comparatifs</h2>
    <div class="panel">{_table(summary)}</div>

    <h2>Resume Par Slice</h2>
    <div class="panel">{_table(summary_by_slice, max_rows=40) if not summary_by_slice.empty else "<p>Non genere pour ce run.</p>"}</div>

    <h2>Temps De Calcul</h2>
    <div class="panel">{_table(timing)}</div>

    <h2>Metadonnees Modeles</h2>
    <div class="panel">{_table(metadata)}</div>
  </main>
</body>
</html>
"""
    ensure_parent(output_path).write_text(html, encoding="utf-8")
    return output_path
