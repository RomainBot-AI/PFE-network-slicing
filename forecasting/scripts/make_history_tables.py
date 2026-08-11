#!/usr/bin/env python3
"""Create thesis-ready tables for input-history sensitivity results."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


LEARNED_MODELS = {"prophet", "lightgbm", "lstm", "nhits", "patchtst"}
HISTORY_ORDER = {"1d": 0, "7d": 1, "14d": 2}
MODEL_LABELS = {
    "prophet": "Prophet tuned",
    "lightgbm": "LightGBM tuned",
    "lstm": "LSTM 5000w",
    "nhits": "N-HiTS tuned",
    "patchtst": "PatchTST tuned",
    "persistence": "Persistence",
    "seasonal_naive_daily": "Seasonal naive daily",
    "seasonal_naive_weekly": "Seasonal naive weekly",
}
METRICS = ["RMSE", "WAPE", "MASE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="reports")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def _history_sort_value(history: str) -> int:
    return HISTORY_ORDER.get(history, 99)


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


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
            sep = ["-" * widths[i] for i in range(len(columns))]
            lines.append("| " + " | ".join(sep) + " |")
    return "\n".join(lines)


def _html_table(df: pd.DataFrame, columns: list[str], classes: str = "data-table") -> str:
    shown = df[columns].copy()
    return shown.to_html(index=False, classes=classes, border=0, escape=True)


def _vote_summary(values: pd.Series) -> str:
    votes = values.value_counts()
    max_votes = votes.iloc[0]
    leaders = sorted(votes[votes == max_votes].index.tolist())
    return ", ".join(leaders), int(max_votes)


def _prepare_global(global_df: pd.DataFrame) -> pd.DataFrame:
    df = global_df.copy()
    df["is_learned"] = df["model"].isin(LEARNED_MODELS)
    df["model_label"] = df["model"].map(_model_label)
    df["history_sort"] = df["history"].map(_history_sort_value)
    df = df.sort_values(["history_sort", "RMSE", "WAPE", "MASE"]).reset_index(drop=True)
    for metric in METRICS:
        df[f"rank_{metric.lower()}_by_history"] = df.groupby("history")[metric].rank(method="dense")
    return df


def _learned_global_table(global_df: pd.DataFrame) -> pd.DataFrame:
    learned = global_df[global_df["is_learned"]].copy()
    learned = learned.sort_values(["history_sort", "RMSE", "WAPE", "MASE"]).reset_index(drop=True)
    for metric in METRICS:
        learned[f"{metric}_display"] = learned[metric].map(lambda value: _metric_value(value, metric))
    return learned[
        [
            "history",
            "model_label",
            "RMSE",
            "WAPE",
            "MASE",
            "RMSE_display",
            "WAPE_display",
            "MASE_display",
        ]
    ]


def _best_by_history(global_df: pd.DataFrame) -> pd.DataFrame:
    learned = global_df[global_df["is_learned"]].copy()
    rows = []
    for history, hist_df in learned.groupby("history", sort=False):
        for metric in METRICS:
            best = hist_df.sort_values([metric, "RMSE", "WAPE", "MASE"]).iloc[0]
            rows.append(
                {
                    "history": history,
                    "criterion": metric,
                    "best_model": best["model_label"],
                    "value": best[metric],
                    "value_display": _metric_value(best[metric], metric),
                }
            )
    out = pd.DataFrame(rows)
    out["history_sort"] = out["history"].map(_history_sort_value)
    return out.sort_values(["history_sort", "criterion"]).drop(columns=["history_sort"])


def _best_history_by_model(global_df: pd.DataFrame) -> pd.DataFrame:
    learned = global_df[global_df["is_learned"]].copy()
    rows = []
    for model, model_df in learned.groupby("model", sort=False):
        row = {"model": model, "model_label": _model_label(model)}
        for metric in METRICS:
            best = model_df.sort_values([metric, "history_sort"]).iloc[0]
            row[f"best_history_{metric.lower()}"] = best["history"]
            row[f"best_{metric.lower()}"] = best[metric]
            row[f"best_{metric.lower()}_display"] = _metric_value(best[metric], metric)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("best_rmse")


def _slice_winners(by_slice_df: pd.DataFrame) -> pd.DataFrame:
    learned = by_slice_df[by_slice_df["model"].isin(LEARNED_MODELS)].copy()
    learned["model_label"] = learned["model"].map(_model_label)
    learned["history_sort"] = learned["history"].map(_history_sort_value)
    rows = []
    for (history, slice_name), slice_df in learned.groupby(["history", "slice"], sort=False):
        row = {"history": history, "slice": slice_name}
        for metric in METRICS:
            best = slice_df.sort_values([metric, "RMSE", "WAPE", "MASE"]).iloc[0]
            row[f"best_{metric.lower()}_model"] = best["model_label"]
            row[f"best_{metric.lower()}"] = best[metric]
            row[f"best_{metric.lower()}_display"] = _metric_value(best[metric], metric)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["history_sort"] = out["history"].map(_history_sort_value)
    return out.sort_values(["history_sort", "slice"]).drop(columns=["history_sort"])


def _best_by_slice_all_histories(by_slice_df: pd.DataFrame) -> pd.DataFrame:
    learned = by_slice_df[by_slice_df["model"].isin(LEARNED_MODELS)].copy()
    learned["model_label"] = learned["model"].map(_model_label)
    learned["history_sort"] = learned["history"].map(_history_sort_value)
    rows = []
    for slice_name, slice_df in learned.groupby("slice", sort=True):
        row = {"slice": slice_name}
        for metric in METRICS:
            best = slice_df.sort_values([metric, "history_sort", "RMSE", "WAPE", "MASE"]).iloc[0]
            row[f"best_{metric.lower()}_history"] = best["history"]
            row[f"best_{metric.lower()}_model"] = best["model_label"]
            row[f"best_{metric.lower()}"] = best[metric]
            row[f"best_{metric.lower()}_display"] = _metric_value(best[metric], metric)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("slice")


def _best_history_by_slice_model(by_slice_df: pd.DataFrame) -> pd.DataFrame:
    learned = by_slice_df[by_slice_df["model"].isin(LEARNED_MODELS)].copy()
    learned["model_label"] = learned["model"].map(_model_label)
    learned["history_sort"] = learned["history"].map(_history_sort_value)
    rows = []
    for (slice_name, model), model_df in learned.groupby(["slice", "model"], sort=True):
        row = {"slice": slice_name, "model": model, "model_label": _model_label(model)}
        for metric in METRICS:
            best = model_df.sort_values([metric, "history_sort"]).iloc[0]
            row[f"best_history_{metric.lower()}"] = best["history"]
            row[f"best_{metric.lower()}"] = best[metric]
            row[f"best_{metric.lower()}_display"] = _metric_value(best[metric], metric)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["slice", "best_rmse", "model_label"])


def _write_markdown(
    path: Path,
    learned_global: pd.DataFrame,
    best_by_history: pd.DataFrame,
    best_history_by_model: pd.DataFrame,
    slice_winners: pd.DataFrame,
    best_by_slice_all_histories: pd.DataFrame,
    best_history_by_slice_model: pd.DataFrame,
) -> None:
    compact_global = learned_global[
        ["history", "model_label", "RMSE_display", "WAPE_display", "MASE_display"]
    ].rename(
        columns={
            "history": "History",
            "model_label": "Model",
            "RMSE_display": "RMSE",
            "WAPE_display": "WAPE",
            "MASE_display": "MASE",
        }
    )
    compact_best = best_by_history[["history", "criterion", "best_model", "value_display"]].rename(
        columns={
            "history": "History",
            "criterion": "Criterion",
            "best_model": "Best model",
            "value_display": "Value",
        }
    )
    compact_history = best_history_by_model[
        [
            "model_label",
            "best_history_rmse",
            "best_rmse_display",
            "best_history_wape",
            "best_wape_display",
            "best_history_mase",
            "best_mase_display",
        ]
    ].rename(
        columns={
            "model_label": "Model",
            "best_history_rmse": "Best RMSE history",
            "best_rmse_display": "RMSE",
            "best_history_wape": "Best WAPE history",
            "best_wape_display": "WAPE",
            "best_history_mase": "Best MASE history",
            "best_mase_display": "MASE",
        }
    )
    compact_slice = slice_winners[
        [
            "history",
            "slice",
            "best_rmse_model",
            "best_rmse_display",
            "best_wape_model",
            "best_wape_display",
            "best_mase_model",
            "best_mase_display",
        ]
    ].rename(
        columns={
            "history": "History",
            "slice": "Slice",
            "best_rmse_model": "Best RMSE model",
            "best_rmse_display": "RMSE",
            "best_wape_model": "Best WAPE model",
            "best_wape_display": "WAPE",
            "best_mase_model": "Best MASE model",
            "best_mase_display": "MASE",
        }
    )
    compact_slice_overall = best_by_slice_all_histories[
        [
            "slice",
            "best_rmse_history",
            "best_rmse_model",
            "best_rmse_display",
            "best_wape_history",
            "best_wape_model",
            "best_wape_display",
            "best_mase_history",
            "best_mase_model",
            "best_mase_display",
        ]
    ].rename(
        columns={
            "slice": "Slice",
            "best_rmse_history": "RMSE history",
            "best_rmse_model": "RMSE model",
            "best_rmse_display": "RMSE",
            "best_wape_history": "WAPE history",
            "best_wape_model": "WAPE model",
            "best_wape_display": "WAPE",
            "best_mase_history": "MASE history",
            "best_mase_model": "MASE model",
            "best_mase_display": "MASE",
        }
    )
    compact_slice_model = best_history_by_slice_model[
        [
            "slice",
            "model_label",
            "best_history_rmse",
            "best_rmse_display",
            "best_history_wape",
            "best_wape_display",
            "best_history_mase",
            "best_mase_display",
        ]
    ].rename(
        columns={
            "slice": "Slice",
            "model_label": "Model",
            "best_history_rmse": "Best RMSE history",
            "best_rmse_display": "RMSE",
            "best_history_wape": "Best WAPE history",
            "best_wape_display": "WAPE",
            "best_history_mase": "Best MASE history",
            "best_mase_display": "MASE",
        }
    )

    rmse_leaders, rmse_count = _vote_summary(best_by_slice_all_histories["best_rmse_model"])
    mase_leaders, mase_count = _vote_summary(best_by_slice_all_histories["best_mase_model"])
    wape_leaders, wape_count = _vote_summary(best_by_slice_all_histories["best_wape_model"])
    recommendation_lines = []
    for _, row in best_by_slice_all_histories.iterrows():
        recommendation_lines.append(
            f"- `{row['slice']}`: RMSE -> {row['best_rmse_model']} ({row['best_rmse_history']}); "
            f"WAPE -> {row['best_wape_model']} ({row['best_wape_history']}); "
            f"MASE -> {row['best_mase_model']} ({row['best_mase_history']})."
        )
    conclusion = "\n".join(
        [
            "## Conclusion",
            "",
            f"Per-slice RMSE winners are led by **{rmse_leaders}** "
            f"({rmse_count} of {len(best_by_slice_all_histories)} slices).",
            f"Per-slice WAPE winners are led by **{wape_leaders}** "
            f"({wape_count} of {len(best_by_slice_all_histories)} slices).",
            f"Per-slice MASE winners are tied between **{mase_leaders}** "
            f"({mase_count} slice each among the leaders).",
            "",
            "Per-slice selection:",
            "\n".join(recommendation_lines),
            "",
            "The choice is therefore slice-dependent: use the per-slice table above for model selection, "
            "not the aggregate global score.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n\n".join(
            [
                "# Input-history sensitivity tables",
                "Histories: `1d` = 144 points, `7d` = 1008 points, `14d` = 2016 points. "
                "All results use the same 6-hour horizon and 5 rolling-origin folds.",
                "Note: MASE for the `1d` LightGBM/deterministic run is not a primary decision metric because "
                "the seasonal scale cannot be estimated normally from a one-day history window. RMSE and WAPE "
                "remain interpretable.",
                "## Best model and history by slice",
                _markdown_table(compact_slice_overall, list(compact_slice_overall.columns)),
                "## Per-slice winners among learned models",
                _markdown_table(compact_slice, list(compact_slice.columns)),
                "## Best history per model and slice",
                _markdown_table(compact_slice_model, list(compact_slice_model.columns)),
                "## Global metrics, appendix",
                _markdown_table(compact_global, list(compact_global.columns)),
                "## Global best learned model by history, appendix",
                _markdown_table(compact_best, list(compact_best.columns)),
                "## Global best history per learned model, appendix",
                _markdown_table(compact_history, list(compact_history.columns)),
                conclusion,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_html(
    path: Path,
    learned_global: pd.DataFrame,
    best_by_history: pd.DataFrame,
    best_history_by_model: pd.DataFrame,
    slice_winners: pd.DataFrame,
    best_by_slice_all_histories: pd.DataFrame,
    best_history_by_slice_model: pd.DataFrame,
) -> None:
    compact_slice_overall = best_by_slice_all_histories[
        [
            "slice",
            "best_rmse_history",
            "best_rmse_model",
            "best_rmse_display",
            "best_wape_history",
            "best_wape_model",
            "best_wape_display",
            "best_mase_history",
            "best_mase_model",
            "best_mase_display",
        ]
    ].rename(
        columns={
            "slice": "Slice",
            "best_rmse_history": "RMSE history",
            "best_rmse_model": "RMSE model",
            "best_rmse_display": "RMSE",
            "best_wape_history": "WAPE history",
            "best_wape_model": "WAPE model",
            "best_wape_display": "WAPE",
            "best_mase_history": "MASE history",
            "best_mase_model": "MASE model",
            "best_mase_display": "MASE",
        }
    )
    compact_slice = slice_winners[
        [
            "history",
            "slice",
            "best_rmse_model",
            "best_rmse_display",
            "best_wape_model",
            "best_wape_display",
            "best_mase_model",
            "best_mase_display",
        ]
    ].rename(
        columns={
            "history": "History",
            "slice": "Slice",
            "best_rmse_model": "Best RMSE model",
            "best_rmse_display": "RMSE",
            "best_wape_model": "Best WAPE model",
            "best_wape_display": "WAPE",
            "best_mase_model": "Best MASE model",
            "best_mase_display": "MASE",
        }
    )
    compact_slice_model = best_history_by_slice_model[
        [
            "slice",
            "model_label",
            "best_history_rmse",
            "best_rmse_display",
            "best_history_wape",
            "best_wape_display",
            "best_history_mase",
            "best_mase_display",
        ]
    ].rename(
        columns={
            "slice": "Slice",
            "model_label": "Model",
            "best_history_rmse": "Best RMSE history",
            "best_rmse_display": "RMSE",
            "best_history_wape": "Best WAPE history",
            "best_wape_display": "WAPE",
            "best_history_mase": "Best MASE history",
            "best_mase_display": "MASE",
        }
    )
    compact_global = learned_global[
        ["history", "model_label", "RMSE_display", "WAPE_display", "MASE_display"]
    ].rename(
        columns={
            "history": "History",
            "model_label": "Model",
            "RMSE_display": "RMSE",
            "WAPE_display": "WAPE",
            "MASE_display": "MASE",
        }
    )
    compact_best = best_by_history[["history", "criterion", "best_model", "value_display"]].rename(
        columns={
            "history": "History",
            "criterion": "Criterion",
            "best_model": "Best model",
            "value_display": "Value",
        }
    )
    compact_history = best_history_by_model[
        [
            "model_label",
            "best_history_rmse",
            "best_rmse_display",
            "best_history_wape",
            "best_wape_display",
            "best_history_mase",
            "best_mase_display",
        ]
    ].rename(
        columns={
            "model_label": "Model",
            "best_history_rmse": "Best RMSE history",
            "best_rmse_display": "RMSE",
            "best_history_wape": "Best WAPE history",
            "best_wape_display": "WAPE",
            "best_history_mase": "Best MASE history",
            "best_mase_display": "MASE",
        }
    )

    rmse_leaders, rmse_count = _vote_summary(best_by_slice_all_histories["best_rmse_model"])
    wape_leaders, wape_count = _vote_summary(best_by_slice_all_histories["best_wape_model"])
    mase_leaders, mase_count = _vote_summary(best_by_slice_all_histories["best_mase_model"])
    recommendation_items = "\n".join(
        "<li>"
        f"<strong>{html.escape(str(row['slice']))}</strong>: "
        f"RMSE -> {html.escape(str(row['best_rmse_model']))} ({html.escape(str(row['best_rmse_history']))}); "
        f"WAPE -> {html.escape(str(row['best_wape_model']))} ({html.escape(str(row['best_wape_history']))}); "
        f"MASE -> {html.escape(str(row['best_mase_model']))} ({html.escape(str(row['best_mase_history']))})."
        "</li>"
        for _, row in best_by_slice_all_histories.iterrows()
    )

    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Input-history sensitivity par slice</title>
  <style>
    :root {{
      --bg: #f5f7f8;
      --panel: #ffffff;
      --text: #182026;
      --muted: #60707d;
      --line: #d8dee4;
      --accent: #21606d;
      --accent-soft: #e7f0f2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 28px 38px;
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 26px 22px 46px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 30px 0 12px; font-size: 20px; }}
    p {{ margin: 0 0 10px; }}
    code {{ background: #edf1f3; padding: 2px 5px; border-radius: 4px; }}
    .muted {{ color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 20px 0 8px;
    }}
    .kpi {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
    }}
    .kpi span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 5px; }}
    .kpi strong {{ display: block; font-size: 18px; }}
    .note {{
      background: var(--accent-soft);
      border-left: 4px solid var(--accent);
      padding: 14px 16px;
      border-radius: 4px;
      margin: 18px 0;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      overflow-x: auto;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 9px;
      text-align: right;
    }}
    .data-table th {{
      background: #f0f3f5;
      color: #25323b;
      font-weight: 700;
    }}
    .data-table th:first-child, .data-table td:first-child,
    .data-table th:nth-child(2), .data-table td:nth-child(2) {{
      text-align: left;
    }}
    ul {{ margin: 8px 0 0 18px; padding: 0; }}
    li {{ margin: 5px 0; }}
    details {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      margin-top: 14px;
    }}
    summary {{ cursor: pointer; font-weight: 700; }}
    @media (max-width: 850px) {{
      header {{ padding: 22px 20px; }}
      .summary {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Input-history sensitivity par slice</h1>
    <p class="muted">Historiques compares: <code>1d</code>, <code>7d</code>, <code>14d</code>. Horizon: 6 heures. Folds: 5.</p>
  </header>
  <main>
    <section class="summary">
      <div class="kpi"><span>RMSE par slice</span><strong>{html.escape(rmse_leaders)}</strong><span>{rmse_count}/4 slices</span></div>
      <div class="kpi"><span>WAPE par slice</span><strong>{html.escape(wape_leaders)}</strong><span>{wape_count}/4 slices</span></div>
      <div class="kpi"><span>MASE par slice</span><strong>{html.escape(mase_leaders)}</strong><span>{mase_count}/4 slices chez les leaders</span></div>
    </section>

    <section class="note">
      <p><strong>Decision:</strong> le classement global est laisse en annexe. Pour le slicing, la selection doit etre faite slice par slice.</p>
      <ul>{recommendation_items}</ul>
    </section>

    <h2>Meilleur modele et historique par slice</h2>
    <div class="panel">{_html_table(compact_slice_overall, list(compact_slice_overall.columns))}</div>

    <h2>Gagnants par historique et par slice</h2>
    <div class="panel">{_html_table(compact_slice, list(compact_slice.columns))}</div>

    <h2>Meilleur historique par modele et par slice</h2>
    <div class="panel">{_html_table(compact_slice_model, list(compact_slice_model.columns))}</div>

    <details>
      <summary>Annexe globale</summary>
      <h2>Metriques globales</h2>
      <div class="panel">{_html_table(compact_global, list(compact_global.columns))}</div>
      <h2>Meilleur modele global par historique</h2>
      <div class="panel">{_html_table(compact_best, list(compact_best.columns))}</div>
      <h2>Meilleur historique global par modele</h2>
      <div class="panel">{_html_table(compact_history, list(compact_history.columns))}</div>
    </details>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    global_df = pd.read_csv(input_dir / "history_sensitivity_global.csv")
    by_slice_df = pd.read_csv(input_dir / "history_sensitivity_by_slice.csv")

    prepared_global = _prepare_global(global_df)
    learned_global = _learned_global_table(prepared_global)
    best_by_history = _best_by_history(prepared_global)
    best_history_by_model = _best_history_by_model(prepared_global)
    slice_winners = _slice_winners(by_slice_df)
    best_by_slice_all_histories = _best_by_slice_all_histories(by_slice_df)
    best_history_by_slice_model = _best_history_by_slice_model(by_slice_df)

    outputs = {
        "global_ranked": output_dir / "history_sensitivity_global_ranked.csv",
        "learned_global": output_dir / "history_sensitivity_learned_global.csv",
        "best_by_history": output_dir / "history_sensitivity_best_by_history.csv",
        "best_history_by_model": output_dir / "history_sensitivity_best_history_by_model.csv",
        "slice_winners": output_dir / "history_sensitivity_slice_winners.csv",
        "best_by_slice": output_dir / "history_sensitivity_best_by_slice.csv",
        "best_history_by_slice_model": output_dir / "history_sensitivity_best_history_by_slice_model.csv",
        "markdown": Path("forecasting/docs/history_sensitivity_tables.md"),
        "html": output_dir / "history_sensitivity_tables.html",
    }

    prepared_global.drop(columns=["history_sort"]).to_csv(outputs["global_ranked"], index=False)
    learned_global.to_csv(outputs["learned_global"], index=False)
    best_by_history.to_csv(outputs["best_by_history"], index=False)
    best_history_by_model.to_csv(outputs["best_history_by_model"], index=False)
    slice_winners.to_csv(outputs["slice_winners"], index=False)
    best_by_slice_all_histories.to_csv(outputs["best_by_slice"], index=False)
    best_history_by_slice_model.to_csv(outputs["best_history_by_slice_model"], index=False)
    _write_markdown(
        outputs["markdown"],
        learned_global,
        best_by_history,
        best_history_by_model,
        slice_winners,
        best_by_slice_all_histories,
        best_history_by_slice_model,
    )
    _write_html(
        outputs["html"],
        learned_global,
        best_by_history,
        best_history_by_model,
        slice_winners,
        best_by_slice_all_histories,
        best_history_by_slice_model,
    )

    print("History tables generated:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
