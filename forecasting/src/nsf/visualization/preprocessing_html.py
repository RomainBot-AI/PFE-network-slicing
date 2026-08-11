"""Standalone HTML report for preprocessing outputs."""

from __future__ import annotations

import base64
import json
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
    if df.empty:
        return "<p class='muted'>Aucune ligne.</p>"
    return df.head(max_rows).to_html(index=False, classes="data-table", border=0)


def _plot_rows_by_slice(features: pd.DataFrame) -> str:
    data = features.groupby("slice", as_index=False).size().sort_values("size", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(data["slice"], data["size"], color="#2f6f9f")
    ax.set_title("Nombre De Lignes Supervisees Par Slice")
    ax.set_xlabel("")
    ax.set_ylabel("lignes")
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_base64(fig)


def _plot_fold_timeline(folds: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for _, row in folds.iterrows():
        fold = int(row["fold"])
        ax.plot([pd.to_datetime(row["train_start"]), pd.to_datetime(row["train_end"])], [fold, fold], linewidth=8, solid_capstyle="butt", color="#2f6f9f")
        ax.plot([pd.to_datetime(row["target_start"]), pd.to_datetime(row["target_end"])], [fold, fold], linewidth=8, solid_capstyle="butt", color="#c77d3c")
    ax.set_title("Decoupage Temporel Rolling-Origin")
    ax.set_xlabel("temps")
    ax.set_ylabel("fold")
    ax.set_yticks(folds["fold"])
    ax.grid(axis="x", alpha=0.25)
    ax.legend(["train", "target"], frameon=False)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _plot_scaler_std(scalers: pd.DataFrame) -> str:
    data = scalers.groupby("slice", as_index=False)["std"].median().sort_values("std", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(data["slice"], data["std"], color="#5a9f6f")
    ax.set_title("Ecart-Type Median Des Scalers Par Slice")
    ax.set_xlabel("")
    ax.set_ylabel("std median apres log1p")
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_base64(fig)


def _format_bool(value: bool) -> str:
    return "OK" if bool(value) else "A verifier"


def build_preprocessing_html(processed_dir: str | Path, output_html: str | Path | None = None) -> Path:
    processed_path = Path(processed_dir)
    output_path = Path(output_html) if output_html else processed_path / "preprocessing_report.html"

    features = pd.read_csv(processed_path / "features.csv")
    scalers = pd.read_csv(processed_path / "scalers.csv")
    folds = pd.read_csv(processed_path / "folds.csv")
    leakage = pd.read_csv(processed_path / "leakage_audit.csv")
    feature_audit = pd.read_csv(processed_path / "feature_audit.csv")
    meta = json.loads((processed_path / "run_meta.json").read_text(encoding="utf-8"))

    leakage_ok = bool(leakage["train_ends_before_target"].all() and (leakage["train_target_overlap_points"] == 0).all())
    feature_audit_ok = bool(feature_audit["train_end_before_target"].all() and feature_audit["max_lag_available_in_train"].all())
    rows_by_slice = features.groupby("slice", as_index=False).agg(
        rows=("y", "size"),
        series=("unique_id", "nunique"),
        mean_target=("y", "mean"),
        median_target=("y", "median"),
        zero_share=("y", lambda values: float((values == 0).mean())),
    )
    rows_by_horizon = features.groupby("horizon", as_index=False).agg(rows=("y", "size"), mean_target=("y", "mean"))
    scaler_summary = scalers.groupby("slice", as_index=False).agg(
        scalers=("unique_id", "count"),
        mean_log=("mean", "median"),
        std_log=("std", "median"),
    )
    lag_columns = sorted([col for col in features.columns if col.startswith("lag_") and not col.endswith("_scaled")], key=lambda value: int(value.split("_")[1]))
    scaled_lag_columns = [f"{col}_scaled" for col in lag_columns if f"{col}_scaled" in features.columns]
    feature_columns = [col for col in features.columns if col not in {"fold", "unique_id", "slice", "origin_timestamp", "target_timestamp", "y", "y_scaled"}]

    figures = {
        "rows_by_slice": _plot_rows_by_slice(features),
        "folds": _plot_fold_timeline(folds),
        "scaler_std": _plot_scaler_std(scalers),
    }

    cards = [
        ("Lignes features", f"{int(meta['rows']):,}".replace(",", " ")),
        ("Series", str(int(meta["series"]))),
        ("Folds", str(int(meta["folds"]))),
        ("Horizon", f"{int(meta['horizon'])} pas"),
        ("Max lag", f"{int(meta['max_lag'])} pas"),
        ("Leakage", _format_bool(leakage_ok and feature_audit_ok)),
    ]
    cards_html = "\n".join(f"<div class='kpi'><span>{label}</span><strong>{value}</strong></div>" for label, value in cards)

    rows_by_slice_display = rows_by_slice.copy()
    rows_by_slice_display["zero_share"] = rows_by_slice_display["zero_share"].map(lambda value: f"{100 * value:.2f}%")

    html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rapport Preprocessing Forecasting</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f7f8fa; color: #17202a; line-height: 1.45; }}
    header {{ background: #fff; border-bottom: 1px solid #d9dee4; padding: 28px 40px 22px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 34px 0 14px; font-size: 21px; }}
    p {{ margin: 0 0 12px; }}
    code {{ background: #eef1f4; padding: 1px 4px; border-radius: 3px; }}
    .muted {{ color: #5d6a75; }}
    .kpis {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }}
    .kpi {{ background: #fff; border: 1px solid #d9dee4; border-radius: 6px; padding: 14px 16px; }}
    .kpi span {{ display: block; color: #5d6a75; font-size: 13px; margin-bottom: 6px; }}
    .kpi strong {{ font-size: 21px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .figure, .panel {{ background: #fff; border: 1px solid #d9dee4; border-radius: 6px; padding: 14px; overflow-x: auto; }}
    .figure img {{ width: 100%; display: block; }}
    .note {{ background: #eef4f8; border-left: 4px solid #2f6f9f; padding: 14px 16px; border-radius: 4px; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #d9dee4; padding: 7px 8px; text-align: right; }}
    .data-table th:first-child, .data-table td:first-child,
    .data-table th:nth-child(2), .data-table td:nth-child(2) {{ text-align: left; }}
    ul {{ margin-top: 0; }}
    @media (max-width: 980px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .grid {{ grid-template-columns: 1fr; }} header {{ padding: 22px 20px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Rapport Preprocessing Forecasting</h1>
    <p class="muted">Dossier traite : <code>{processed_path}</code></p>
    <div class="kpis">{cards_html}</div>
  </header>
  <main>
    <section class="note">
      <p>Ce preprocessing transforme le panel subnet/slice dense en dataset supervise compatible avec les modeles classiques et les audits temporels.</p>
      <p>Le decoupage reste chronologique : aucun split aleatoire, folds rolling-origin communs, et aucune cible ne chevauche les donnees d'entrainement.</p>
    </section>

    <h2>Structure Generee</h2>
    <div class="grid">
      <div class="figure"><img alt="Lignes par slice" src="data:image/png;base64,{figures['rows_by_slice']}"></div>
      <div class="figure"><img alt="Timeline des folds" src="data:image/png;base64,{figures['folds']}"></div>
    </div>

    <h2>Normalisation</h2>
    <div class="grid">
      <div class="figure"><img alt="Scaler std par slice" src="data:image/png;base64,{figures['scaler_std']}"></div>
      <div class="panel">
        <p><strong>Transformation cible :</strong> <code>log1p_zscore</code>.</p>
        <p>Chaque scaler est estime par <code>fold</code> et par <code>unique_id</code> uniquement sur la fenetre d'entrainement du fold.</p>
        <p>Nombre de scalers : {len(scalers)} = {int(meta['folds'])} folds x {int(meta['series'])} series.</p>
      </div>
    </div>

    <h2>Audit Anti-Leakage</h2>
    <div class="panel">
      <p><strong>Folds sans chevauchement train/target :</strong> {_format_bool(leakage_ok)}.</p>
      <p><strong>Lags disponibles uniquement dans l'historique :</strong> {_format_bool(feature_audit_ok)}.</p>
      <p>Les features temporelles calendaires viennent des timestamps d'origine et de cible. Elles ne dependent pas des valeurs futures de trafic.</p>
    </div>

    <h2>Interpretation</h2>
    <div class="note">
      <p><strong>Unite supervisee.</strong> Une ligne correspond a un couple <code>unique_id, horizon</code> pour un fold donne. Les {int(meta['rows']):,} lignes couvrent {int(meta['series'])} series, {int(meta['folds'])} folds et {int(meta['horizon'])} horizons.</p>
      <p><strong>Historique exploite.</strong> Les lags retenus sont {', '.join(lag_columns)}. Les lags journaliers et hebdomadaires restent explicites, ce qui rend les baselines saisonnieres et LightGBM comparables.</p>
      <p><strong>Scaling.</strong> Les colonnes scalees incluent <code>y_scaled</code> et {len(scaled_lag_columns)} lags normalises. Comme les moyennes/ecarts-types sont appris fold par fold, le scaling ne voit pas les cibles futures.</p>
      <p><strong>Limite.</strong> Ce preprocessing produit surtout un dataset tabulaire. Les modeles sequentiels comme LSTM, N-HiTS et PatchTST peuvent reutiliser les memes folds tout en construisant leurs propres tenseurs.</p>
    </div>

    <h2>Synthese Par Slice</h2>
    <div class="panel">{_table(rows_by_slice_display)}</div>

    <h2>Synthese Par Horizon</h2>
    <div class="panel">{_table(rows_by_horizon, max_rows=40)}</div>

    <h2>Scalers Par Slice</h2>
    <div class="panel">{_table(scaler_summary)}</div>

    <h2>Folds</h2>
    <div class="panel">{_table(folds)}</div>

    <h2>Colonnes De Features</h2>
    <div class="panel">
      <p>{', '.join(feature_columns)}</p>
    </div>
  </main>
</body>
</html>
"""
    ensure_parent(output_path).write_text(html, encoding="utf-8")
    return output_path
