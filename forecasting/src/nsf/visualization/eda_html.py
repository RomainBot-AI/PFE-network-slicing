"""Standalone HTML report for subnet/slice EDA."""

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


def _format_pct_axis(ax) -> None:
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{100 * value:.0f}%")


def _table_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "<p class='muted'>No rows.</p>"
    shown = df.head(max_rows).copy()
    return shown.to_html(index=False, classes="data-table", border=0)


def plot_slice_traffic_share(slice_summary: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data = slice_summary.sort_values("traffic_share", ascending=False)
    ax.bar(data["slice"], data["traffic_share"], color=["#2f6f9f", "#5a9f6f", "#c77d3c", "#7b6aa8"])
    _format_pct_axis(ax)
    ax.set_title("Part Du Trafic Par Slice")
    ax.set_xlabel("")
    ax.set_ylabel("part du volume total")
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_base64(fig)


def plot_zero_share(slice_summary: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data = slice_summary.sort_values("mean_zero_share", ascending=False)
    ax.bar(data["slice"], data["mean_zero_share"], color="#8b4e4e")
    _format_pct_axis(ax)
    ax.set_title("Taux Moyen De Zeros Par Slice")
    ax.set_xlabel("")
    ax.set_ylabel("part de zeros")
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_base64(fig)


def plot_concentration(concentration: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for slice_name, sub in concentration.groupby("slice"):
        ax.plot(sub["top_k_series"], sub["traffic_share"], marker="o", label=slice_name)
    _format_pct_axis(ax)
    ax.set_title("Concentration Du Trafic Dans Chaque Slice")
    ax.set_xlabel("top K series subnet/slice")
    ax.set_ylabel("part du trafic")
    ax.set_xticks(sorted(concentration["top_k_series"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def plot_hourly_profile(hourly: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for slice_name, sub in hourly.groupby("slice"):
        ax.plot(sub["hour"], sub["mean_bytes"], marker="o", linewidth=1.8, label=slice_name)
    ax.set_title("Profil Moyen Du Trafic Par Heure")
    ax.set_xlabel("heure de la journee")
    ax.set_ylabel("octets moyens par subnet/slice")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def plot_weekday_profile(weekday: pd.DataFrame) -> str:
    labels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for slice_name, sub in weekday.groupby("slice"):
        ax.plot(sub["dayofweek"], sub["mean_bytes"], marker="o", linewidth=1.8, label=slice_name)
    ax.set_title("Profil Moyen Du Trafic Par Jour")
    ax.set_xlabel("")
    ax.set_ylabel("octets moyens par subnet/slice")
    ax.set_xticks(range(7), labels)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def plot_autocorrelation(autocorrelation: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for slice_name, sub in autocorrelation.groupby("slice"):
        ax.plot(sub["lag"], sub["median_autocorrelation"], marker="o", linewidth=1.8, label=slice_name)
    ax.set_title("Autocorrelation Mediane Sur Les Plus Grosses Series")
    ax.set_xlabel("lag en pas de 10 minutes")
    ax.set_ylabel("autocorrelation mediane")
    ax.set_xscale("symlog", linthresh=12)
    ax.set_xticks([1, 6, 12, 36, 144, 1008], ["1", "6", "12", "36", "144", "1008"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def build_html(
    slice_summary: pd.DataFrame,
    series_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    hourly: pd.DataFrame,
    weekday: pd.DataFrame,
    autocorrelation: pd.DataFrame,
    output_html: str | Path,
    panel_path: str,
) -> Path:
    output_path = ensure_parent(output_html)
    top_series = series_summary.sort_values("total_bytes", ascending=False).head(15).copy()
    top_series["zero_share"] = top_series["zero_share"].map(lambda value: f"{100 * value:.2f}%")
    top_series["traffic_share_global"] = top_series["total_bytes"] / series_summary["total_bytes"].sum()
    top_series["traffic_share_global"] = top_series["traffic_share_global"].map(lambda value: f"{100 * value:.2f}%")

    figures = {
        "traffic_share": plot_slice_traffic_share(slice_summary),
        "zero_share": plot_zero_share(slice_summary),
        "concentration": plot_concentration(concentration),
        "hourly": plot_hourly_profile(hourly),
        "weekday": plot_weekday_profile(weekday),
        "autocorrelation": plot_autocorrelation(autocorrelation),
    }

    dominant_slice = slice_summary.sort_values("traffic_share", ascending=False).iloc[0]
    highest_zero_slice = slice_summary.sort_values("mean_zero_share", ascending=False).iloc[0]
    embb_row = slice_summary[slice_summary["slice"] == "eMBB"].iloc[0]
    urllc_concentration = concentration[
        (concentration["slice"] == "URLLC") & (concentration["top_k_series"] == 5)
    ]["traffic_share"]
    embb_concentration = concentration[
        (concentration["slice"] == "eMBB") & (concentration["top_k_series"] == 5)
    ]["traffic_share"]
    urllc_top5 = float(urllc_concentration.iloc[0]) if not urllc_concentration.empty else 0.0
    embb_top5 = float(embb_concentration.iloc[0]) if not embb_concentration.empty else 0.0

    cards = [
        ("Series", f"{int(slice_summary['series'].sum())}"),
        ("Subnets", f"{int(slice_summary['subnets'].sum())} slice-links"),
        ("Slice dominante", str(dominant_slice["slice"])),
        ("Zéros max.", f"{100 * float(highest_zero_slice['mean_zero_share']):.2f}%"),
    ]
    cards_html = "\n".join(f"<div class='kpi'><span>{label}</span><strong>{value}</strong></div>" for label, value in cards)

    html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rapport EDA Subnet/Slice</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --text: #17202a;
      --muted: #5d6a75;
      --line: #d9dee4;
      --panel: #ffffff;
      --accent: #2f6f9f;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--text);
      background: var(--bg);
      line-height: 1.45;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 28px 40px 22px;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 34px 0 14px;
      font-size: 21px;
    }}
    p {{
      margin: 0 0 12px;
    }}
    .muted {{
      color: var(--muted);
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .kpi {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px 16px;
    }}
    .kpi span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .kpi strong {{
      font-size: 22px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .figure, .table-wrap {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      overflow-x: auto;
    }}
    .figure img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: right;
    }}
    .data-table th:first-child, .data-table td:first-child,
    .data-table th:nth-child(2), .data-table td:nth-child(2) {{
      text-align: left;
    }}
    .notes {{
      background: #eef4f8;
      border-left: 4px solid var(--accent);
      padding: 14px 16px;
      border-radius: 4px;
    }}
    @media (max-width: 860px) {{
      header {{
        padding: 22px 20px;
      }}
      .kpis, .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Rapport EDA Subnet/Slice</h1>
    <p class="muted">Jeu de donnees panel : <code>{panel_path}</code></p>
    <div class="kpis">{cards_html}</div>
  </header>
  <main>
    <section class="notes">
      <p>Ce rapport analyse uniquement le panel subnet/slice. Les 4 series agregees par slice ne sont pas utilisees comme unite principale de modelisation.</p>
      <p>Les risques methodologiques principaux sont la forte concentration du trafic, la presence frequente de zeros, et des dynamiques differentes selon les slices.</p>
    </section>

    <h2>Synthese Par Slice</h2>
    <div class="grid">
      <div class="figure"><img alt="Part du trafic par slice" src="data:image/png;base64,{figures['traffic_share']}"></div>
      <div class="figure"><img alt="Taux de zeros par slice" src="data:image/png;base64,{figures['zero_share']}"></div>
    </div>

    <h2>Concentration</h2>
    <div class="figure"><img alt="Concentration du trafic" src="data:image/png;base64,{figures['concentration']}"></div>

    <h2>Profils Calendaires</h2>
    <div class="grid">
      <div class="figure"><img alt="Profil horaire" src="data:image/png;base64,{figures['hourly']}"></div>
      <div class="figure"><img alt="Profil par jour de semaine" src="data:image/png;base64,{figures['weekday']}"></div>
    </div>

    <h2>Autocorrelation</h2>
    <div class="figure"><img alt="Synthese de l'autocorrelation" src="data:image/png;base64,{figures['autocorrelation']}"></div>

    <h2>Interpretation Des Resultats</h2>
    <div class="notes">
      <p><strong>Granularite.</strong> Le panel contient {int(slice_summary['series'].sum())} series subnet/slice. Cette granularite est plus pertinente que les 4 series agregees, car elle fournit plusieurs trajectoires temporelles par slice et permet aux modeles globaux ou par-slice d'apprendre des comportements locaux.</p>
      <p><strong>Desequilibre du trafic.</strong> La slice {dominant_slice['slice']} domine tres largement le volume total avec {100 * float(dominant_slice['traffic_share']):.2f}% du trafic. En particulier, eMBB represente {100 * float(embb_row['traffic_share']):.2f}% du volume total. Les metriques globales risquent donc d'etre tirees par eMBB si elles ne sont pas aussi rapportees par slice.</p>
      <p><strong>Concentration interne.</strong> Le trafic n'est pas seulement desequilibre entre slices, il est aussi concentre dans quelques subnets. Les 5 plus grosses series URLLC couvrent {100 * urllc_top5:.2f}% du trafic URLLC, et les 5 plus grosses series eMBB couvrent {100 * embb_top5:.2f}% du trafic eMBB. Cela signifie qu'un modele peut obtenir de bonnes performances agregees tout en ignorant mal les petites series.</p>
      <p><strong>Zeros et intermittence.</strong> La slice la plus intermittente est {highest_zero_slice['slice']}, avec {100 * float(highest_zero_slice['mean_zero_share']):.2f}% de zeros en moyenne. Les erreurs relatives doivent donc etre interpretees avec prudence, car elles deviennent instables lorsque les valeurs reelles sont nulles ou tres faibles.</p>
      <p><strong>Saisonnalite.</strong> Les profils horaires et hebdomadaires montrent si les volumes suivent des rythmes calendaires exploitables. Les lags journaliers, soit 144 pas de 10 minutes, et hebdomadaires, soit 1008 pas, doivent rester des references explicites dans les baselines et les features.</p>
      <p><strong>Consequence pour le benchmark.</strong> Les resultats doivent etre presentes par horizon, par slice, et idealement avec une analyse de stabilite par serie. Une moyenne unique sur tout le panel serait methodologiquement trop faible.</p>
    </div>

    <h2>Table De Synthese Par Slice</h2>
    <div class="table-wrap">{_table_html(slice_summary)}</div>

    <h2>Top Series Subnet/Slice</h2>
    <div class="table-wrap">{_table_html(top_series)}</div>

    <h2>Implications Pour La Modelisation</h2>
    <ul>
      <li>Utiliser subnet/slice comme granularite principale de forecasting.</li>
      <li>Rapporter les metriques par horizon et par slice, avec des controles de stabilite par serie.</li>
      <li>Conserver des baselines saisonnieres journalieres et hebdomadaires explicites.</li>
      <li>Interpreter avec prudence les metriques en pourcentage, surtout pour mMTC et les series tres zero-heavy.</li>
      <li>Appliquer la normalisation et les statistiques de features uniquement sur les folds d'entrainement.</li>
    </ul>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_html_from_report_tables(report_dir: str | Path, output_html: str | Path, panel_path: str) -> Path:
    report_dir = Path(report_dir)
    return build_html(
        slice_summary=pd.read_csv(report_dir / "subnet_slice_eda_slice_summary.csv"),
        series_summary=pd.read_csv(report_dir / "subnet_slice_eda_series_summary.csv"),
        concentration=pd.read_csv(report_dir / "subnet_slice_eda_concentration.csv"),
        hourly=pd.read_csv(report_dir / "subnet_slice_eda_hourly_profile.csv"),
        weekday=pd.read_csv(report_dir / "subnet_slice_eda_weekday_profile.csv"),
        autocorrelation=pd.read_csv(report_dir / "subnet_slice_eda_autocorrelation.csv"),
        output_html=output_html,
        panel_path=panel_path,
    )
