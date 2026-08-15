#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure generator for the simulation (saved under ``<output_dir>/<model>/``).

Produces seven figures. Figure text (titles, labels, legends) is kept in French
to match the report/defense:
  1. traffic prediction (real vs predicted)   2. active-slice timeline (c_final)
  3. per-slice bandwidth allocation (rho)      4. QoS satisfaction and violations
  5. energy consumption (all-active vs PPO+SDN) 6. PPO reward convergence
  7. all-models benchmark comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Any


def aggregate_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-RAN metrics per unique timestamp.

    Sum traffic and power (total_*_traffic, f_b_t, f_b_base); max the activation
    and violation flags (c_final_*, qos_violation, surcharge, c_eco2); mean every
    other numeric column (QoS, rho, reward).
    """
    if 'timestamp' not in df.columns or len(df) == 0:
        return df.copy()

    agg_rules = {}
    for col in df.columns:
        if col == 'timestamp':
            continue
        elif col in ['total_real_traffic', 'total_pred_traffic', 'f_b_t', 'f_b_base']:
            agg_rules[col] = 'sum'
        elif col in ['qos_violation', 'surcharge', 'c_eco2'] or col.startswith('c_final_'):
            agg_rules[col] = 'max'
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg_rules[col] = 'mean'
        else:
            agg_rules[col] = 'first'

    df_agg = df.groupby('timestamp', as_index=False).agg(agg_rules)
    df_agg['timestamp'] = pd.to_datetime(df_agg['timestamp'])
    return df_agg.sort_values(by='timestamp').reset_index(drop=True)


def add_metadata_badge(
    fig,
    beta: float = 10.0,
    lambda_loss: float = 50.0,
    num_rans: int = 4,
    num_subnets: int = 69
):
    """Add a small experiment-metadata banner at the bottom of the figure."""
    param_str = f"Paramètres Simulation :  β={beta:.1f} (Poids QoS)  |  λ_loss={lambda_loss:.1f} (Pénalité Loss)  |  Macro-RANs={num_rans}  |  Subnets Bruts={num_subnets}"
    fig.text(
        0.01, 0.005, param_str,
        fontsize=8.0, fontweight='bold', color='#2c3e50',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f1f5f9', edgecolor='#cbd5e1', alpha=0.95)
    )


def generate_all_plots(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    slice_names: List[str],
    data_plots_dir: str,
    artifacts_dir: str,
    model_name: str = "passthrough",
    beta: float = 10.0,
    lambda_loss: float = 50.0,
    num_rans: int = 4,
    num_subnets: int = 69
):
    """Generate and save the per-model explanatory figures."""
    model_dir = os.path.join(data_plots_dir, model_name.lower())
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Aggregate globally per timestamp (sum traffic/energy, mean QoS/ratios).
    df_train_plt = aggregate_by_timestamp(df_train)
    df_test_plt = aggregate_by_timestamp(df_test)

    time_train = pd.to_datetime(df_train_plt['timestamp'])
    time_test = pd.to_datetime(df_test_plt['timestamp'])
    date_format_year = mdates.DateFormatter('%d/%m/%Y')

    # Dynamic smoothing window (rolling mean).
    win_smooth = max(5, min(50, len(df_train_plt) // 100))
    colors_slice = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#1abc9c', '#e74c3c']

    # -------------------------------------------------------------------------
    # Figure 1: total real vs predicted traffic.
    # -------------------------------------------------------------------------
    fig1, (ax1_train, ax1_test) = plt.subplots(1, 2, figsize=(16, 5.2))
    
    # Train Set
    train_real_smooth = df_train_plt['total_real_traffic'].rolling(win_smooth, min_periods=1).mean()
    ax1_train.plot(time_train, train_real_smooth, color='#1b4965', linewidth=2.0, label='Trafic Réel Total')
    
    if model_name != "passthrough":
        train_pred_smooth = df_train_plt['total_pred_traffic'].rolling(win_smooth, min_periods=1).mean()
        ax1_train.plot(time_train, train_pred_smooth, color='#e67e22', linestyle='--', linewidth=2.0, label=f'Prédit ({model_name.upper()})')

    ax1_train.set_title(f'Trafic Réseau Total ({model_name.upper()}) - TRAIN SET ({len(df_train_plt)} pas)', fontsize=11, fontweight='bold')
    ax1_train.set_xlabel('Horodatage Réel (%d/%m/%Y)', fontsize=10)
    ax1_train.set_ylabel('Trafic Total Cumulé (Unités)', fontsize=10)
    ax1_train.xaxis.set_major_formatter(date_format_year)
    ax1_train.tick_params(axis='x', rotation=25)
    ax1_train.legend(loc='upper right', frameon=True)

    # Test Set
    test_real_smooth = df_test_plt['total_real_traffic'].rolling(win_smooth, min_periods=1).mean()
    ax1_test.plot(time_test, test_real_smooth, color='#145a32', linewidth=2.0, label='Trafic Réel Total')

    if model_name != "passthrough":
        test_pred_smooth = df_test_plt['total_pred_traffic'].rolling(win_smooth, min_periods=1).mean()
        ax1_test.plot(time_test, test_pred_smooth, color='#e74c3c', linestyle='--', linewidth=2.0, label=f'Prédit ({model_name.upper()})')

    ax1_test.set_title(f'Trafic Réseau Total ({model_name.upper()}) - TEST SET ({len(df_test_plt)} pas)', fontsize=11, fontweight='bold')
    ax1_test.set_xlabel('Horodatage Réel (%d/%m/%Y)', fontsize=10)
    ax1_test.set_ylabel('Trafic Total Cumulé (Unités)', fontsize=10)
    ax1_test.xaxis.set_major_formatter(date_format_year)
    ax1_test.tick_params(axis='x', rotation=25)
    ax1_test.legend(loc='upper right', frameon=True)

    add_metadata_badge(fig1, beta, lambda_loss, num_rans, num_subnets)
    fig1.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig1, '1_traffic_prediction_train_test.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")

    # -------------------------------------------------------------------------
    # Figure 2: dynamic slice reallocation timeline (test set, heatmap).
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(16, 4.8))
    
    display_slices = slice_names + ['Eco1', 'Eco2']
    act_matrix = []
    for s in display_slices:
        col_name = f'c_final_{s}' if s not in ['Eco1', 'Eco2'] else ('c_eco2' if s == 'Eco2' else None)
        if col_name and col_name in df_test_plt.columns:
            act_matrix.append(df_test_plt[col_name].values)
        elif s == 'Eco1':
            act_matrix.append(np.ones(len(df_test_plt)))
        else:
            act_matrix.append(np.zeros(len(df_test_plt)))

    act_matrix = np.array(act_matrix)
    
    cax = ax2.imshow(act_matrix, aspect='auto', cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_yticks(np.arange(len(display_slices)))
    ax2.set_yticklabels(display_slices, fontsize=11, fontweight='bold')

    if len(time_test) > 0:
        tick_indices = np.linspace(0, len(time_test) - 1, min(10, len(time_test)), dtype=int)
        tick_labels = [time_test.iloc[idx].strftime('%d/%m/%Y\n%H:%M') for idx in tick_indices]
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels(tick_labels, rotation=0, fontsize=9)

    ax2.set_xlabel('Horodatage Réel (%d/%m/%Y)', fontsize=10)
    ax2.set_title(f'Timeline de Réallocation Dynamique des Slices ({model_name.upper()}) (1 = Vert/Actif, 0 = Jaune/Désactivé)', fontsize=12, fontweight='bold')
    
    cbar = fig2.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Désactivé (0)', 'Activé (1)'])

    add_metadata_badge(fig2, beta, lambda_loss, num_rans, num_subnets)
    fig2.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig2, '2_active_slices_timeline.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")

    # -------------------------------------------------------------------------
    # Figure 3: smoothed per-slice bandwidth allocation (rho_i).
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(len(slice_names), 1, figsize=(16, 2.5 * len(slice_names)), sharex=True)
    if len(slice_names) == 1:
        axes3 = [axes3]

    colors_slice = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#1abc9c', '#e74c3c']

    for idx, s in enumerate(slice_names):
        ax = axes3[idx]
        col_rho = f'rho_{s}'
        if col_rho in df_test_plt.columns:
            rho_smooth = df_test_plt[col_rho].rolling(win_smooth, min_periods=1).mean()
            ax.plot(time_test, rho_smooth, color=colors_slice[idx % len(colors_slice)], linewidth=2.0, label=f'Ratio Bande Passante $\\rho_{{{s}}}$')
            ax.fill_between(time_test, 0, rho_smooth, color=colors_slice[idx % len(colors_slice)], alpha=0.15)
            
            # Dynamic proportional Y scale based on the slice's own data.
            max_rho_val = float(rho_smooth.max()) if len(rho_smooth) > 0 else 0.1
            ax.set_ylim(0, max(0.02, max_rho_val * 1.20))
        
        ax.set_title(f'Slice {s} — Ratio de Bande Passante Allocated $\\rho$', fontsize=11, fontweight='bold')
        ax.set_ylabel('Ratio $\\rho_i$', fontsize=9)
        ax.legend(loc='upper right', frameon=True)

    axes3[-1].set_xlabel('Horodatage Réel (%d/%m/%Y)', fontsize=10)
    axes3[-1].xaxis.set_major_formatter(date_format_year)
    axes3[-1].tick_params(axis='x', rotation=25)
    
    add_metadata_badge(fig3, beta, lambda_loss, num_rans, num_subnets)
    fig3.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig3, '3_slice_bandwidth_allocation.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")

    # -------------------------------------------------------------------------
    # Figure 4: global QoS satisfaction and mean per-slice QoS bars.
    # -------------------------------------------------------------------------
    fig4, (ax4_top, ax4_bot) = plt.subplots(2, 1, figsize=(16, 7), gridspec_kw={'height_ratios': [1.3, 1]})

    qos_smooth = df_test_plt['eta_b_t'].rolling(win_smooth, min_periods=1).mean()
    ax4_top.plot(time_test, qos_smooth, color='#4a235a', linewidth=2.2, label='Satisfaction QoS Moyenne $\\eta_b^t$')
    ax4_top.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

    violations_df = df_test_plt[df_test_plt['qos_violation'] == 1]
    qos_min_val = float(qos_smooth.min()) if len(qos_smooth) > 0 else 0.5

    if len(violations_df) > 0:
        ax4_top.scatter(
            pd.to_datetime(violations_df['timestamp']),
            violations_df['eta_b_t'],
            color='#e74c3c', s=25, zorder=5, label=f'Violations QoS ({len(violations_df)} au total)'
        )
        qos_min_val = min(qos_min_val, float(violations_df['eta_b_t'].min()))

    ax4_top.set_title(f'Analyse de la QoS et Violations sous Réallocation Dynamique ({model_name.upper()})', fontsize=12, fontweight='bold')
    ax4_top.set_ylabel('Satisfaction QoS (0.0 - 1.0)', fontsize=10)
    
    # Dynamically adapt the lower Y bound to zoom on the variations.
    ax4_top.set_ylim(max(0.0, qos_min_val - 0.08), 1.05)
    ax4_top.xaxis.set_major_formatter(date_format_year)
    ax4_top.tick_params(axis='x', rotation=25)
    ax4_top.legend(loc='lower left', frameon=True)

    # Barres QoS Moyenne par Slice
    qos_means = [float(df_test_plt[f'qos_{s}'].mean()) * 100.0 if f'qos_{s}' in df_test_plt.columns else 100.0 for s in slice_names]
    bars = ax4_bot.bar(slice_names, qos_means, color=colors_slice[:len(slice_names)], alpha=0.85, width=0.55)
    ax4_bot.axhline(y=100.0, color='gray', linestyle='--', alpha=0.7)
    ax4_bot.set_title('Taux de Satisfaction QoS Moyenne par Slice (%)', fontsize=11, fontweight='bold')
    ax4_bot.set_ylabel('QoS Moyenne par Slice', fontsize=10)
    
    # Dynamic proportional Y axis for the QoS bars.
    max_q_val = max(qos_means) if len(qos_means) > 0 else 100.0
    ax4_bot.set_ylim(0, min(118.0, max(20.0, max_q_val * 1.18)))

    for bar, val in zip(bars, qos_means):
        ax4_bot.text(bar.get_x() + bar.get_width()/2.0, val + 2.0, f"{val:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)

    add_metadata_badge(fig4, beta, lambda_loss, num_rans, num_subnets)
    fig4.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig4, '4_qos_violations_analysis.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")

    # -------------------------------------------------------------------------
    # Figure 4b: normalized per-slice latency ratios (SLA limit = 1.0).
    # -------------------------------------------------------------------------
    fig4b, ax4b = plt.subplots(figsize=(16, 5))
    sla_map = {'URLLC': 1.0, 'URLLC_eMBB_MIX': 5.0, 'eMBB': 10.0, 'mMTC': 20.0}
    delta_active_map = {'URLLC': 0.8, 'URLLC_eMBB_MIX': 4.0, 'eMBB': 8.0, 'mMTC': 15.0}

    for idx, s in enumerate(slice_names):
        sla_val = sla_map.get(s, 20.0)
        c_col = f'c_final_{s}'
        if c_col in df_test_plt.columns:
            d_act = delta_active_map.get(s, 10.0)
            latencies = np.where(df_test_plt[c_col] == 1, d_act, 11.0)
            ratios = latencies / sla_val
            ratios_smooth = pd.Series(ratios, index=df_test_plt.index).rolling(win_smooth, min_periods=1).mean()
            ax4b.plot(time_test, ratios_smooth, color=colors_slice[idx % len(colors_slice)], linewidth=2.0, label=f'Ratio Latence {s} (SLA = {sla_val}ms)')

    ax4b.axhline(y=1.0, color='red', linestyle='--', linewidth=2.0, label='Limite SLA Maximale Tolérée (Ratio = 1.0)')
    ax4b.set_title(f'Ratios de Latence Normalisée / Seuil SLA par Slice ({model_name.upper()})', fontsize=12, fontweight='bold')
    ax4b.set_ylabel('Ratio (Latence / SLA)', fontsize=10)
    ax4b.xaxis.set_major_formatter(date_format_year)
    ax4b.tick_params(axis='x', rotation=25)
    
    # Dynamic scale adapted to the ratios.
    ax4b.set_ylim(0, 2.5)
    ax4b.legend(loc='upper right', frameon=True)

    add_metadata_badge(fig4b, beta, lambda_loss, num_rans, num_subnets)
    fig4b.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig4b, '4b_slice_latencies_sla.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")

    # -------------------------------------------------------------------------
    # Figure 5: total cumulative energy consumption (Watts).
    # -------------------------------------------------------------------------
    fig5, (ax5_train, ax5_test) = plt.subplots(1, 2, figsize=(16, 5.2))

    # Train Set
    train_base_smooth = df_train_plt['f_b_base'].rolling(win_smooth, min_periods=1).mean()
    train_opt_smooth = df_train_plt['f_b_t'].rolling(win_smooth, min_periods=1).mean()

    ax5_train.plot(time_train, train_base_smooth, label='Baseline All-Active (Tous Activés)', color='#e74c3c', linestyle='--', linewidth=2.0)
    ax5_train.plot(time_train, train_opt_smooth, label=f'Réallocation PPO+SDN ({model_name.upper()})', color='#2ecc71', linewidth=2.2)
    ax5_train.fill_between(time_train, train_opt_smooth, train_base_smooth, color='#2ecc71', alpha=0.25, label='Énergie Économisée')
    ax5_train.set_title(f'Gain Énergétique Total Cumulé - TRAIN SET ({model_name.upper()})', fontsize=12, fontweight='bold')
    ax5_train.set_xlabel('Horodatage Réel (%d/%m/%Y)', fontsize=10)
    ax5_train.set_ylabel('Puissance Totale $f_b(t)$ (Watts)', fontsize=10)
    ax5_train.xaxis.set_major_formatter(date_format_year)
    ax5_train.tick_params(axis='x', rotation=25)
    ax5_train.legend(loc='upper right', frameon=True)

    # Test Set
    test_base_smooth = df_test_plt['f_b_base'].rolling(win_smooth, min_periods=1).mean()
    test_opt_smooth = df_test_plt['f_b_t'].rolling(win_smooth, min_periods=1).mean()

    ax5_test.plot(time_test, test_base_smooth, label='Baseline All-Active (Tous Activés)', color='#e74c3c', linestyle='--', linewidth=2.0)
    ax5_test.plot(time_test, test_opt_smooth, label=f'Réallocation PPO+SDN ({model_name.upper()})', color='#2ecc71', linewidth=2.2)
    ax5_test.fill_between(time_test, test_opt_smooth, test_base_smooth, color='#2ecc71', alpha=0.25, label='Énergie Économisée')
    ax5_test.set_title(f'Gain Énergétique Total Cumulé - TEST SET ({model_name.upper()})', fontsize=12, fontweight='bold')
    ax5_test.set_xlabel('Horodatage Réel (%d/%m/%Y)', fontsize=10)
    ax5_test.set_ylabel('Puissance Totale $f_b(t)$ (Watts)', fontsize=10)
    ax5_test.xaxis.set_major_formatter(date_format_year)
    ax5_test.tick_params(axis='x', rotation=25)
    ax5_test.legend(loc='upper right', frameon=True)

    add_metadata_badge(fig5, beta, lambda_loss, num_rans, num_subnets)
    fig5.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig5, '5_energy_consumption_train_test.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")

    # -------------------------------------------------------------------------
    # Figure 6: PPO reward convergence.
    # -------------------------------------------------------------------------
    fig6, ax6 = plt.subplots(figsize=(16, 4.8))
    
    r_train_smooth = df_train_plt['reward'].rolling(win_smooth, min_periods=1).mean()
    r_test_smooth = df_test_plt['reward'].rolling(win_smooth, min_periods=1).mean()

    ax6.plot(time_train, r_train_smooth, label='Récompense PPO (Train Set)', color='#1d6f42', linewidth=2.2)
    ax6.plot(time_test, r_test_smooth, label='Récompense PPO (Test Set)', color='#d35400', linewidth=2.2)

    ax6.set_title(f'Évolution Temporelle de la Récompense PPO ({model_name.upper()}) — CONVERGENCE LISSÉE', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Horodatage Réel (%d/%m/%Y)', fontsize=11)
    ax6.set_ylabel('Récompense Moyenne $r_t$', fontsize=11)
    ax6.xaxis.set_major_formatter(date_format_year)
    ax6.tick_params(axis='x', rotation=25)
    ax6.legend(loc='upper right', frameon=True)

    add_metadata_badge(fig6, beta, lambda_loss, num_rans, num_subnets)
    fig6.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig6, '6_ppo_train_vs_test_performance.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")


def generate_comparison_plot(
    results_summary: Dict[str, Dict[str, Any]],
    data_plots_dir: str,
    artifacts_dir: str,
    beta: float = 10.0,
    lambda_loss: float = 50.0,
    num_rans: int = 4,
    num_subnets: int = 69
):
    """
    Génère et sauvegarde le graphique comparatif récapitulatif entre tous les modèles (Échelles Dynamiques Proportionnelles).
    """
    models = list(results_summary.keys())
    energy_gains = [results_summary[m]['energy_gain_test'] for m in models]
    qos_scores = [results_summary[m]['qos_test'] * 100.0 for m in models]
    nmae_errors = [results_summary[m]['nmae_test'] for m in models]

    fig, (ax_energy, ax_qos, ax_nmae) = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c', '#1abc9c']

    # 1. Energy gain (%) - dynamic proportional scale.
    ax_energy.bar([m.upper() for m in models], energy_gains, color=colors[:len(models)])
    ax_energy.set_title('Gain Énergétique Moyen (%) sur Test Set', fontsize=12, fontweight='bold')
    ax_energy.set_ylabel('Gain Delta E (%)', fontsize=11)
    max_e = max(energy_gains) if len(energy_gains) > 0 else 20.0
    ax_energy.set_ylim(0, max(10.0, max_e * 1.22))

    for i, v in enumerate(energy_gains):
        ax_energy.text(i, v + (max_e * 0.02), f"{v:.1f}%", ha='center', fontweight='bold')

    # 2. QoS satisfaction (%) - dynamic proportional scale.
    ax_qos.bar([m.upper() for m in models], qos_scores, color=colors[:len(models)])
    ax_qos.set_title('Satisfaction QoS Moyenne (%) sur Test Set', fontsize=12, fontweight='bold')
    ax_qos.set_ylabel('Satisfaction QoS (%)', fontsize=11)
    max_q = max(qos_scores) if len(qos_scores) > 0 else 100.0
    ax_qos.set_ylim(0, min(118.0, max(20.0, max_q * 1.18)))

    for i, v in enumerate(qos_scores):
        ax_qos.text(i, v + (max_q * 0.02), f"{v:.1f}%", ha='center', fontweight='bold')

    # 3. NMAE error (%) - dynamic scale adapted to small errors.
    ax_nmae.bar([m.upper() for m in models], nmae_errors, color=colors[:len(models)])
    ax_nmae.set_title('Erreur NMAE Normalisée (%) du Trafic', fontsize=12, fontweight='bold')
    ax_nmae.set_ylabel('NMAE (%)', fontsize=11)
    
    # Adapt dynamically to the real max instead of capping at 100%.
    max_n = max(nmae_errors) if len(nmae_errors) > 0 and max(nmae_errors) > 0 else 1.0
    ax_nmae.set_ylim(0, max(1.0, max_n * 1.25))

    for i, v in enumerate(nmae_errors):
        ax_nmae.text(i, v + (max_n * 0.02), f"{v:.1f}%", ha='center', fontweight='bold')

    fig.suptitle('BENCHMARK COMPARATIF NORMALISÉ DES MODÈLES D\'INTELLIGENCE RÉSEAU', fontsize=14, fontweight='bold')
    add_metadata_badge(fig, beta, lambda_loss, num_rans, num_subnets)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, '7_benchmark_all_models.png', data_plots_dir, artifacts_dir)


def generate_comparison_per_ran_plot(
    results_summary: Dict[str, Dict[str, Any]],
    data_plots_dir: str,
    artifacts_dir: str,
    beta: float = 10.0,
    lambda_loss: float = 50.0,
    num_rans: int = 4,
    num_subnets: int = 69
):
    """Break the model benchmark down per Macro-RAN (energy gain and QoS)."""
    models = list(results_summary.keys())
    if not models:
        return

    first_res = results_summary[models[0]]
    if not first_res.get('per_ran_metrics'):
        return

    rans = sorted(first_res['per_ran_metrics'].keys())

    fig, (ax_energy, ax_qos) = plt.subplots(2, 1, figsize=(16, 12))
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c', '#1abc9c']
    width = 0.8 / len(models)
    x = np.arange(len(rans))

    max_e, max_q = 20.0, 100.0
    for m in models:
        per_ran = results_summary[m].get('per_ran_metrics', {})
        e_gains = [per_ran.get(r, {}).get('energy_gain', 0.0) for r in rans]
        q_scores = [per_ran.get(r, {}).get('qos', 1.0) * 100.0 for r in rans]
        if e_gains:
            max_e = max(max_e, max(e_gains))
        if q_scores:
            max_q = max(max_q, max(q_scores))

    ax_energy.set_ylim(0, max(10.0, max_e * 1.25))
    ax_qos.set_ylim(0, min(125.0, max(20.0, max_q * 1.25)))

    for i, m in enumerate(models):
        per_ran = results_summary[m].get('per_ran_metrics', {})
        e_gains = [per_ran.get(r, {}).get('energy_gain', 0.0) for r in rans]
        q_scores = [per_ran.get(r, {}).get('qos', 1.0) * 100.0 for r in rans]
        offset = (i - len(models) / 2.0 + 0.5) * width

        ax_energy.bar(x + offset, e_gains, width, label=m.upper(), color=colors[i % len(colors)])
        ax_qos.bar(x + offset, q_scores, width, label=m.upper(), color=colors[i % len(colors)])

        # Rotated labels so the grouped bars stay readable with many models.
        for j, (eg, qs) in enumerate(zip(e_gains, q_scores)):
            ax_energy.text(x[j] + offset, eg + (max_e * 0.02), f"{eg:.1f}%",
                           ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)
            ax_qos.text(x[j] + offset, qs + (max_q * 0.02), f"{qs:.1f}%",
                        ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

    ax_energy.set_title('Gain Énergétique par Macro-RAN', fontsize=12, fontweight='bold')
    ax_energy.set_ylabel('Gain Énergie (%)', fontsize=11)
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels([f"RAN {r}" for r in rans])
    ax_energy.legend(loc='upper right')
    ax_energy.grid(True, linestyle='--', alpha=0.5, axis='y')

    ax_qos.set_title('Satisfaction QoS par Macro-RAN', fontsize=12, fontweight='bold')
    ax_qos.set_ylabel('QoS (%)', fontsize=11)
    ax_qos.set_xticks(x)
    ax_qos.set_xticklabels([f"RAN {r}" for r in rans])
    ax_qos.legend(loc='lower right')
    ax_qos.grid(True, linestyle='--', alpha=0.5, axis='y')

    fig.suptitle('BENCHMARK COMPARATIF 7B : GAIN & QoS PAR STATION (MACRO-RAN)', fontsize=14, fontweight='bold')
    add_metadata_badge(fig, beta, lambda_loss, num_rans, num_subnets)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, '7b_benchmark_per_ran.png', data_plots_dir, artifacts_dir)


def save_fig(fig, filename, target_dir, artifacts_dir, prefix: str = ""):
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    path_data = os.path.join(target_dir, filename)
    path_artifact = os.path.join(artifacts_dir, f"{prefix}{filename}")
    fig.savefig(path_data, dpi=300)
    fig.savefig(path_artifact, dpi=300)
    plt.close(fig)
