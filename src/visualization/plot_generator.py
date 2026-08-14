#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/visualization/plot_generator.py
 OBJET  : Générateur de Graphiques Explicatifs et Comparatifs Ultra-Lisibles (Publication Quality)
====================================================================================================

DESCRIPTION DÉTAILLÉE :
-----------------------
Génère et sauvegarde les 6 figures explicatives de la simulation sous `./data/plots/<model_name>/` :
  1. 1_traffic_prediction_train_test.png : Trafic Réel Total vs Prédit.
  2. 2_active_slices_timeline.png         : Timeline d'activation des slices (c_final).
  3. 3_slice_bandwidth_allocation.png    : Allocation de bande passante (\rho) lissée par tranche.
  4. 4_qos_violations_analysis.png        : Satisfaction QoS eta_b et suivi des violations.
  5. 5_energy_consumption_train_test.png  : Économies d'énergie en Watts (Baseline All-Active vs PPO+SDN).
  6. 6_ppo_train_vs_test_performance.png  : Courbe de convergence de la récompense PPO.
  7. 7_benchmark_all_models.png           : Graphique comparatif récapitulatif.

====================================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Any


def aggregate_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les métriques Multi-RAN par timestamp unique :
      - SOMME pour le Trafic (total_real_traffic, total_pred_traffic) et la Puissance (f_b_t, f_b_base)
      - MOYENNE pour la QoS (eta_b_t, qos_*), Ratios (\rho_*), Récompense (reward)
      - MAX pour les activations (c_final_*), violations (qos_violation) et surcharge.
    """
    if 'timestamp' not in df.columns or len(df) == 0:
        return df.copy()

    agg_rules = {}
    for col in df.columns:
        if col == 'timestamp':
            continue
        elif col in ['total_real_traffic', 'total_pred_traffic', 'f_b_t', 'f_b_base'] or col.startswith('real_') or col.startswith('pred_'):
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
    """
    Ajoute un bandeau discret de métadonnées d'expérimentation en bas du graphique.
    """
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
    num_subnets: int = 69,
    macro_map: dict = None
):
    """
    Génère et sauvegarde les 6 figures explicatives avec des lignes simples, nettes et épurées.
    """
    model_dir = os.path.join(data_plots_dir, model_name.lower())
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Agrégation globale par timestamp (Somme Trafic/Énergie, Moyenne QoS/Ratios)
    df_train_plt = aggregate_by_timestamp(df_train)
    df_test_plt = aggregate_by_timestamp(df_test)

    time_train = pd.to_datetime(df_train_plt['timestamp'])
    time_test = pd.to_datetime(df_test_plt['timestamp'])
    date_format_year = mdates.DateFormatter('%d/%m/%Y')

    # Fenêtre de lissage dynamique (Moyenne Glissante)
    win_smooth = max(5, min(50, len(df_train_plt) // 100))
    colors_slice = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#1abc9c', '#e74c3c']

    # -------------------------------------------------------------------------
    # Graphique 1 : Trafic Réel Total vs Prédit (Nette & Sans Bruit)
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
    # Graphique 1b : Trafic Prédit vs Réel PAR SLICE
    # -------------------------------------------------------------------------
    fig1b, axes1b = plt.subplots(len(slice_names), 1, figsize=(16, 2.5 * len(slice_names)), sharex=True)
    if len(slice_names) == 1: axes1b = [axes1b]
    
    for idx, s in enumerate(slice_names):
        ax = axes1b[idx]
        real_col, pred_col = f'real_{s}', f'pred_{s}'
        
        if real_col in df_test_plt.columns:
            real_smooth = df_test_plt[real_col].rolling(win_smooth, min_periods=1).mean()
            ax.plot(time_test, real_smooth, color='#145a32', linewidth=2.0, label='Trafic Réel')
            
        if pred_col in df_test_plt.columns and model_name != "passthrough":
            pred_smooth = df_test_plt[pred_col].rolling(win_smooth, min_periods=1).mean()
            ax.plot(time_test, pred_smooth, color='#e74c3c', linestyle='--', linewidth=2.0, label='Prédit')
            
        ax.set_title(f'Prédiction de Trafic - Slice {s} ({model_name.upper()})', fontsize=11, fontweight='bold')
        ax.set_ylabel('Trafic (Unités)', fontsize=9)
        ax.legend(loc='upper right', frameon=True)
        
    axes1b[-1].set_xlabel('Horodatage Réel (%d/%m/%Y)', fontsize=10)
    axes1b[-1].xaxis.set_major_formatter(date_format_year)
    axes1b[-1].tick_params(axis='x', rotation=25)
    
    add_metadata_badge(fig1b, beta, lambda_loss, num_rans, num_subnets)
    fig1b.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig1b, '1b_traffic_prediction_per_slice.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")

    # -------------------------------------------------------------------------
    # Graphique 2 : Timeline de Réallocation Dynamique des Slices (Test Set - Heatmap Matrix)
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
    # Graphique 3 : Allocation de Bande Passante (rho_i) Lissée par Slice (Proportionnel DYNAMIQUE)
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
            
            # Échelle Y Proportionnelle et Dynamique selon les vraies données de la slice !
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
    # Graphique 4 : Satisfaction QoS Global & Barres QoS Moyenne par Slice (Dynamique Proportionnel)
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
    
    # Adaptation dynamique de la limite inférieure Y pour zoomer proprement sur les variations
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
    
    # Adaptation proportionnelle dynamique de l'axe Y des barres QoS
    max_q_val = max(qos_means) if len(qos_means) > 0 else 100.0
    ax4_bot.set_ylim(0, min(118.0, max(20.0, max_q_val * 1.18)))

    for bar, val in zip(bars, qos_means):
        ax4_bot.text(bar.get_x() + bar.get_width()/2.0, val + 2.0, f"{val:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)

    add_metadata_badge(fig4, beta, lambda_loss, num_rans, num_subnets)
    fig4.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig4, '4_qos_violations_analysis.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")

    # -------------------------------------------------------------------------
    # Graphique 4b : Ratios de Latence Normalisée par Slice (SLA Limit = 1.0)
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
    
    # Échelle dynamique adaptée aux ratios
    ax4b.set_ylim(0, 2.5)
    ax4b.legend(loc='upper right', frameon=True)

    add_metadata_badge(fig4b, beta, lambda_loss, num_rans, num_subnets)
    fig4b.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig4b, '4b_slice_latencies_sla.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")

    # -------------------------------------------------------------------------
    # Graphique 5 : Consommation Énergétique Total Cumulée (Watts)
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
    # Graphique 2b & 5b : Vues détaillées par RAN (si num_rans > 1)
    # -------------------------------------------------------------------------
    unique_subnets = df_test['subnet_id'].unique() if 'subnet_id' in df_test.columns else []
    
    if len(unique_subnets) > 1:
        # Graphique 2b : Timeline par RAN
        fig2b, axes2b = plt.subplots(len(unique_subnets), 1, figsize=(16, 4.0 * len(unique_subnets)))
        if len(unique_subnets) == 1: axes2b = [axes2b]
        
        for idx, subnet in enumerate(unique_subnets):
            ax = axes2b[idx]
            df_sub = df_test[df_test['subnet_id'] == subnet].sort_values(by='timestamp')
            time_sub = pd.to_datetime(df_sub['timestamp'])
            
            act_matrix_sub = []
            for s in display_slices:
                col_name = f'c_final_{s}' if s not in ['Eco1', 'Eco2'] else ('c_eco2' if s == 'Eco2' else None)
                if col_name and col_name in df_sub.columns:
                    act_matrix_sub.append(df_sub[col_name].values)
                elif s == 'Eco1':
                    act_matrix_sub.append(np.ones(len(df_sub)))
                else:
                    act_matrix_sub.append(np.zeros(len(df_sub)))
            
            act_matrix_sub = np.array(act_matrix_sub)
            cax = ax.imshow(act_matrix_sub, aspect='auto', cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
            ax.set_yticks(np.arange(len(display_slices)))
            ax.set_yticklabels(display_slices, fontsize=11, fontweight='bold')
            
            if len(time_sub) > 0:
                tick_indices = np.linspace(0, len(time_sub) - 1, min(10, len(time_sub)), dtype=int)
                tick_labels = [time_sub.iloc[i].strftime('%d/%m/%Y\n%H:%M') for i in tick_indices]
                ax.set_xticks(tick_indices)
                ax.set_xticklabels(tick_labels, rotation=0, fontsize=9)
            
            sub_str = ""
            if macro_map:
                subs = [k for k, v in macro_map.items() if v == subnet]
                if len(subs) > 0:
                    subs_txt = ', '.join(map(str, subs))
                    if len(subs_txt) > 80: subs_txt = subs_txt[:77] + "..."
                    sub_str = f" - {len(subs)} Subnets originaux: [{subs_txt}]"

            ax.set_title(f'Timeline Macro-RAN {subnet}{sub_str} ({model_name.upper()})', fontsize=11, fontweight='bold')
            if idx == len(unique_subnets) - 1:
                ax.set_xlabel('Horodatage Réel (%d/%m/%Y)', fontsize=10)
                
        cbar2b = fig2b.colorbar(cax, ax=axes2b, ticks=[0, 1], fraction=0.02, pad=0.04)
        cbar2b.ax.set_yticklabels(['Désactivé (0)', 'Activé (1)'])
        add_metadata_badge(fig2b, beta, lambda_loss, num_rans, num_subnets)
        save_fig(fig2b, '2b_active_slices_per_ran.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")

        # Graphique 5b : Énergie par RAN
        fig5b, axes5b = plt.subplots(len(unique_subnets), 1, figsize=(16, 4.0 * len(unique_subnets)), sharex=True)
        if len(unique_subnets) == 1: axes5b = [axes5b]
        
        for idx, subnet in enumerate(unique_subnets):
            ax = axes5b[idx]
            df_sub = df_test[df_test['subnet_id'] == subnet].sort_values(by='timestamp')
            time_sub = pd.to_datetime(df_sub['timestamp'])
            
            base_smooth_sub = df_sub['f_b_base'].rolling(win_smooth, min_periods=1).mean()
            opt_smooth_sub = df_sub['f_b_t'].rolling(win_smooth, min_periods=1).mean()
            
            ax.plot(time_sub, base_smooth_sub, label='Baseline', color='#e74c3c', linestyle='--', linewidth=2.0)
            ax.plot(time_sub, opt_smooth_sub, label='PPO+SDN', color='#2ecc71', linewidth=2.2)
            ax.fill_between(time_sub, opt_smooth_sub, base_smooth_sub, color='#2ecc71', alpha=0.25)
            
            sub_str = ""
            if macro_map:
                subs = [k for k, v in macro_map.items() if v == subnet]
                if len(subs) > 0:
                    subs_txt = ', '.join(map(str, subs))
                    if len(subs_txt) > 80: subs_txt = subs_txt[:77] + "..."
                    sub_str = f" - {len(subs)} Subnets originaux: [{subs_txt}]"

            ax.set_title(f'Gain Énergétique Macro-RAN {subnet}{sub_str} ({model_name.upper()})', fontsize=11, fontweight='bold')
            ax.set_ylabel('Puissance (W)', fontsize=10)
            ax.legend(loc='upper right', frameon=True)
            
        axes5b[-1].set_xlabel('Horodatage Réel (%d/%m/%Y)', fontsize=10)
        axes5b[-1].xaxis.set_major_formatter(date_format_year)
        axes5b[-1].tick_params(axis='x', rotation=25)
        
        add_metadata_badge(fig5b, beta, lambda_loss, num_rans, num_subnets)
        fig5b.tight_layout(rect=[0, 0.03, 1, 1])
        save_fig(fig5b, '5b_energy_per_ran.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")

    # -------------------------------------------------------------------------
    # Graphique 6 : Convergence de la Récompense Agent PPO
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

    # -------------------------------------------------------------------------
    # Graphique 6b : Convergence de la Récompense PPO par RAN
    # -------------------------------------------------------------------------
    unique_subnets = df_test['subnet_id'].unique() if 'subnet_id' in df_test.columns else []
    if len(unique_subnets) > 1:
        fig6b, axes6b = plt.subplots(len(unique_subnets), 1, figsize=(16, 4.0 * len(unique_subnets)), sharex=True)
        if len(unique_subnets) == 1: axes6b = [axes6b]
        
        for idx, subnet in enumerate(unique_subnets):
            ax = axes6b[idx]
            
            df_train_sub = df_train[df_train['subnet_id'] == subnet].sort_values(by='timestamp')
            df_test_sub = df_test[df_test['subnet_id'] == subnet].sort_values(by='timestamp')
            
            time_train_sub = pd.to_datetime(df_train_sub['timestamp'])
            time_test_sub = pd.to_datetime(df_test_sub['timestamp'])
            
            r_train_sub = df_train_sub['reward'].rolling(win_smooth, min_periods=1).mean()
            r_test_sub = df_test_sub['reward'].rolling(win_smooth, min_periods=1).mean()
            
            ax.plot(time_train_sub, r_train_sub, label='Récompense (Train Set)', color='#1d6f42', linewidth=2.2)
            ax.plot(time_test_sub, r_test_sub, label='Récompense (Test Set)', color='#d35400', linewidth=2.2)
            
            sub_str = ""
            if macro_map:
                subs = [k for k, v in macro_map.items() if v == subnet]
                if len(subs) > 0:
                    subs_txt = ', '.join(map(str, subs))
                    if len(subs_txt) > 80: subs_txt = subs_txt[:77] + "..."
                    sub_str = f" - {len(subs)} Subnets originaux: [{subs_txt}]"
                    
            ax.set_title(f'Récompense PPO Macro-RAN {subnet}{sub_str} ({model_name.upper()})', fontsize=11, fontweight='bold')
            ax.set_ylabel('Récompense $r_t$', fontsize=10)
            ax.legend(loc='upper right', frameon=True)
            
        axes6b[-1].set_xlabel('Horodatage Réel (%d/%m/%Y)', fontsize=10)
        axes6b[-1].xaxis.set_major_formatter(date_format_year)
        axes6b[-1].tick_params(axis='x', rotation=25)
        
        add_metadata_badge(fig6b, beta, lambda_loss, num_rans, num_subnets)
        fig6b.tight_layout(rect=[0, 0.03, 1, 1])
        save_fig(fig6b, '6b_ppo_reward_per_ran.png', model_dir, artifacts_dir, prefix=f"{model_name.lower()}_")


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

    # 1. Gain Énergétique (%) - Échelle Proportionnelle Dynamique
    ax_energy.bar([m.upper() for m in models], energy_gains, color=colors[:len(models)])
    ax_energy.set_title('Gain Énergétique Moyen (%) sur Test Set', fontsize=12, fontweight='bold')
    ax_energy.set_ylabel('Gain Delta E (%)', fontsize=11)
    max_e = max(energy_gains) if len(energy_gains) > 0 else 20.0
    ax_energy.set_ylim(0, max(10.0, max_e * 1.22))

    for i, v in enumerate(energy_gains):
        ax_energy.text(i, v + (max_e * 0.02), f"{v:.1f}%", ha='center', fontweight='bold')

    # 2. Satisfaction QoS (%) - Échelle Proportionnelle Dynamique
    ax_qos.bar([m.upper() for m in models], qos_scores, color=colors[:len(models)])
    ax_qos.set_title('Satisfaction QoS Moyenne (%) sur Test Set', fontsize=12, fontweight='bold')
    ax_qos.set_ylabel('Satisfaction QoS (%)', fontsize=11)
    max_q = max(qos_scores) if len(qos_scores) > 0 else 100.0
    ax_qos.set_ylim(0, min(118.0, max(20.0, max_q * 1.18)))

    for i, v in enumerate(qos_scores):
        ax_qos.text(i, v + (max_q * 0.02), f"{v:.1f}%", ha='center', fontweight='bold')

    # 3. Erreur NMAE Normalisée (%) - Échelle Dynamique Adaptée aux Faibles Erreurs !
    ax_nmae.bar([m.upper() for m in models], nmae_errors, color=colors[:len(models)])
    ax_nmae.set_title('Erreur NMAE Normalisée (%) du Trafic', fontsize=12, fontweight='bold')
    ax_nmae.set_ylabel('NMAE (%)', fontsize=11)
    
    # Fix crucial : Adaptation dynamique au max réel au lieu de bloquer à 100% !
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
    """
    Génère un graphique comparatif récapitulatif entre tous les modèles, mais par station (Macro-RAN).
    """
    models = list(results_summary.keys())
    if not models:
        return
        
    first_res = results_summary[models[0]]
    if 'per_ran_metrics' not in first_res or not first_res['per_ran_metrics']:
        return
        
    rans = sorted(list(first_res['per_ran_metrics'].keys()))
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    ax_energy, ax_qos = axes
    
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c', '#1abc9c']
    width = 0.8 / len(models)
    x = np.arange(len(rans))
    
    # 1. Calcul des maximums globaux pour l'échelle dynamique
    max_e = 20.0
    max_q = 100.0
    for m in models:
        per_ran = results_summary[m].get('per_ran_metrics', {})
        e_gains = [per_ran.get(r, {}).get('energy_gain', 0.0) for r in rans]
        q_scores = [per_ran.get(r, {}).get('qos', 1.0) * 100.0 for r in rans]
        if e_gains: max_e = max(max_e, max(e_gains))
        if q_scores: max_q = max(max_q, max(q_scores))
        
    ax_energy.set_ylim(0, max(10.0, max_e * 1.25))
    ax_qos.set_ylim(0, min(125.0, max(20.0, max_q * 1.25)))
    
    # 2. Tracé des barres groupées et des textes
    for i, m in enumerate(models):
        per_ran = results_summary[m].get('per_ran_metrics', {})
        e_gains = [per_ran.get(r, {}).get('energy_gain', 0.0) for r in rans]
        q_scores = [per_ran.get(r, {}).get('qos', 1.0) * 100.0 for r in rans]
        
        offset = (i - len(models)/2.0 + 0.5) * width
        
        ax_energy.bar(x + offset, e_gains, width, label=m.upper(), color=colors[i % len(colors)])
        ax_qos.bar(x + offset, q_scores, width, label=m.upper(), color=colors[i % len(colors)])
        
        # Ajout des pourcentages au-dessus des barres (comme le graphe 7, mais orienté pour éviter le chevauchement)
        for j, (eg, qs) in enumerate(zip(e_gains, q_scores)):
            ax_energy.text(x[j] + offset, eg + (max_e * 0.02), f"{eg:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)
            ax_qos.text(x[j] + offset, qs + (max_q * 0.02), f"{qs:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)
        
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
    
    fig.suptitle("BENCHMARK COMPARATIF 7B : GAIN & QoS PAR STATION (MACRO-RAN)", fontsize=14, fontweight='bold')
    add_metadata_badge(fig, beta, lambda_loss, num_rans, num_subnets)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, '7b_benchmark_per_ran.png', data_plots_dir, artifacts_dir)


def save_fig(fig, filename, target_dir, *args, **kwargs):
    os.makedirs(target_dir, exist_ok=True)
    path_data = os.path.join(target_dir, filename)
    fig.savefig(path_data, dpi=300)
    plt.close(fig)
