#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/visualization/generate_pareto.py
 OBJET  : Générateur de la Frontière de Pareto Multi-Objectif (Économies d'Énergie vs QoS)
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce script indépendant permet de tracer la frontière de Pareto des 7 régimes d'expérimentation du PFE.
Il illustre le compromis fondamental (Trade-off) entre l'économie d'énergie (%) et le maintien de la QoS (%)
selon différentes valeurs des hyperparamètres de récompense \beta (poids QoS) et \lambda_{\text{loss}} (pénalité).
====================================================================================================
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Métriques consolidées issues des 7 campagnes d'expérimentation PFE
experiments = {
    'Exp 2 (β=2, λ=10)': {
        'Gain Énergie (%)': 29.49,
        'Satisfaction QoS (%)': 82.92,
        'Color': '#e67e22',
        'Marker': 'o'
    },
    'Exp 5 (β=2, λ=50)': {
        'Gain Énergie (%)': 25.24,
        'Satisfaction QoS (%)': 90.82,
        'Color': '#d35400',
        'Marker': 'D'
    },
    'Exp 1 (β=10, λ=50)': {
        'Gain Énergie (%)': 24.62,
        'Satisfaction QoS (%)': 79.06,
        'Color': '#3498db',
        'Marker': 's'
    },
    'Exp 7 (β=2, λ=100)': {
        'Gain Énergie (%)': 20.07,
        'Satisfaction QoS (%)': 88.64,
        'Color': '#e74c3c',
        'Marker': 'v'
    },
    'Exp 6 (β=35, λ=50)': {
        'Gain Énergie (%)': 18.39,
        'Satisfaction QoS (%)': 94.71,
        'Color': '#27ae60',
        'Marker': 'p'
    },
    'Exp 4 (β=10, λ=200)': {
        'Gain Énergie (%)': 14.64,
        'Satisfaction QoS (%)': 84.72,
        'Color': '#9b59b6',
        'Marker': '^'
    },
    'Exp 3 (β=35, λ=100)': {
        'Gain Énergie (%)': 8.92,
        'Satisfaction QoS (%)': 94.98,
        'Color': '#2ecc71',
        'Marker': '*'
    }
}


def main():
    """Génère le graphique de la frontière de Pareto et le sauvegarde au format PNG."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(11, 7))

    gains = [v['Gain Énergie (%)'] for v in experiments.values()]
    qos_scores = [v['Satisfaction QoS (%)'] for v in experiments.values()]
    labels = list(experiments.keys())

    # Tri par gain d'énergie pour tracer la courbe continue de compromis
    sorted_indices = np.argsort(gains)
    gains_sorted = np.array(gains)[sorted_indices]
    qos_sorted = np.array(qos_scores)[sorted_indices]

    ax.plot(gains_sorted, qos_sorted, linestyle='--', color='#7f8c8d', linewidth=2.0, label='Courbe de Compromis (Pareto)', zorder=1)

    for label, info in experiments.items():
        ax.scatter(info['Gain Énergie (%)'], info['Satisfaction QoS (%)'], color=info['Color'], s=220, zorder=5, label=label, edgecolors='black', linewidth=1.5)
        ax.annotate(
            label,
            (info['Gain Énergie (%)'], info['Satisfaction QoS (%)']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontweight='bold',
            fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8fafc', edgecolor=info['Color'], alpha=0.9)
        )

    ax.set_title('FRONTIÈRE DE PARETO À 7 RÉGIMES : ARBITRAGE ÉNERGIE VS QOS (PFE 5G/6G)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Gain Énergétique Moyen (%) sur Test Set', fontsize=11, fontweight='bold')
    ax.set_ylabel('Satisfaction QoS Moyenne (%) sur Test Set', fontsize=11, fontweight='bold')

    ax.set_xlim(5, 33)
    ax.set_ylim(75, 98)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    target_dir = os.path.join(project_root, "data", "plots")

    os.makedirs(target_dir, exist_ok=True)

    fig.tight_layout()
    fig.savefig(os.path.join(target_dir, "pareto_energy_vs_qos.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
