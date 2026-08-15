#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Energy-vs-QoS Pareto frontier chart for the seven experiment regimes.

Figure labels are kept in French to match the report/defense.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

_IMPL_ROOT = Path(__file__).resolve().parents[2]

# Consolidated results from the seven PFE experiments.
experiments = {
    'Exp 2 : Sobriété Extrême\n(β=2, λ=10)': {
        'Gain Énergie (%)': 29.49,
        'Satisfaction QoS (%)': 82.92,
        'Color': '#e67e22',
        'Marker': 'o'
    },
    'Exp 5 : Sobriété Protégée\n(β=2, λ=50)': {
        'Gain Énergie (%)': 25.24,
        'Satisfaction QoS (%)': 90.82,
        'Color': '#d35400',
        'Marker': 'D'
    },
    'Exp 1 : Équilibré Standard\n(β=10, λ=50)': {
        'Gain Énergie (%)': 24.62,
        'Satisfaction QoS (%)': 79.06,
        'Color': '#3498db',
        'Marker': 's'
    },
    'Exp 7 : Sobriété Anti-Surcharge\n(β=2, λ=100)': {
        'Gain Énergie (%)': 20.07,
        'Satisfaction QoS (%)': 88.64,
        'Color': '#e74c3c',
        'Marker': 'v'
    },
    'Exp 6 : QoS Prioritaire\n(β=35, λ=50)': {
        'Gain Énergie (%)': 18.39,
        'Satisfaction QoS (%)': 94.71,
        'Color': '#27ae60',
        'Marker': 'p'
    },
    'Exp 4 : Anti-Surcharge\n(β=10, λ=200)': {
        'Gain Énergie (%)': 14.64,
        'Satisfaction QoS (%)': 84.72,
        'Color': '#9b59b6',
        'Marker': '^'
    },
    'Exp 3 : Haute Priorité SLA\n(β=35, λ=100)': {
        'Gain Énergie (%)': 8.92,
        'Satisfaction QoS (%)': 94.98,
        'Color': '#2ecc71',
        'Marker': '*'
    }
}

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6))

gains = [v['Gain Énergie (%)'] for v in experiments.values()]
qos_scores = [v['Satisfaction QoS (%)'] for v in experiments.values()]
labels = list(experiments.keys())
colors = [v['Color'] for v in experiments.values()]
markers = [v['Marker'] for v in experiments.values()]

# Pareto frontier: keep only the non-dominated regimes (no other regime beats
# them on energy gain *and* QoS at once), then join those. Connecting every
# point in x-order instead would draw a zig-zag through dominated regimes.
points = sorted(zip(gains, qos_scores))
frontier = []
for gain, qos in reversed(points):
    if not frontier or qos > frontier[-1][1]:
        frontier.append((gain, qos))
frontier.reverse()

ax.plot(
    [p[0] for p in frontier], [p[1] for p in frontier],
    linestyle='--', color='#7f8c8d', linewidth=2.0,
    label='Courbe de Compromis (Frontière de Pareto)', zorder=1
)

for label, info in experiments.items():
    ax.scatter(info['Gain Énergie (%)'], info['Satisfaction QoS (%)'], color=info['Color'], s=220, zorder=5, label=label.replace('\n', ' '), edgecolors='black', linewidth=1.5)
    ax.annotate(
        label,
        (info['Gain Énergie (%)'], info['Satisfaction QoS (%)']),
        textcoords="offset points",
        xytext=(0, 12),
        ha='center',
        fontweight='bold',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8fafc', edgecolor=info['Color'], alpha=0.9)
    )

ax.set_title('FRONTIÈRE DE PARETO : COMPROMIS ÉNERGIE VS QOS SELON LES RÉGIMES (PFE 5G/6G)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Gain Énergétique Moyen (%) sur Test Set', fontsize=11, fontweight='bold')
ax.set_ylabel('Satisfaction QoS Moyenne (%) sur Test Set', fontsize=11, fontweight='bold')

ax.set_xlim(5, 35)
ax.set_ylim(75, 100)

target_dir = str(_IMPL_ROOT / "data" / "plots")
artifact_dir = os.path.join(target_dir, "_artifacts")

os.makedirs(target_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

fig.tight_layout()
fig.savefig(os.path.join(target_dir, "pareto_energy_vs_qos.png"), dpi=300)
fig.savefig(os.path.join(artifact_dir, "pareto_energy_vs_qos.png"), dpi=300)
plt.close(fig)

print("Graphique de Pareto généré avec succès !")
