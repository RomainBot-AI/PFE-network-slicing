#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/visualization/plot_full_dataset.py
 OBJET  : Visualisation de la Chronologie d'Activation sur l'Intégralité du Dataset (10 Mois)
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce script autonome génère la heatmap d'activation $c_{\text{final}}^t$ pour l'ensemble des 40 308 pas
de temps (10 mois de données réelles CESNET) sur la station 0.
Il permet d'observer la dynamique d'extinction/rallumage saisonnière des tranches sur une longue période.
====================================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.environment.sdn_controller_env import SDN_DoubleController_Env


def plot_full_dataset_activations() -> None:
    """
    Simule la chaîne décisionnelle sur l'intégralité du dataset (40 308 pas)
    et produit la heatmap d'activation des tranches.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = os.environ.get("DATASET_PATH", os.path.join(project_root, "subnet_slice_traffic_min2016_dense.csv"))
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(project_root, "subnet_slice_traffic_min2016_dense.csv")
    output_dir = os.path.join(project_root, "data", "plots")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(" VISUALISATION DE L'ACTIVATION DES SLICES SUR TOUT LE DATASET (10 MOIS) ")
    print("=" * 80)

    print("\nChargement des données complètes pour la station Subnet 0...")
    df_raw = pd.read_csv(dataset_path)
    df_raw = df_raw[df_raw['id_institution_subnet'] == 0].copy()
    df_raw['ds'] = pd.to_datetime(df_raw['ds'])

    env = SDN_DoubleController_Env(df_raw, alpha_seuil=0.01, seed=42)
    slice_names = sorted(env.slice_names)
    display_slices = slice_names + ['Eco1', 'Eco2']

    print(f"Parcours séquentiel de l'intégralité du dataset ({env.max_steps} pas de temps de 10 min)...")

    history = []
    env.reset()

    for step_i in range(env.max_steps):
        row = env.pivoted_pred.iloc[step_i]
        ts = row['ds']
        l_real = {s: float(row[s]) for s in slice_names}

        c_init_dict = {s: 1 if l_real[s] > 0 else 0 for s in slice_names}
        _, _, _, info = env.step_controller(c_init_dict)

        step_dict = {'timestamp': ts}
        for s in display_slices:
            if s == 'Eco1':
                step_dict[s] = 1
            elif s == 'Eco2':
                step_dict[s] = info['c_eco2']
            else:
                step_dict[s] = info['c_final'].get(s, 0)

        history.append(step_dict)

    df_act = pd.DataFrame(history)
    time_dates = pd.to_datetime(df_act['timestamp'])

    print("\nGénération du graphique global d'activation...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    fig, ax = plt.subplots(figsize=(18, 6))
    act_matrix = np.array([df_act[s].values for s in display_slices])

    cax = ax.imshow(
        act_matrix,
        aspect='auto',
        cmap='YlGn',
        interpolation='nearest',
        vmin=0,
        vmax=1
    )

    ax.set_yticks(np.arange(len(display_slices)))
    ax.set_yticklabels(display_slices, fontsize=12, fontweight='bold')

    tick_indices = np.linspace(0, len(time_dates) - 1, 10, dtype=int)
    tick_labels = [time_dates.iloc[idx].strftime('%d/%m/%Y\n%H:%M') for idx in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=9.5)

    ax.set_xlabel('Horodatage Réel sur Tout le Dataset (10 Mois)', fontsize=12, fontweight='bold')
    ax.set_title('Activation Temporelle des Slices sur l\'Intégralité du Dataset ($c_{\\text{final}}^t$ : 1 = Vert, 0 = Jaune)', fontsize=14, fontweight='bold')

    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Désactivé (0)', 'Activé (1)'])

    fig.tight_layout()

    out_path = os.path.join(output_dir, "7_full_dataset_slice_activations.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"\n✓ Graphique d'activation globale sauvegardé dans : {out_path}")


if __name__ == "__main__":
    plot_full_dataset_activations()
