#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 PROJET PFE : SIMULATION DE NETWORK SLICING 5G/6G & AGENT PPO (MULTI-RAN / MACRO-RAN)
 FICHIER PRINCIPAL : main.py
====================================================================================================

DESCRIPTION DÉTAILLÉE :
-----------------------
Point d'entrée principal du projet. Ce script :
  1. Effectue l'Action Préalable Obligatoire : Inspection sécurisée de quelques lignes aléatoires
     du dataset 'subnet_slice_traffic_min2016_dense.csv' (~7 millions de lignes) sans tout charger.
  2. Accepte des arguments en ligne de commande pour configurer la simulation Multi-RAN / Macro-RAN :
       - python3 main.py --model passthrough   (Choix du modèle : passthrough, ridge, lightgbm, lstm, nhits, all)
       - python3 main.py --num_rans 4          (Regroupement spatial des 69 subnets en 4 Macro-RANs - Accélération 17x)
       - python3 main.py --subnet all          (Sélection des subnets : 'all' = les 69 stations, 'top5', 'top10', ou '0')
       - python3 main.py --steps 5000           (Nombre de pas de temps max par épisode, 0 = tout le dataset)
       - python3 main.py --episodes 10          (Nombre d'épisodes d'entraînement PPO)
       - python3 main.py --beta 10.0            (Poids d'importance de la satisfaction QoS dans la récompense)
       - python3 main.py --lambda_loss 50.0     (Pénalité de violation QoS / surcharge)
       - python3 main.py --log_freq 1000        (Fréquence d'affichage du suivi dans le terminal)

ARBORESCENCE DU PROJET :
------------------------
  src/
  ├── simulator/
  │   └── ran_simulator.py       # Moteur physique et énergétique 5G/6G (Phyu et al., 2023)
  ├── environment/
  │   └── sdn_controller_env.py  # Environnement SDN à Double Contrôleur Multi-RAN (Algorithmes 1 à 4)
  ├── agents/
  │   └── ppo_agent.py           # Agent Apprentissage par Renforcement PPO (PyTorch Multi-Binaire)
  ├── models/
  │   ├── base_predictor.py      # Interface abstraite (NMAE %, MAE, RMSE)
  │   ├── passthrough_predictor.py # Oracle / Naive passthrough
  │   ├── ridge_predictor.py     # Régression Ridge
  │   ├── lightgbm_predictor.py  # LightGBM Gradient Boosting (features enrichies)
  │   ├── lstm_predictor.py      # PyTorch LSTM
  │   ├── nhits_predictor.py     # PyTorch N-HiTS
  │   └── predictor_factory.py   # Factory usine à modèles
  ├── pipeline/
  │   ├── macro_ran.py           # Regroupement et agrégation spatiale en K Macro-RANs (K-means / quantiles)
  │   └── trainer_evaluator.py   # Orchestrateur Train/Test 80/20 & Benchmark Multi-RAN
  └── visualization/
      ├── plot_generator.py      # Générateur des 6 figures et graphique comparatif
      └── plot_full_dataset.py   # Visualiseur autonome sur tout le dataset (10 mois)
====================================================================================================
"""

import sys
import argparse
import pandas as pd

from src.pipeline.trainer_evaluator import run_single_model_pipeline, run_all_models_pipeline, log_info
from src.models.predictor_factory import AVAILABLE_MODELS


def action_prealable_obligatoire(dataset_path: str):
    """
    Action Préalable Obligatoire : Inspecte la première ligne et quelques lignes aléatoires
    du dataset d'entrée sans charger les 7 millions de lignes en mémoire.
    """
    print("=" * 90)
    print(" ACTION PRÉALABLE OBLIGATOIRE : INSPECTION SÉCURISÉE DU DATASET ")
    print("=" * 90)

    # 1. Lecture de la toute première ligne pour vérifier les colonnes et le format
    df_head = pd.read_csv(dataset_path, nrows=1)
    print("\n→ 1. Noms des colonnes et structure de la toute première ligne :")
    print(df_head.to_string())

    # 2. Inspection sécurisée d'un petit échantillon aléatoire (10 lignes)
    df_sample = pd.read_csv(dataset_path, skiprows=lambda i: i > 0 and i % 500000 != 0, nrows=10)
    print("\n→ 2. Échantillon de quelques lignes prélevées dans le dataset :")
    print(df_sample.to_string())
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Simulation RAN Network Slicing 5G/6G avec Agent PPO & Contrôleurs SDN Multi-RAN / Macro-RAN."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="passthrough",
        help=f"Choix du modèle prédicteur : {AVAILABLE_MODELS} ou 'all' pour comparer tous les modèles."
    )
    parser.add_argument(
        "--num_rans",
        type=int,
        default=0,
        help="Regroupement spatial des 69 subnets en K Macro-RANs (ex: --num_rans 4). Accélère l'entraînement 17x tout en conservant 100% du trafic réseau."
    )
    parser.add_argument(
        "--subnet",
        type=str,
        default="all",
        help="Choix des stations RAN / subnets : 'all' (les 69 subnets), 'top5', 'top10', ou un ID spécifique (ex: '0', '106'). Par défaut 'all'."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Nombre maximal de pas de temps par épisode (0 = tout le dataset)."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Nombre d'épisodes d'entraînement PPO (par défaut 1)."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=10.0,
        help="Poids d'importance de la satisfaction QoS dans la récompense PPO (par défaut 10.0)."
    )
    parser.add_argument(
        "--lambda_loss",
        type=float,
        default=50.0,
        help="Pénalité de perte attribuée aux violations QoS / surcharge (par défaut 50.0)."
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=1000,
        help="Fréquence d'affichage des logs dans le terminal (par défaut tous les 1000 pas)."
    )

    args = parser.parse_args()
    dataset_path = "/home/cytech/Ing3/PFE/dataset_creation/subnet_slice_traffic_min2016_dense.csv"

    # Action Préalable
    action_prealable_obligatoire(dataset_path)

    max_steps_val = args.steps if args.steps > 0 else None

    # Lancement selon l'argument du modèle
    model_choice = args.model.lower()
    if model_choice == "all":
        run_all_models_pipeline(
            subnet_choice=args.subnet, num_rans=args.num_rans, max_steps=max_steps_val, episodes=args.episodes,
            beta=args.beta, lambda_loss=args.lambda_loss, log_freq=args.log_freq
        )
    else:
        run_single_model_pipeline(
            model_name=model_choice, subnet_choice=args.subnet, num_rans=args.num_rans, max_steps=max_steps_val, episodes=args.episodes,
            beta=args.beta, lambda_loss=args.lambda_loss, log_freq=args.log_freq
        )


if __name__ == "__main__":
    main()
