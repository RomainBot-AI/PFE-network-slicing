#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : main.py
 OBJET  : Point d'entrée principal (CLI) pour l'exécution des simulations 5G/6G & PPO
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce script orchestre l'ensemble du projet via l'interface en ligne de commande (CLI).
Il s'insère à la racine du projet comme point de départ unique pour :
  1. Vérifier la présence et la structure du dataset de trafic (`subnet_slice_traffic_min2016_dense.csv`).
  2. Parser les arguments utilisateurs (choix du modèle prédicteur, découpage spatial Macro-RAN,
     paramètres de récompense PPO `beta` et `lambda_loss`, nombre d'épisodes, etc.).
  3. Déclencher le pipeline d'entraînement et d'évaluation via `src/pipeline/trainer_evaluator.py`.

EXEMPLES D'UTILISATION CLI :
----------------------------
  - Exécution avec LightGBM en régime d'équilibre (beta=5.0) :
      uv run main.py --model lightgbm --num_rans 4 --beta 5.0 --lambda_loss 10.0 --episodes 15

  - Benchmark comparatif de l'ensemble des modèles disponibles :
      uv run main.py --model all --num_rans 4 --beta 5.0 --lambda_loss 10.0 --episodes 15
====================================================================================================
"""

import os
import sys
import argparse
import pandas as pd

from src.pipeline.trainer_evaluator import run_single_model_pipeline, run_all_models_pipeline, log_info
from src.models.predictor_factory import AVAILABLE_MODELS


def action_prealable_obligatoire(dataset_path: str) -> None:
    """
    Effectue une vérification sécurisée du fichier dataset sans charger l'intégralité
    des 7+ millions de lignes en mémoire.

    Pourquoi cette fonction ?
    Les fichiers de traces de trafic volumineux peuvent provoquer des dépassements de mémoire (OOM)
    en cas d'erreur de format. Inspecter les entêtes et un échantillon léger permet de valider
    la structure du CSV avant de démarrer les simulations.
    """
    print("=" * 90)
    print(" ACTION PRÉALABLE OBLIGATOIRE : INSPECTION SÉCURISÉE DU DATASET ")
    print("=" * 90)

    # 1. Vérification de l'entête et du format des colonnes
    df_head = pd.read_csv(dataset_path, nrows=1)
    print("\n→ 1. Noms des colonnes et structure de la toute première ligne :")
    print(df_head.to_string())

    # 2. Échantillonnage régulier (skiprows) pour valider des lignes distribuées dans le temps
    df_sample = pd.read_csv(dataset_path, skiprows=lambda i: i > 0 and i % 500000 != 0, nrows=10)
    print("\n→ 2. Échantillon de quelques lignes prélevées dans le dataset :")
    print(df_sample.to_string())
    print("=" * 90 + "\n")


def main() -> None:
    """
    Fonction principale parsant la CLI et lançant le pipeline approprié.
    """
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
        help="Regroupement spatial des 69 subnets en K Macro-RANs (ex: --num_rans 4). Accélère l'entraînement du trafic réseau."
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

    # Détermination dynamique du répertoire racine du projet
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.environ.get("DATASET_PATH", os.path.join(project_root, "subnet_slice_traffic_min2016_dense.csv"))
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(project_root, "subnet_slice_traffic_min2016_dense.csv")

    # Exécution du contrôle préalable de structure
    action_prealable_obligatoire(dataset_path)

    max_steps_val = args.steps if args.steps > 0 else None
    model_choice = args.model.lower()

    # Dispatching selon l'argument utilisateur : modèle unique ou suite complète de benchmarks
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
