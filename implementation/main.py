#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point for the offline network-slicing PPO simulation.

Loads the subnet/slice panel, trains the selected traffic predictor and the PPO
agent, evaluates on the test set, and writes figures. See the module tree in
``src/`` and the project README for details.

Examples:
    python3 main.py --model lightgbm --num_rans 4 --episodes 10
    python3 main.py --model all --num_rans 4
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.models.predictor_factory import AVAILABLE_MODELS
from src.pipeline.config import DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_DIR, RunConfig
from src.pipeline.trainer_evaluator import run_all_models_pipeline, run_single_model_pipeline


def inspect_dataset(dataset_path: str) -> None:
    """Print the header and a sparse sample of the dataset without full load."""
    print("=" * 90)
    print(" INSPECTION SÉCURISÉE DU DATASET ")
    print("=" * 90)

    df_head = pd.read_csv(dataset_path, nrows=1)
    print("\n→ 1. Noms des colonnes et structure de la toute première ligne :")
    print(df_head.to_string())

    df_sample = pd.read_csv(dataset_path, skiprows=lambda i: i > 0 and i % 500000 != 0, nrows=10)
    print("\n→ 2. Échantillon de quelques lignes prélevées dans le dataset :")
    print(df_sample.to_string())
    print("=" * 90 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation RAN Network Slicing 5G/6G avec Agent PPO & Contrôleurs SDN Multi-RAN / Macro-RAN."
    )
    parser.add_argument(
        "--model", type=str, default="passthrough",
        help=f"Choix du modèle prédicteur : {AVAILABLE_MODELS} ou 'all' pour comparer tous les modèles.",
    )
    parser.add_argument(
        "--num_rans", type=int, default=0,
        help="Regroupement spatial des subnets en K Macro-RANs (ex: --num_rans 4). Accélère l'entraînement tout en conservant 100% du trafic.",
    )
    parser.add_argument(
        "--subnet", type=str, default="all",
        help="Choix des stations RAN / subnets : 'all', 'top5', 'top10', ou un ID (ex: '0', '106'). Ignoré si --num_rans > 0.",
    )
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Nombre maximal de pas de temps par épisode (0 = tout le dataset).",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Nombre d'épisodes d'entraînement PPO (par défaut 1).",
    )
    parser.add_argument(
        "--beta", type=float, default=10.0,
        help="Poids de la satisfaction QoS dans la récompense PPO (par défaut 10.0).",
    )
    parser.add_argument(
        "--lambda_loss", type=float, default=50.0,
        help="Pénalité de violation QoS / surcharge (par défaut 50.0).",
    )
    parser.add_argument(
        "--log_freq", type=int, default=1000,
        help="Fréquence d'affichage des logs (par défaut tous les 1000 pas).",
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET_PATH,
        help=f"Chemin du dataset subnet/slice (par défaut : {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Répertoire de sortie des figures (par défaut : {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        model_name=args.model.lower(),
        subnet_choice=args.subnet,
        num_rans=args.num_rans,
        max_steps=args.steps if args.steps > 0 else None,
        episodes=args.episodes,
        beta=args.beta,
        lambda_loss=args.lambda_loss,
        log_freq=args.log_freq,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )

    inspect_dataset(config.dataset_path)

    if config.model_name == "all":
        run_all_models_pipeline(config)
    else:
        run_single_model_pipeline(config)


if __name__ == "__main__":
    main()
