#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/pipeline/trainer_evaluator.py
 OBJET  : Pipeline d'Entraînement, d'Évaluation & de Benchmark Multi-Modèles (Multi-RAN & Macro-RAN)
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce module est l'orchestrateur central du projet.
Invoqué depuis `main.py`, il supervise l'ensemble du cycle de vie expérimental :
  1. Préparation des données (Support subnets bruts ou agrégation K Macro-RANs via `src/pipeline/macro_ran.py`).
  2. Split chronologique Train / Test (80% / 20%) sans fuite temporelle.
  3. Ajustement du modèle de prédiction de trafic (`fit` et `evaluate`).
  4. Calibration des capacités EcoSlice sur le Train Set et propagation figée vers l'environnement Test.
  5. Boucle d'entraînement PPO (`is_train=True`, mode stochastique) et évaluation (`is_train=False`, mode déterministe).
  6. Génération automatisée de l'ensemble des figures et graphiques comparatifs (`src/visualization/plot_generator.py`).
====================================================================================================
"""

import os
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from src.environment.sdn_controller_env import SDN_DoubleController_Env
from src.agents.ppo_agent import PPOAgent
from src.models.predictor_factory import get_traffic_predictor, AVAILABLE_MODELS
from src.visualization.plot_generator import generate_all_plots, generate_comparison_plot, generate_comparison_per_ran_plot
from src.pipeline.macro_ran import build_subnet_to_macro_map, aggregate_to_macro_rans


def log_info(msg: str) -> None:
    """
    Affiche un message de log formaté avec horodatage standard dans le terminal.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f"{now_str} [INFO] {msg}", flush=True)


def run_single_model_pipeline(
    model_name: str = "passthrough",
    subnet_choice: str = "all",
    num_rans: int = 0,
    max_steps: Optional[int] = None,
    episodes: int = 1,
    beta: float = 10.0,
    lambda_loss: float = 50.0,
    log_freq: int = 1000
) -> Dict[str, Any]:
    """
    Exécute le pipeline complet pour un modèle prédicteur unique.

    :param model_name: Nom du modèle prédicteur ('lightgbm', 'lstm', 'nhits', 'prophet', 'passthrough').
    :param subnet_choice: Sélection des subnets ('all', 'top5', 'top10' ou ID spécifique).
    :param num_rans: Nombre de Macro-RANs pour le regroupement spatial (0 = subnets bruts, 4 = Macro-RANs).
    :param max_steps: Nombre maximal de pas de temps par épisode (None = tout le dataset).
    :param episodes: Nombre d'épisodes d'entraînement de l'agent PPO.
    :param beta: Poids de la satisfaction QoS dans la récompense PPO.
    :param lambda_loss: Pénalité de perte / surcharge dans la récompense PPO.
    :param log_freq: Fréquence d'affichage des logs dans la console.
    :return: Dictionnaire synthétique des métriques obtenues sur le jeu de test.
    """
    # Répertoire racine du projet dérivé dynamiquement par rapport à l'emplacement du fichier
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = os.environ.get("DATASET_PATH", os.path.join(project_root, "subnet_slice_traffic_min2016_dense.csv"))
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(project_root, "subnet_slice_traffic_min2016_dense.csv")

    data_plots_dir = os.path.join(project_root, "data", "plots")
    artifacts_dir = os.path.join(project_root, "data", "artifacts")

    os.makedirs(data_plots_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    log_info(f"Chargement du dataset {os.path.basename(dataset_path)} (Subnet = '{subnet_choice}', Num_Rans = {num_rans})...")

    # 1. Ingestion du dataset de trafic
    df_raw = pd.read_csv(dataset_path)
    num_subnets_total = df_raw['id_institution_subnet'].nunique()

    # Si num_rans > 0, regrouper les 69 subnets en K Macro-RANs régionaux
    macro_map = None
    if num_rans > 0:
        log_info(f"Mode Macro-RAN Activé : Regroupement et somme des {num_subnets_total} subnets en {num_rans} Macro-RANs !")
        macro_map = build_subnet_to_macro_map(df_raw, num_rans=num_rans)
        df_raw = aggregate_to_macro_rans(df_raw, macro_map)
    else:
        # Filtrage standard par subnets
        sub_choice_clean = str(subnet_choice).strip().lower()
        if sub_choice_clean == "all":
            log_info("Mode Multi-RAN Global : Ingestion de TOUTES les 69 stations RAN (subnets) !")
        elif sub_choice_clean.startswith("top"):
            try:
                n_top = int(sub_choice_clean.replace("top", ""))
                top_subnets = df_raw.groupby('id_institution_subnet')['ds'].count().nlargest(n_top).index.tolist()
                log_info(f"Mode Top {n_top} RAN Stations : Ingestion des subnets {top_subnets}")
                df_raw = df_raw[df_raw['id_institution_subnet'].isin(top_subnets)].copy()
            except ValueError:
                log_info("Format 'topN' non valide, fallback sur Top 5")
                top_subnets = df_raw.groupby('id_institution_subnet')['ds'].count().nlargest(5).index.tolist()
                df_raw = df_raw[df_raw['id_institution_subnet'].isin(top_subnets)].copy()
        elif "," in sub_choice_clean:
            sub_ids = [int(s.strip()) for s in sub_choice_clean.split(",") if s.strip().isdigit()]
            log_info(f"Mode Liste de Subnets Spécifiques : Ingestion des subnets {sub_ids}")
            df_raw = df_raw[df_raw['id_institution_subnet'].isin(sub_ids)].copy()
        else:
            try:
                sub_id = int(sub_choice_clean)
                log_info(f"Mode Station RAN Unique : Ingestion du Subnet {sub_id}")
                df_raw = df_raw[df_raw['id_institution_subnet'] == sub_id].copy()
            except ValueError:
                log_info("Choix subnet non reconnu, fallback sur Subnet 0")
                df_raw = df_raw[df_raw['id_institution_subnet'] == 0].copy()

    df_raw['ds'] = pd.to_datetime(df_raw['ds'])

    pivoted_full = df_raw.pivot_table(
        index=['ds', 'id_institution_subnet'],
        columns='slice',
        values='y',
        aggfunc='sum',
        fill_value=0.0
    ).reset_index().sort_values(by=['ds', 'id_institution_subnet']).reset_index(drop=True)

    total_rows = len(pivoted_full)
    split_idx = int(0.80 * total_rows)

    # Split chronologique 80/20 pour préserver l'ordre temporel strict sans fuite
    df_train_pivoted = pivoted_full.iloc[:split_idx].copy()
    df_test_pivoted = pivoted_full.iloc[split_idx:].copy()

    if max_steps and max_steps > 0:
        df_train_pivoted = df_train_pivoted.iloc[:max_steps].copy()
        df_test_pivoted = df_test_pivoted.iloc[:min(max_steps, len(df_test_pivoted))].copy()

    train_timestamps = set(df_train_pivoted['ds'])
    test_timestamps = set(df_test_pivoted['ds'])
    df_train_raw = df_raw[df_raw['ds'].isin(train_timestamps)].copy()
    df_test_raw = df_raw[df_raw['ds'].isin(test_timestamps)].copy()

    num_subnets_loaded = pivoted_full['id_institution_subnet'].nunique()
    log_info(f"Dataset Multi-RAN ({num_subnets_loaded} RANs) : Train = {len(df_train_pivoted)} pas, Test = {len(df_test_pivoted)} pas")

    # 2. Entraînement et Évaluation du Prédicteur
    log_info(f"=== ENTRAÎNEMENT DU PRÉDICTEUR [{model_name.upper()}] ===")
    predictor = get_traffic_predictor(model_name=model_name)
    predictor.fit(df_train_pivoted)

    eval_metrics = predictor.evaluate(df_test_pivoted, df_context=df_train_pivoted)
    log_info(
        f"Précision Prédicteur [{model_name.upper()}] sur Test Set : "
        f"NMAE = {eval_metrics['NMAE']:.2f}% | MAE = {eval_metrics['MAE']:.0f} | RMSE = {eval_metrics['RMSE']:.0f}"
    )

    global_slice_names = sorted([c for c in pivoted_full.columns if c not in ['ds', 'id_institution_subnet']])

    # 3. Instanciation des Environnements SDN (Train et Test)
    # Les capacités EcoSlice (capacity_eco1/eco2) sont autocalibrées sur env_train puis transmises figées à env_test
    env_train = SDN_DoubleController_Env(
        df_train_raw, predictor=predictor, slice_names=global_slice_names,
        beta=beta, lambda_loss=lambda_loss, seed=42
    )
    env_test = SDN_DoubleController_Env(
        df_test_raw, predictor=predictor, slice_names=global_slice_names,
        df_context=df_train_pivoted, beta=beta, lambda_loss=lambda_loss, seed=42,
        capacity_eco1=env_train.capacity_eco1, capacity_eco2=env_train.capacity_eco2,
        ran_ids=env_train.unique_subnet_ids
    )

    # 4. Instanciation de l'Agent PPO
    state_sample = env_train.reset()
    agent = PPOAgent(state_dim=len(state_sample), action_dim=len(global_slice_names), lr=5e-4)

    # 5. Entraînement PPO (Stochastique) & Évaluation Test (Déterministe p_i >= 0.5)
    log_info(f"=== ENTRAÎNEMENT PPO [{model_name.lower()}] (beta={beta}, lambda={lambda_loss}) : {episodes} épisode(s) × {env_train.max_steps} pas ===")

    train_history = []
    for ep in range(episodes):
        ep_history = run_loop(
            env_train, agent, model_name=model_name, is_train=True,
            ep_idx=ep, episodes=episodes, log_freq=log_freq
        )
        if ep == episodes - 1:
            train_history = ep_history

    log_info(f"=== ÉVALUATION DU MODÈLE [{model_name.upper()}] SUR LE TEST SET ({env_test.max_steps} pas) ===")
    test_history = run_loop(
        env_test, agent, model_name=model_name, is_train=False,
        ep_idx=0, episodes=1, log_freq=log_freq
    )

    df_train_res = pd.DataFrame(train_history)
    df_test_res = pd.DataFrame(test_history)

    energy_base_test = df_test_res['f_b_base'].mean()
    energy_opt_test = df_test_res['f_b_t'].mean()
    energy_gain_test = df_test_res['delta_E_t'].mean() * 100.0
    qos_test = df_test_res['eta_b_t'].mean()

    # 6. Génération des figures et artefacts
    generate_all_plots(
        df_train_res, df_test_res, global_slice_names, data_plots_dir, artifacts_dir,
        model_name=model_name, beta=beta, lambda_loss=lambda_loss,
        num_rans=num_rans if num_rans > 0 else 1, num_subnets=num_subnets_total,
        macro_map=macro_map
    )

    per_ran_metrics = {}
    if 'subnet_id' in df_test_res.columns:
        for subnet_id, group in df_test_res.groupby('subnet_id'):
            per_ran_metrics[subnet_id] = {
                'energy_gain': group['delta_E_t'].mean() * 100.0,
                'qos': group['eta_b_t'].mean()
            }

    results = {
        'model_name': model_name,
        'subnet_choice': subnet_choice,
        'num_rans': num_rans,
        'num_subnets': num_subnets_loaded,
        'num_subnets_total': num_subnets_total,
        'energy_base_test': energy_base_test,
        'energy_opt_test': energy_opt_test,
        'energy_gain_test': energy_gain_test,
        'qos_test': qos_test,
        'mae_test': eval_metrics['MAE'],
        'nmae_test': eval_metrics['NMAE'],
        'rmse_test': eval_metrics['RMSE'],
        'per_ran_metrics': per_ran_metrics
    }

    return results


def run_all_models_pipeline(
    subnet_choice: str = "all",
    num_rans: int = 0,
    max_steps: Optional[int] = None,
    episodes: int = 1,
    beta: float = 10.0,
    lambda_loss: float = 50.0,
    log_freq: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """
    Exécute le pipeline pour TOUS les modèles prédicteurs et génère les graphiques comparatifs globaux.

    :return: Dictionnaire regroupant les résultats de tous les modèles.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_plots_dir = os.path.join(project_root, "data", "plots")
    artifacts_dir = os.path.join(project_root, "data", "artifacts")

    results_summary = {}
    log_info(f"=== BENCHMARK GLOBAL MULTI-RAN SUR TOUS LES MODÈLES (subnets={subnet_choice}, num_rans={num_rans}, beta={beta}, lambda={lambda_loss}) ===")

    for m in AVAILABLE_MODELS:
        res = run_single_model_pipeline(
            model_name=m, subnet_choice=subnet_choice, num_rans=num_rans, max_steps=max_steps, episodes=episodes,
            beta=beta, lambda_loss=lambda_loss, log_freq=log_freq
        )
        results_summary[m] = res

    # Synthèse texte sous forme de tableau terminal
    print("\n" + "=" * 105)
    print(f" TABLEAU COMPARATIF FINAL BENCHMARK MODÈLES MULTI-RAN (SUBNET = '{subnet_choice}', NUM_RANS = {num_rans}) ")
    print("=" * 105)
    print(f"{'Modèle Prédicteur':<18} | {'NMAE Erreur':<12} | {'Consommation W':<16} | {'Gain Énergie %':<16} | {'Satisfaction QoS %':<18}")
    print("-" * 105)

    for m, r in results_summary.items():
        print(f"{m.upper():<18} | {r['nmae_test']:<11.2f}% | {r['energy_opt_test']:<16.2f} W | {r['energy_gain_test']:<16.2f} % | {r['qos_test']*100:<18.2f} %")
    print("=" * 105)

    # Génération des deux figures comparatives globales (générale et par Macro-RAN)
    first_res = list(results_summary.values())[0] if len(results_summary) > 0 else {}
    generate_comparison_plot(
        results_summary, data_plots_dir, artifacts_dir,
        beta=beta, lambda_loss=lambda_loss,
        num_rans=num_rans if num_rans > 0 else 1,
        num_subnets=first_res.get('num_subnets_total', 69)
    )

    generate_comparison_per_ran_plot(
        results_summary, data_plots_dir, artifacts_dir,
        beta=beta, lambda_loss=lambda_loss,
        num_rans=num_rans if num_rans > 0 else 1,
        num_subnets=first_res.get('num_subnets_total', 69)
    )

    return results_summary


def run_loop(
    env: SDN_DoubleController_Env,
    agent: PPOAgent,
    model_name: str = "passthrough",
    is_train: bool = True,
    ep_idx: int = 0,
    episodes: int = 1,
    log_freq: int = 1000,
    update_interval: int = 64
) -> List[Dict[str, Any]]:
    """
    Exécute la boucle pas-à-pas sur un épisode complet et affiche le suivi terminal.

    :param env: Environnement SDN.
    :param agent: Agent PPO.
    :param model_name: Nom du modèle courant.
    :param is_train: True pour la phase d'entraînement (stochastique), False pour l'évaluation (déterministe).
    :param ep_idx: Index de l'épisode courant.
    :param episodes: Nombre total d'épisodes.
    :param log_freq: Fréquence des logs console.
    :param update_interval: Nombre de pas entre chaque mise à jour PPO (batching GAE).
    :return: Historique complet sous forme de liste de dictionnaires.
    """
    state = env.reset()
    history = []
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

    total_wh = 0.0

    for step_i in range(env.max_steps):
        # Mode déterministe lors du test (is_train=False) pour éliminer le bruit d'échantillonnage aléatoire
        action_binary, log_prob, val = agent.select_action(state, deterministic=not is_train)
        raw_action_dict = {s: int(action_binary[idx]) for idx, s in enumerate(sorted(env.slice_names))}

        next_state, reward, done, info = env.step_controller(raw_action_dict)

        if is_train:
            states.append(state)
            actions.append(action_binary)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(val)
            dones.append(done)

        # Calcul de la consommation cumulée en Watt-heures (1 pas = 10 min = 1/6h)
        step_wh = info['f_b_t'] * (10.0 / 60.0)
        total_wh += step_wh

        step_log = {
            'step': step_i,
            'timestamp': info['timestamp'],
            'subnet_id': info.get('subnet_id', 0),
            'f_b_base': info['f_b_base'],
            'f_b_t': info['f_b_t'],
            'delta_E_t': info['delta_E_t'],
            'eta_b_t': info['eta_b_t'],
            'qos_violation': info['qos_violation'],
            'L_t': info['L_t'],
            'reward': reward,
            'c_eco2': info['c_eco2'],
            'total_real_traffic': info['total_real_traffic'],
            'total_pred_traffic': info['total_pred_traffic']
        }

        for s in env.slice_names:
            step_log[f'c_final_{s}'] = info['c_final'].get(s, 1)
            step_log[f'rho_{s}'] = info['rho'].get(s, 0.0)
            step_log[f'qos_{s}'] = info['qos_slice'].get(s, 1.0)
            step_log[f'real_{s}'] = info['l_real'].get(s, 0.0)
            step_log[f'pred_{s}'] = info['l_pred'].get(s, 0.0)

        history.append(step_log)
        state = next_state

        # Mise à jour périodique des poids de l'agent PPO pendant l'entraînement
        if is_train and (step_i + 1) % update_interval == 0:
            agent.update(states, actions, log_probs, rewards, values, dones)
            states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        # Affichage du suivi dans le terminal à chaque log_freq
        if (step_i == 0) or ((step_i + 1) % log_freq == 0) or done:
            recent_logs = history[-log_freq:] if len(history) >= log_freq else history
            avg_reward = float(np.mean([l['reward'] for l in recent_logs]))
            avg_power = float(np.mean([l['f_b_t'] for l in recent_logs]))
            avg_eco = float(np.mean([l['delta_E_t'] for l in recent_logs])) * 100.0
            avg_qos = float(np.mean([l['eta_b_t'] for l in recent_logs])) * 100.0
            total_kwh = total_wh / 1000.0

            mode_label = f"{model_name.lower()}"
            if episodes > 1:
                log_info(
                    f"[{mode_label}] Ep {ep_idx:4d} | Step {step_i + 1:5d}/{env.max_steps} | "
                    f"Reward: {avg_reward:6.4f} | Énergie moy: {avg_power:6.1f}W | "
                    f"Cumul: {total_kwh:7.1f} kWh | QoS: {avg_qos:5.1f}% | Éco: {avg_eco:5.1f}%"
                )
            else:
                log_info(
                    f"[{mode_label}] Step {step_i + 1:5d}/{env.max_steps} | "
                    f"Reward: {avg_reward:6.4f} | Énergie moy: {avg_power:6.1f}W | "
                    f"Cumul: {total_kwh:7.1f} kWh | QoS: {avg_qos:5.1f}% | Éco: {avg_eco:5.1f}%"
                )

        if done:
            break

    return history
