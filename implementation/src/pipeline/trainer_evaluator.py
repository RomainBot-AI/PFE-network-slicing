"""Training, evaluation, and benchmark pipeline for the offline slicing sim.

Orchestrates: load and pivot the dataset (single subnet, topN, all subnets, or K
macro-RANs) -> chronological 80/20 train/test split -> fit the traffic predictor
-> train the PPO agent -> evaluate on the test set -> generate plots. With
``model_name="all"`` every predictor is run in turn and a global benchmark table
and comparison chart are produced.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.agents.ppo_agent import PPOAgent
from src.environment.sdn_controller_env import SDN_DoubleController_Env
from src.models.predictor_factory import AVAILABLE_MODELS, get_traffic_predictor
from src.pipeline.config import DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_DIR, RunConfig
from src.pipeline.macro_ran import aggregate_to_macro_rans, build_subnet_to_macro_map
from src.visualization.plot_generator import generate_all_plots, generate_comparison_plot

__all__ = [
    "RunConfig",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_OUTPUT_DIR",
    "run_single_model_pipeline",
    "run_all_models_pipeline",
    "log_info",
]


def log_info(msg: str) -> None:
    """Print a timestamped log line to the terminal."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f"{now_str} [INFO] {msg}", flush=True)


def _output_dirs(config: RunConfig) -> Tuple[str, str]:
    data_plots_dir = config.output_dir
    artifacts_dir = os.path.join(config.output_dir, "_artifacts")
    os.makedirs(data_plots_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    return data_plots_dir, artifacts_dir


def _load_dataset(config: RunConfig) -> Tuple[pd.DataFrame, int]:
    """Load the raw panel and apply macro-RAN grouping or subnet selection."""
    log_info(
        f"Chargement du dataset {os.path.basename(config.dataset_path)} "
        f"(Subnet = '{config.subnet_choice}', Num_Rans = {config.num_rans})..."
    )
    df_raw = pd.read_csv(config.dataset_path)
    num_subnets_total = df_raw["id_institution_subnet"].nunique()

    if config.num_rans > 0:
        log_info(
            f"Mode Macro-RAN Activé : Regroupement et somme des {num_subnets_total} subnets "
            f"en {config.num_rans} Macro-RANs !"
        )
        macro_map = build_subnet_to_macro_map(df_raw, num_rans=config.num_rans)
        df_raw = aggregate_to_macro_rans(df_raw, macro_map)
        df_raw["ds"] = pd.to_datetime(df_raw["ds"])
        return df_raw, num_subnets_total

    choice = str(config.subnet_choice).strip().lower()
    if choice == "all":
        log_info("Mode Multi-RAN Global : Ingestion de TOUTES les 69 stations RAN (subnets) !")
    elif choice.startswith("top"):
        try:
            n_top = int(choice.replace("top", ""))
        except ValueError:
            log_info("Format 'topN' non valide, fallback sur Top 5")
            n_top = 5
        top_subnets = df_raw.groupby("id_institution_subnet")["ds"].count().nlargest(n_top).index.tolist()
        log_info(f"Mode Top {n_top} RAN Stations : Ingestion des subnets {top_subnets}")
        df_raw = df_raw[df_raw["id_institution_subnet"].isin(top_subnets)].copy()
    elif "," in choice:
        sub_ids = [int(s.strip()) for s in choice.split(",") if s.strip().isdigit()]
        log_info(f"Mode Liste de Subnets Spécifiques : Ingestion des subnets {sub_ids}")
        df_raw = df_raw[df_raw["id_institution_subnet"].isin(sub_ids)].copy()
    else:
        try:
            sub_id = int(choice)
            log_info(f"Mode Station RAN Unique : Ingestion du Subnet {sub_id}")
        except ValueError:
            log_info("Choix subnet non reconnu, fallback sur Subnet 0")
            sub_id = 0
        df_raw = df_raw[df_raw["id_institution_subnet"] == sub_id].copy()

    df_raw["ds"] = pd.to_datetime(df_raw["ds"])
    return df_raw, num_subnets_total


def _split_panel(df_raw: pd.DataFrame, config: RunConfig):
    """Pivot to a dense panel and split chronologically 80/20 (train/test)."""
    pivoted_full = (
        df_raw.pivot_table(
            index=["ds", "id_institution_subnet"],
            columns="slice",
            values="y",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .sort_values(by=["ds", "id_institution_subnet"])
        .reset_index(drop=True)
    )

    split_idx = int(0.80 * len(pivoted_full))
    df_train_pivoted = pivoted_full.iloc[:split_idx].copy()
    df_test_pivoted = pivoted_full.iloc[split_idx:].copy()

    if config.max_steps and config.max_steps > 0:
        df_train_pivoted = df_train_pivoted.iloc[: config.max_steps].copy()
        df_test_pivoted = df_test_pivoted.iloc[: min(config.max_steps, len(df_test_pivoted))].copy()

    df_train_raw = df_raw[df_raw["ds"].isin(set(df_train_pivoted["ds"]))].copy()
    df_test_raw = df_raw[df_raw["ds"].isin(set(df_test_pivoted["ds"]))].copy()

    return pivoted_full, df_train_pivoted, df_test_pivoted, df_train_raw, df_test_raw


def run_single_model_pipeline(config: RunConfig) -> Dict[str, Any]:
    """Run the full pipeline (predictor + PPO) for a single model."""
    data_plots_dir, artifacts_dir = _output_dirs(config)

    df_raw, num_subnets_total = _load_dataset(config)
    pivoted_full, df_train_pivoted, df_test_pivoted, df_train_raw, df_test_raw = _split_panel(df_raw, config)

    num_subnets_loaded = pivoted_full["id_institution_subnet"].nunique()
    log_info(
        f"Dataset Multi-RAN ({num_subnets_loaded} RANs) : "
        f"Train = {len(df_train_pivoted)} pas, Test = {len(df_test_pivoted)} pas"
    )

    log_info(f"=== ENTRAÎNEMENT DU PRÉDICTEUR [{config.model_name.upper()}] ===")
    predictor = get_traffic_predictor(model_name=config.model_name)
    predictor.fit(df_train_pivoted)

    eval_metrics = predictor.evaluate(df_test_pivoted, df_context=df_train_pivoted)
    log_info(
        f"Précision Prédicteur [{config.model_name.upper()}] sur Test Set : "
        f"NMAE = {eval_metrics['NMAE']:.2f}% | MAE = {eval_metrics['MAE']:.0f} | RMSE = {eval_metrics['RMSE']:.0f}"
    )

    global_slice_names = sorted(c for c in pivoted_full.columns if c not in ["ds", "id_institution_subnet"])

    env_train = SDN_DoubleController_Env(
        df_train_raw, predictor=predictor, slice_names=global_slice_names,
        beta=config.beta, lambda_loss=config.lambda_loss, seed=config.seed,
    )
    env_test = SDN_DoubleController_Env(
        df_test_raw, predictor=predictor, slice_names=global_slice_names,
        df_context=df_train_pivoted, beta=config.beta, lambda_loss=config.lambda_loss, seed=config.seed,
    )

    state_sample = env_train.reset()
    agent = PPOAgent(state_dim=len(state_sample), action_dim=len(global_slice_names), lr=5e-4)

    log_info(
        f"=== ENTRAÎNEMENT PPO [{config.model_name.lower()}] "
        f"(beta={config.beta}, lambda={config.lambda_loss}) : {config.episodes} épisode(s) × {env_train.max_steps} pas ==="
    )
    train_history: List[dict] = []
    for ep in range(config.episodes):
        ep_history = run_loop(
            env_train, agent, model_name=config.model_name, is_train=True,
            ep_idx=ep, episodes=config.episodes, log_freq=config.log_freq,
        )
        if ep == config.episodes - 1:
            train_history = ep_history

    log_info(f"=== ÉVALUATION DU MODÈLE [{config.model_name.upper()}] SUR LE TEST SET ({env_test.max_steps} pas) ===")
    test_history = run_loop(
        env_test, agent, model_name=config.model_name, is_train=False,
        ep_idx=0, episodes=1, log_freq=config.log_freq,
    )

    df_train_res = pd.DataFrame(train_history)
    df_test_res = pd.DataFrame(test_history)

    generate_all_plots(
        df_train_res, df_test_res, global_slice_names, data_plots_dir, artifacts_dir,
        model_name=config.model_name, beta=config.beta, lambda_loss=config.lambda_loss,
        num_rans=config.num_rans if config.num_rans > 0 else 1, num_subnets=num_subnets_total,
    )

    return {
        "model_name": config.model_name,
        "subnet_choice": config.subnet_choice,
        "num_rans": config.num_rans,
        "num_subnets": num_subnets_loaded,
        "num_subnets_total": num_subnets_total,
        "energy_base_test": df_test_res["f_b_base"].mean(),
        "energy_opt_test": df_test_res["f_b_t"].mean(),
        "energy_gain_test": df_test_res["delta_E_t"].mean() * 100.0,
        "qos_test": df_test_res["eta_b_t"].mean(),
        "mae_test": eval_metrics["MAE"],
        "nmae_test": eval_metrics["NMAE"],
        "rmse_test": eval_metrics["RMSE"],
    }


def run_all_models_pipeline(config: RunConfig) -> Dict[str, Dict[str, Any]]:
    """Run every available predictor in turn and print a benchmark table."""
    data_plots_dir, artifacts_dir = _output_dirs(config)

    log_info(
        f"=== BENCHMARK GLOBAL MULTI-RAN SUR TOUS LES MODÈLES "
        f"(subnets={config.subnet_choice}, num_rans={config.num_rans}, beta={config.beta}, lambda={config.lambda_loss}) ==="
    )

    results_summary: Dict[str, Dict[str, Any]] = {}
    for model_name in AVAILABLE_MODELS:
        try:
            results_summary[model_name] = run_single_model_pipeline(dataclasses.replace(config, model_name=model_name))
        except ModuleNotFoundError as exc:
            log_info(f"Modèle '{model_name}' ignoré : backend optionnel manquant ({exc.name}).")

    print("\n" + "=" * 105)
    print(f" TABLEAU COMPARATIF FINAL BENCHMARK MODÈLES MULTI-RAN (SUBNET = '{config.subnet_choice}', NUM_RANS = {config.num_rans}) ")
    print("=" * 105)
    print(f"{'Modèle Prédicteur':<18} | {'NMAE Erreur':<12} | {'Consommation W':<16} | {'Gain Énergie %':<16} | {'Satisfaction QoS %':<18}")
    print("-" * 105)
    for model_name, r in results_summary.items():
        print(
            f"{model_name.upper():<18} | {r['nmae_test']:<11.2f}% | {r['energy_opt_test']:<16.2f} W | "
            f"{r['energy_gain_test']:<16.2f} % | {r['qos_test'] * 100:<18.2f} %"
        )
    print("=" * 105)

    first_res = next(iter(results_summary.values()), {})
    generate_comparison_plot(
        results_summary, data_plots_dir, artifacts_dir,
        beta=config.beta, lambda_loss=config.lambda_loss,
        num_rans=config.num_rans if config.num_rans > 0 else 1,
        num_subnets=first_res.get("num_subnets_total", 69),
    )
    return results_summary


def run_loop(
    env,
    agent,
    model_name: str = "passthrough",
    is_train: bool = True,
    ep_idx: int = 0,
    episodes: int = 1,
    log_freq: int = 1000,
    update_interval: int = 64,
) -> List[dict]:
    """Run one episode, logging normalized metrics (kWh, QoS %, energy gain %)."""
    state = env.reset()
    history: List[dict] = []
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
    total_wh = 0.0

    for step_i in range(env.max_steps):
        action_binary, log_prob, val = agent.select_action(state)
        raw_action_dict = {s: int(action_binary[idx]) for idx, s in enumerate(sorted(env.slice_names))}

        next_state, reward, done, info = env.step_controller(raw_action_dict)

        if is_train:
            states.append(state)
            actions.append(action_binary)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(val)
            dones.append(done)

        # Energy over one 10-minute step, in Wh (10 min = 1/6 h).
        total_wh += info["f_b_t"] * (10.0 / 60.0)

        step_log = {
            "step": step_i,
            "timestamp": info["timestamp"],
            "subnet_id": info.get("subnet_id", 0),
            "f_b_base": info["f_b_base"],
            "f_b_t": info["f_b_t"],
            "delta_E_t": info["delta_E_t"],
            "eta_b_t": info["eta_b_t"],
            "qos_violation": info["qos_violation"],
            "L_t": info["L_t"],
            "reward": reward,
            "c_eco2": info["c_eco2"],
            "total_real_traffic": info["total_real_traffic"],
            "total_pred_traffic": info["total_pred_traffic"],
        }
        for s in env.slice_names:
            step_log[f"c_final_{s}"] = info["c_final"].get(s, 1)
            step_log[f"rho_{s}"] = info["rho"].get(s, 0.0)
            step_log[f"qos_{s}"] = info["qos_slice"].get(s, 1.0)
            step_log[f"real_{s}"] = info["l_real"].get(s, 0.0)
            step_log[f"pred_{s}"] = info["l_pred"].get(s, 0.0)

        history.append(step_log)
        state = next_state

        if is_train and (step_i + 1) % update_interval == 0:
            agent.update(states, actions, log_probs, rewards, values, dones)
            states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        if (step_i == 0) or ((step_i + 1) % log_freq == 0) or done:
            recent_logs = history[-log_freq:] if len(history) >= log_freq else history
            avg_reward = float(np.mean([l["reward"] for l in recent_logs]))
            avg_power = float(np.mean([l["f_b_t"] for l in recent_logs]))
            avg_eco = float(np.mean([l["delta_E_t"] for l in recent_logs])) * 100.0
            avg_qos = float(np.mean([l["eta_b_t"] for l in recent_logs])) * 100.0
            total_kwh = total_wh / 1000.0

            prefix = f"[{model_name.lower()}]"
            if episodes > 1:
                prefix += f" Ep {ep_idx:4d}"
            log_info(
                f"{prefix} Step {step_i + 1:5d}/{env.max_steps} | "
                f"Reward: {avg_reward:6.4f} | Énergie moy: {avg_power:6.1f}W | "
                f"Cumul: {total_kwh:7.1f} kWh | QoS: {avg_qos:5.1f}% | Éco: {avg_eco:5.1f}%"
            )

        if done:
            break

    return history
