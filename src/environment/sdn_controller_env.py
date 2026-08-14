#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
====================================================================================================
 MODULE : src/environment/sdn_controller_env.py
 OBJET  : Environnement Gymnasium du Contrôleur SDN à Double Niveau (Multi-RAN / Multi-Subnets)
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Cet environnement encapsule la logique de décision du réseau et sert d'interface entre :
  - Le prédicteur de trafic (`src/models/base_predictor.py`) qui fournit les prédictions \hat{l}^{t+1}.
  - L'agent RL PPO (`src/agents/ppo_agent.py`) qui choisit l'action d'activation binaire c_init^t.
  - Le moteur physique RAN (`src/simulator/ran_simulator.py`) qui évalue la consommation et la QoS.

CHAÎNE DÉCISIONNELLE EN 4 ALGORITHMES SDN :
-------------------------------------------
  - Algorithme 1 (Décision PPO) :
      Reçoit l'action brute c_{init}^t de l'agent PPO. Maintient EcoSlice 1 toujours active.

  - Algorithme 2 (Filtre des Seuils Prédictifs SDN 1 & Veto URLLC) :
      Compare la prédiction \hat{l}_{i,b}^{t+1} au seuil \alpha_{\text{seuil}} \cdot \text{Max}_{i,b}.
      Éteint la tranche si c_{i,\text{init}}^t = 0 ou si la prédiction est sous le seuil.
      Applique un Veto URLLC déterministe qui réactive la tranche URLLC si du trafic réel est détecté.

  - Algorithme 3 (Routage Prioritaire & EcoSlice Bis - SDN 2) :
      Trie le trafic orphelin par priorité (URLLC > MIX > mMTC > eMBB) et le déverse dans EcoSlice 1.
      Si EcoSlice 1 dépasse 75% de sa capacité, allume EcoSlice Bis (c_{\text{eco2}} = 1).

  - Algorithme 4 (Sécurité & Calcul de la Récompense PPO) :
      Rallume la tranche la plus lourde en cas de surcharge.
      Calcule le score QoS \eta_b^t, le taux de perte L_t, et la récompense multi-objectif r_t :
        r_t = \frac{1000.0}{f_b(t)} + \beta \cdot \eta_b(t) - \lambda \cdot L_t
====================================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from src.simulator.ran_simulator import RAN_Simulator
from src.models.base_predictor import BaseTrafficPredictor
from src.models.passthrough_predictor import PassthroughTrafficPredictor


class SDN_DoubleController_Env:
    """
    Environnement SDN à Double Contrôleur pour l'optimisation énergétique et QoS du RAN 5G/6G.
    """

    def __init__(
        self,
        df_traffic: pd.DataFrame,
        predictor: Optional[BaseTrafficPredictor] = None,
        slice_names: Optional[List[str]] = None,
        df_context: Optional[pd.DataFrame] = None,
        alpha_seuil: float = 0.01,
        capacity_eco1: Optional[float] = None,
        capacity_eco2: Optional[float] = None,
        seuil_75_ratio: float = 0.75,
        beta: float = 10.0,
        lambda_loss: float = 50.0,
        seed: int = 42,
        ran_ids: Optional[List[int]] = None
    ):
        """
        Initialise l'environnement, calibre les capacités EcoSlice et précalcule les valeurs de pic Max_i.

        :param df_traffic: DataFrame du trafic réseau d'entrée.
        :param predictor: Prédicteur de trafic instancié (par défaut Passthrough).
        :param slice_names: Noms des tranches dédiées à gérer.
        :param df_context: DataFrame de contexte d'historique pour les prédicteurs tabulaires.
        :param alpha_seuil: Coefficient du seuil d'extinction prédictif (ex: 0.01 = 1% du pic).
        :param capacity_eco1: Capacité d'EcoSlice 1 (autocalibrée sur le p95 si None).
        :param capacity_eco2: Capacité d'EcoSlice Bis (autocalibrée sur le p95 si None).
        :param seuil_75_ratio: Ratio de déclenchement d'EcoSlice Bis (0.75 = 75%).
        :param beta: Poids de la satisfaction QoS dans la récompense PPO.
        :param lambda_loss: Pénalité de perte / surcharge dans la récompense PPO.
        :param seed: Graine aléatoire.
        :param ran_ids: Identifiants des Macro-RANs (pour le codage One-Hot de l'état).
        """
        self.df_traffic = df_traffic.copy()
        self.alpha_seuil = alpha_seuil

        # Auto-calibration des capacités EcoSlice sur le 95e percentile du trafic total instantané
        if capacity_eco1 is None or capacity_eco2 is None:
            _tmp_piv = self.df_traffic.copy()
            if 'slice' in _tmp_piv.columns and 'y' in _tmp_piv.columns:
                _tmp_piv = _tmp_piv.pivot_table(
                    index=['ds', 'id_institution_subnet'],
                    columns='slice', values='y',
                    aggfunc='sum', fill_value=0.0
                ).reset_index()
            _traffic_cols = [c for c in _tmp_piv.columns if c not in ['ds', 'id_institution_subnet']]
            _total_per_step = _tmp_piv[_traffic_cols].sum(axis=1)
            _p95 = float(_total_per_step.quantile(0.95))
            _auto_cap = max(_p95, 1.0)
            self.capacity_eco1 = capacity_eco1 if capacity_eco1 is not None else _auto_cap
            self.capacity_eco2 = capacity_eco2 if capacity_eco2 is not None else _auto_cap
        else:
            self.capacity_eco1 = capacity_eco1
            self.capacity_eco2 = capacity_eco2

        self.seuil_75 = seuil_75_ratio * self.capacity_eco1
        self.beta = beta
        self.lambda_loss = lambda_loss
        self.seed = seed

        np.random.seed(seed)
        self.ran_sim = RAN_Simulator(include_ecoslice_in_qos=True)

        # Pivotement de la table pour disposer d'un vecteur de trafic par pas de temps et subnet
        if 'slice' in self.df_traffic.columns and 'y' in self.df_traffic.columns:
            self.pivoted = self.df_traffic.pivot_table(
                index=['ds', 'id_institution_subnet'],
                columns='slice',
                values='y',
                aggfunc='sum',
                fill_value=0.0
            ).reset_index()
        else:
            self.pivoted = self.df_traffic.copy()

        self.pivoted['ds'] = pd.to_datetime(self.pivoted['ds'])
        self.pivoted = self.pivoted.sort_values(by=['ds', 'id_institution_subnet']).reset_index(drop=True)

        self.slice_names = slice_names or [c for c in self.pivoted.columns if c not in ['ds', 'id_institution_subnet']]
        for s in self.slice_names:
            if s not in self.pivoted.columns:
                self.pivoted[s] = 0.0

        # Calcul causal glissant du pic Max_i (fenêtre passée de 7 jours, décalée d'un pas pour éviter toute fuite)
        self.max_i_window = 1008
        self.Max_i_subnet = {}
        unique_subnets = self.pivoted['id_institution_subnet'].unique()

        # Identification unique des Macro-RANs pour le vecteur One-Hot de l'état PPO
        self.unique_subnet_ids = sorted(int(s) for s in ran_ids) if ran_ids is not None \
            else sorted(int(s) for s in unique_subnets)
        self.subnet_id_to_idx = {sid: i for i, sid in enumerate(self.unique_subnet_ids)}

        for sub_id in unique_subnets:
            sub_df = (self.pivoted[self.pivoted['id_institution_subnet'] == sub_id]
                      .set_index('ds').sort_index())
            rolling_max_i = sub_df[self.slice_names].rolling(
                window=self.max_i_window, min_periods=1
            ).quantile(0.95)
            # Décalage d'un pas (shift) pour garantir l'absence stricte de fuite du futur
            rolling_max_i = rolling_max_i.shift(1).fillna(sub_df[self.slice_names].iloc[0])
            self.Max_i_subnet[sub_id] = rolling_max_i

        # Prédicteur de trafic (Passthrough par défaut si non fourni)
        if predictor is None:
            self.predictor = PassthroughTrafficPredictor()
            self.predictor.fit(self.pivoted)
        else:
            self.predictor = predictor

        # Inférence batch des prédictions sur la période de simulation
        self.pivoted_pred = self.predictor.predict_pivoted(self.pivoted, df_context=df_context)

        self.current_step_idx = 0
        self.max_steps = len(self.pivoted_pred)
        self.past_f_b = 1000.0
        self.past_eta_b = 0.95

    def reset(self) -> np.ndarray:
        """
        Réinitialise l'environnement au pas 0 et renvoie l'état initial.
        """
        self.current_step_idx = 0
        self.past_f_b = 1000.0
        self.past_eta_b = 0.95
        return self._get_state(0)

    def _get_state(self, step_idx: int) -> np.ndarray:
        """
        Construit le vecteur d'état S_t normalisé pour l'agent PPO :
          - Consommation passée f_b / 2000.0 (ramenée à ~[0,1])
          - Satisfaction QoS passée \eta_b \in [0,1]
          - Trafic futur prédit ramené à sa charge relative \hat{l} / Max_i \in [0,1]
          - Vector One-Hot identifiant le Macro-RAN courant.

        :param step_idx: Pas de temps courant.
        :return: Tableau NumPy 1D normalisé.
        """
        row_real = self.pivoted.iloc[step_idx]
        row_pred = self.pivoted_pred.iloc[step_idx]
        current_subnet_id = int(row_real['id_institution_subnet'])

        station_max_i_series = self.Max_i_subnet.get(current_subnet_id)
        ts = row_real['ds']
        if station_max_i_series is not None and ts in station_max_i_series.index:
            station_max_i = station_max_i_series.loc[ts].to_dict()
        else:
            station_max_i = next(iter(self.Max_i_subnet.values())).iloc[0].to_dict()

        # Normalisation du trafic prédit par rapport au pic Max_i
        pred_traffic_vector = [
            float(row_pred.get(f'pred_{s}', row_pred.get(s, 0.0))) / max(station_max_i.get(s, 1.0), 1.0)
            for s in self.slice_names
        ]

        # Codage One-Hot de la station RAN
        ran_onehot = [1.0 if current_subnet_id == sid else 0.0 for sid in self.unique_subnet_ids]

        state = [self.past_f_b / 2000.0, self.past_eta_b] + pred_traffic_vector + ran_onehot
        return np.array(state, dtype=np.float32)

    def step_controller(self, raw_action_dict: Dict[str, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Exécute la chaîne décisionnelle complète en 4 algorithmes pour une étape temporelle.

        :param raw_action_dict: Action brute c_{init} générée par l'agent PPO.
        :return: Tuple (nouvel état, récompense, flag de fin, dictionnaire info).
        """
        row_real = self.pivoted.iloc[self.current_step_idx]
        ts = row_real['ds']
        subnet_id = int(row_real['id_institution_subnet'])

        l_real = {s: float(row_real[s]) for s in self.slice_names}
        row_pred = self.pivoted_pred.iloc[self.current_step_idx]
        l_pred = {s: float(row_pred.get(f'pred_{s}', row_pred.get(s, 0.0))) for s in self.slice_names}

        station_max_i_series = self.Max_i_subnet.get(subnet_id)
        if station_max_i_series is not None and ts in station_max_i_series.index:
            station_max_i = station_max_i_series.loc[ts].to_dict()
        else:
            first_series = next(iter(self.Max_i_subnet.values()))
            station_max_i = first_series.iloc[0].to_dict()

        # ---------------------------------------------------------------------
        # Algorithme 1 : Décision PPO c_init (Eco1 toujours actif)
        # ---------------------------------------------------------------------
        c_init = {s: raw_action_dict.get(s, 1) for s in self.slice_names}
        c_init['Eco1'] = 1

        if l_real.get('URLLC', 0.0) > 10.0:
            c_init['URLLC'] = 1

        # ---------------------------------------------------------------------
        # Algorithme 2 : Filtrage SDN 1 Seuil PRÉDICTIF (\alpha_{seuil} * Max_{i,b})
        # ---------------------------------------------------------------------
        c_filtre = {}
        V_rediriger = {}

        for s in self.slice_names:
            max_i_s = station_max_i.get(s, 1.0)
            threshold = self.alpha_seuil * max_i_s
            # Extinction si demandée par PPO, ou si prédiction <= seuil, ou si tranche inactive historiquement
            if c_init[s] == 0 or l_pred[s] <= threshold or max_i_s <= 0:
                c_filtre[s] = 0
                V_rediriger[s] = l_real[s]
            else:
                c_filtre[s] = 1

        c_filtre['Eco1'] = 1

        # Veto déterministe URLLC : réactivation forcée si du trafic réel est mesuré
        if l_real.get('URLLC', 0.0) > 10.0 and c_filtre.get('URLLC', 1) == 0:
            c_filtre['URLLC'] = 1
            V_rediriger.pop('URLLC', None)

        # ---------------------------------------------------------------------
        # Algorithme 3 : Routage et Débordement (Contrôleur SDN 2)
        # ---------------------------------------------------------------------
        priority_order = ['URLLC', 'URLLC_eMBB_MIX', 'mMTC', 'eMBB']
        V_rediriger_sorted = {s: V_rediriger[s] for s in priority_order if s in V_rediriger}
        for s, traf in V_rediriger.items():
            if s not in V_rediriger_sorted:
                V_rediriger_sorted[s] = traf

        total_redir = sum(V_rediriger_sorted.values())
        l_eco1_b = l_real.get('Eco1', 0.0) + total_redir
        l_eco2_b = 0.0

        c_eco2 = 0
        surcharge = False

        if l_eco1_b > self.seuil_75:
            c_eco2 = 1
            excess = l_eco1_b - self.seuil_75
            l_eco1_b = self.seuil_75
            l_eco2_b = excess
            if l_eco2_b > self.capacity_eco2:
                surcharge = True

        # ---------------------------------------------------------------------
        # Algorithme 4 : Sécurité et Réallocation (Contrôleur SDN 2)
        # ---------------------------------------------------------------------
        c_final = c_filtre.copy()
        c_final['Eco1'] = 1
        c_final['Eco2'] = c_eco2

        # Rallumage d'urgence de la tranche la plus lourde si surcharge détectée
        if surcharge:
            non_eco_active = [s for s in self.slice_names if c_final[s] == 1]
            if len(non_eco_active) < len(self.slice_names):
                heaviest_slice = max(
                    [s for s in self.slice_names if c_final[s] == 0],
                    key=lambda k: l_real[k],
                    default=None
                )
                if heaviest_slice:
                    c_final[heaviest_slice] = 1
                    l_eco1_b = max(0.0, l_eco1_b - l_real[heaviest_slice])

        # Calcul physique (consommation f_b et satisfaction QoS \eta_b)
        res_step = self.ran_sim.step(ts, subnet_id, l_real, c_final)
        f_b_t = res_step['f_b']
        eta_b_t = res_step['eta_b']

        # Calcul du référentiel All-Active pour déterminer le gain d'énergie \Delta E(t)
        all_active = {s: 1 for s in self.slice_names}
        all_active['Eco1'] = 1
        all_active['Eco2'] = 0
        res_base = self.ran_sim.step(ts, subnet_id, l_real, all_active)
        f_b_base = res_base['f_b']

        delta_E_t = max(0.0, (f_b_base - f_b_t) / f_b_base) if f_b_base > 0 else 0.0

        total_eco_traffic = l_eco1_b + l_eco2_b
        if surcharge and total_eco_traffic > 0:
            L_t = max(0.0, (total_eco_traffic - (self.capacity_eco1 + self.capacity_eco2)) / total_eco_traffic)
        else:
            L_t = 0.0

        qos_violation = (eta_b_t < 1.0) or (L_t > 0.0)

        # Calcul de la récompense multi-objectif r_t
        reward = (1000.0 / f_b_t) + (self.beta * eta_b_t) - (self.lambda_loss * L_t)

        self.past_f_b = f_b_t
        self.past_eta_b = eta_b_t

        self.current_step_idx += 1
        done = (self.current_step_idx >= self.max_steps)

        next_state = self._get_state(self.current_step_idx if not done else self.current_step_idx - 1)

        info = {
            'timestamp': ts,
            'subnet_id': subnet_id,
            'f_b_t': f_b_t,
            'f_b_base': f_b_base,
            'delta_E_t': delta_E_t,
            'eta_b_t': eta_b_t,
            'c_final': c_final,
            'c_eco2': c_eco2,
            'surcharge': surcharge,
            'qos_violation': qos_violation,
            'L_t': L_t,
            'rho': res_step['rho'],
            'qos_slice': res_step['qos_slice'],
            'delta_slice': res_step.get('delta_slice', {}),
            'l_real': l_real,
            'l_pred': l_pred,
            'total_real_traffic': sum(l_real.values()),
            'total_pred_traffic': sum(l_pred.values())
        }

        return next_state, reward, done, info