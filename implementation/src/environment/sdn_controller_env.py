#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/environment/sdn_controller_env.py
 OBJET  : Environnement d'Apprentissage par Renforcement SDN à Double Contrôleur (Multi-RAN / Multi-Subnets)
====================================================================================================

DESCRIPTION DÉTAILLÉE :
-----------------------
Cet environnement (compatible avec l'interface OpenAI Gym) implémente les 4 Algorithmes
du Double Contrôleur SDN décrits dans le document d'architecture pour une ou plusieurs stations RAN :

  - Algorithme 1 (PPO Decision) :
      Reçoit l'action binaire c_init^t de l'agent PPO pour les slices spécialisées.
      Impose l'activation permanente d'EcoSlice 1 (c_eco1 = 1).

  - Algorithme 2 (SDN 1 Threshold Filter) :
      Éteint les slices dont le trafic est sous le seuil d'extinction propre à chaque station b :
        l_{i,b}^t < alpha_seuil * Max_{i,b}  (Max_{i,b} basé sur le 95e percentile du subnet b).
      Redirige le trafic orphelin vers le vecteur V_rediriger.

  - Algorithme 3 (SDN 2 Priority Routing & Eco2) :
      Trie V_rediriger par ordre de priorité décroissante (URLLC > URLLC_eMBB_MIX > mMTC > eMBB).
      Remplit EcoSlice 1 jusqu'à sa capacité C_eco1.
      Si la charge d'EcoSlice 1 dépasse 75% (Seuil75 = 0.75 * C_eco1), active **EcoSlice Bis** (c_eco2 = 1).
      Si EcoSlice Bis déborde à son tour (charge > C_eco2), déclenche l'alerte Surcharge = Vrai.

  - Algorithme 4 (SDN 2 Safety Fallback, Allocations & Loss Rate) :
      En cas de surcharge ou d'atteinte critique à URLLC, rallume la slice d'origine du flux majeur.
      Calcule le gain énergétique Delta E(t), le taux de perte L(t) et le taux de satisfaction QoS eta_b^t
      selon l'Étape 4 (Ligne 17 du PDF).

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
    Environnement SDN à Double Contrôleur pour le RAN Slicing 5G/6G (Support Multi-RAN / Multi-Subnets).
    """

    def __init__(
        self,
        df_traffic: pd.DataFrame,
        predictor: Optional[BaseTrafficPredictor] = None,
        slice_names: Optional[List[str]] = None,
        df_context: Optional[pd.DataFrame] = None,
        alpha_seuil: float = 0.01,
        capacity_eco1: float = 2500000.0,
        capacity_eco2: float = 2500000.0,
        seuil_75_ratio: float = 0.75,
        beta: float = 10.0,
        lambda_loss: float = 50.0,
        seed: int = 42
    ):
        self.df_traffic = df_traffic.copy()
        self.alpha_seuil = alpha_seuil
        self.capacity_eco1 = capacity_eco1
        self.capacity_eco2 = capacity_eco2
        self.seuil_75 = seuil_75_ratio * capacity_eco1
        self.beta = beta
        self.lambda_loss = lambda_loss
        self.seed = seed

        np.random.seed(seed)
        self.ran_sim = RAN_Simulator()

        # Pivotement des données brutes
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

        # ── CORRECTION (Fix 2) : Max_i causal, jamais de fuite du futur ──
        # Avant : un seul 95e percentile calculé une fois sur TOUT self.pivoted
        # (train + test confondus) -> le seuil d'extinction pouvait être influencé
        # par un pic de trafic qui n'existe que dans la période de test, ce qu'un
        # vrai contrôleur déployé ne pourrait jamais connaître à l'avance.
        # Maintenant : Max_i est une série qui varie dans le temps, calculée sur une
        # fenêtre glissante des pas de temps PASSÉS uniquement (décalée d'un pas),
        # par sous-réseau (station RAN).
        self.max_i_window = 1008  # ~7 jours à 10 min — ajustez si votre pas de temps diffère
        self.Max_i_subnet = {}
        unique_subnets = self.pivoted['id_institution_subnet'].unique()
        for sub_id in unique_subnets:
            sub_df = (self.pivoted[self.pivoted['id_institution_subnet'] == sub_id]
                      .set_index('ds').sort_index())
            rolling_max_i = sub_df[self.slice_names].rolling(
                window=self.max_i_window, min_periods=1
            ).quantile(0.95)
            # décalage d'un pas : au pas t, on ne connaît que les pics jusqu'à t-1
            rolling_max_i = rolling_max_i.shift(1).fillna(sub_df[self.slice_names].iloc[0])
            self.Max_i_subnet[sub_id] = rolling_max_i

        # Prédicteur par défaut Passthrough (Oracle) si non spécifié
        if predictor is None:
            self.predictor = PassthroughTrafficPredictor()
            self.predictor.fit(self.pivoted)
        else:
            self.predictor = predictor

        # Précalcul des prédictions de trafic avec contexte glissant
        self.pivoted_pred = self.predictor.predict_pivoted(self.pivoted, df_context=df_context)

        self.current_step_idx = 0
        self.max_steps = len(self.pivoted_pred)
        self.past_f_b = 1000.0
        self.past_eta_b = 0.95

    def reset(self) -> np.ndarray:
        self.current_step_idx = 0
        self.past_f_b = 1000.0
        self.past_eta_b = 0.95
        return self._get_state(0)

    def _get_state(self, step_idx: int) -> np.ndarray:
        row_pred = self.pivoted_pred.iloc[step_idx]
        pred_traffic_vector = [float(row_pred.get(f'pred_{s}', row_pred.get(s, 0.0))) for s in self.slice_names]

        state = [self.past_f_b / 2000.0, self.past_eta_b] + pred_traffic_vector
        return np.array(state, dtype=np.float32)

    def step_controller(self, raw_action_dict: Dict[str, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        row_real = self.pivoted.iloc[self.current_step_idx]
        ts = row_real['ds']
        subnet_id = int(row_real['id_institution_subnet'])

        l_real = {s: float(row_real[s]) for s in self.slice_names}
        row_pred = self.pivoted_pred.iloc[self.current_step_idx]
        l_pred = {s: float(row_pred.get(f'pred_{s}', row_pred.get(s, 0.0))) for s in self.slice_names}

        # Obtenir les pics Max_i de la station RAN spécifique, AU TEMPS ts (causal)
        station_max_i_series = self.Max_i_subnet.get(subnet_id)
        if station_max_i_series is not None and ts in station_max_i_series.index:
            station_max_i = station_max_i_series.loc[ts].to_dict()
        else:
            # repli si le timestamp n'est pas trouvé (ne devrait pas arriver en usage normal)
            first_series = next(iter(self.Max_i_subnet.values()))
            station_max_i = first_series.iloc[0].to_dict()

        # ---------------------------------------------------------------------
        # Algorithme 1 : Décision PPO c_init (Eco1 toujours actif)
        # ---------------------------------------------------------------------
        c_init = {s: raw_action_dict.get(s, 1) for s in self.slice_names}
        c_init['Eco1'] = 1

        # Protection URLLC : Si du trafic URLLC est présent, forcer c_init['URLLC'] = 1
        if l_real.get('URLLC', 0.0) > 10.0:
            c_init['URLLC'] = 1

        # ---------------------------------------------------------------------
        # Algorithme 2 : Filtrage SDN 1 Seuil (alpha_seuil * Max_{i,b})
        # ---------------------------------------------------------------------
        c_filtre = {}
        V_rediriger = {}

        for s in self.slice_names:
            threshold = self.alpha_seuil * station_max_i.get(s, 1.0)
            if c_init[s] == 0 or l_real[s] < threshold:
                c_filtre[s] = 0
                V_rediriger[s] = l_real[s]
            else:
                c_filtre[s] = 1

        c_filtre['Eco1'] = 1

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

        # Lignes 2-6 du PDF : Rallumage d'urgence du flux le plus lourd si Surcharge
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

        # Moteur Physique RAN
        res_step = self.ran_sim.step(ts, subnet_id, l_real, c_final)
        f_b_t = res_step['f_b']
        eta_b_t = res_step['eta_b']

        # Station All-Active pour calculer Delta E(t)
        all_active = {s: 1 for s in self.slice_names}
        all_active['Eco1'] = 1
        all_active['Eco2'] = 0
        res_base = self.ran_sim.step(ts, subnet_id, l_real, all_active)
        f_b_base = res_base['f_b']

        # Ligne 11 du PDF : Gain Énergétique Delta E(t)
        delta_E_t = max(0.0, (f_b_base - f_b_t) / f_b_base) if f_b_base > 0 else 0.0
        
        # Lignes 12-16 du PDF : Taux de perte L(t)
        total_eco_traffic = l_eco1_b + l_eco2_b
        if surcharge and total_eco_traffic > 0:
            L_t = max(0.0, (total_eco_traffic - (self.capacity_eco1 + self.capacity_eco2)) / total_eco_traffic)
        else:
            L_t = 0.0

        qos_violation = (eta_b_t < 1.0) or (L_t > 0.0)

        # Fonction de Récompense PPO r_t
        reward = (1.0 / f_b_t) + (self.beta * eta_b_t) - (self.lambda_loss * L_t)

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