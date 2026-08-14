#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
====================================================================================================
 MODULE : src/simulator/ran_simulator.py
 OBJET  : Moteur Physique & Énergétique du Réseau d'Accès Radio (RAN) 5G/6G
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce module simule la couche physique et énergétique d'une station de base (gNodeB) 5G/6G
supportant le découpage en tranches réseau (Network Slicing).
Il est instancié au sein de l'environnement Gymnasium (`src/environment/sdn_controller_env.py`)
et calcule à chaque pas de temps $\tau = 10$ minutes :
  1. La répartition dynamique du trafic et la charge relative ($\rho_{i,b}$).
  2. La consommation électrique globale en Watts ($f_b$) basée sur le modèle 5G (Phyu et al., 2023).
  3. La latence atteinte ($\delta_i$) par tranche et le taux de satisfaction QoS moyen ($\eta_b$).

FORMULES MATHÉMATIQUES & PARAMÈTRES PHYSIQUES :
------------------------------------------------
  - Consommation par tranche active $i$ :
      E_{i,b}^t = c_i^t \cdot (\rho_{i,b}^t \cdot \psi_i \cdot P_b^{\text{dynamic}} + \psi_i \cdot P_b^{\text{fixed}})
      avec :
        * P_static  = 18.0 W   (Puissance statique permanente)
        * P_fixed   = 139.0 W  (Surcoût fixe par tranche active VNF/CNF)
        * P_dynamic = 742.0 W  (Puissance dynamique maximale à 100% de charge)
        * \psi      = {URLLC: 1.4, mMTC: 1.2, eMBB: 1.6, Eco: 1.0} (Coefficients d'amplification)

  - Satisfaction de Latence ($\eta$) :
      - Slice ACTIVE   : \delta_i = \delta_{\text{active}}[i] (latence nominale du service < exigence SLA)
      - Slice ÉTEINTE  : \delta_i = \delta_{\text{eco}} (trafic réacheminé vers l'EcoSlice, ex: 11.0 ms)
      - \eta_i = 1.0 si \delta_i \le d_i^{\text{SLA}}, sinon dégradation linéaire.
====================================================================================================
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class RAN_Simulator:
    """
    Simulateur de la couche physique et du bilan énergétique d'une station RAN 5G/6G.
    """

    def __init__(
        self,
        tau_minutes: int = 10,
        P_static: float = 18.0,
        P_fixed: float = 139.0,
        P_dynamic: float = 742.0,
        delta_active: Optional[Dict[str, float]] = None,
        delta_eco: float = 25.0,
        include_ecoslice_in_qos: bool = False,
    ):
        """
        Initialise les constantes matérielles et les seuils de latence SLA pour chaque tranche.

        :param tau_minutes: Durée d'un pas de temps en minutes (SADI : Slice Activation Interval).
        :param P_static: Puissance statique permanente du gNodeB (Watts).
        :param P_fixed: Puissance fixe par tranche active (Watts).
        :param P_dynamic: Puissance dynamique maximale à pleine charge (Watts).
        :param delta_active: Dictionnaire des latences obtenues quand une tranche est active (ms).
        :param delta_eco: Latence de réacheminement vers l'EcoSlice (ms).
        :param include_ecoslice_in_qos: Si True, inclut EcoSlice dans la moyenne QoS globale (Option V9).
        """
        self.tau_minutes = tau_minutes
        self.P_static = P_static
        self.P_fixed = P_fixed
        self.P_dynamic = P_dynamic
        self.include_ecoslice_in_qos = include_ecoslice_in_qos

        # Coefficients d'amplification énergétique par tranche (\psi_i)
        self.psi = {
            'URLLC': 1.4,
            'mMTC': 1.2,
            'eMBB': 1.6,
            'URLLC_eMBB_MIX': 1.3,
            'Eco1': 1.0,
            'Eco2': 1.0
        }

        # Exigences de latence SLA d_u (ms) par classe de service 5G
        self.d_u_requirements = {
            'URLLC': 1.0,
            'URLLC_eMBB_MIX': 5.0,
            'eMBB': 10.0,
            'mMTC': 20.0,
            'Eco1': 50.0,
            'Eco2': 50.0,
        }

        # Latences prédéfinies obtenues lorsque la tranche est active (delta_active < d_u)
        self.delta_active = delta_active or {
            'URLLC': 0.8,           # Conforme aux SLA (1.0 ms)
            'URLLC_eMBB_MIX': 4.0,  # Conforme aux SLA (5.0 ms)
            'eMBB': 8.0,            # Conforme aux SLA (10.0 ms)
            'mMTC': 15.0,           # Conforme aux SLA (20.0 ms)
        }

        # Latence fixe subie lors d'une redirection vers l'EcoSlice
        self.delta_eco = delta_eco

    def compute_bandwidth_ratio(
        self,
        slice_traffic: Dict[str, float],
        slice_states: Dict[str, int]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calcule la charge relative \rho_{i,b} de chaque tranche active.
        Le trafic des tranches désactivées (c_i = 0) est comptabilisé dans l'EcoSlice 1.

        :param slice_traffic: Dictionnaire du trafic entrant par tranche.
        :param slice_states: État binaire (0 ou 1) de chaque tranche.
        :return: Tuple (ratios \rho, trafic effectif par tranche).
        """
        total_traffic = sum(slice_traffic.values())
        if total_traffic <= 0:
            default_rho = {s: 0.0 for s in slice_traffic.keys()}
            return default_rho, slice_traffic.copy()

        effective_traffic = {}
        redirected_traffic = 0.0

        for s, traffic in slice_traffic.items():
            state = slice_states.get(s, 1)
            if state == 1 and s not in ['Eco1', 'Eco2']:
                effective_traffic[s] = traffic
            else:
                effective_traffic[s] = 0.0
                redirected_traffic += traffic

        # EcoSlice 1 absorbe le trafic orphelin issu des tranches éteintes
        effective_traffic['Eco1'] = effective_traffic.get('Eco1', 0.0) + redirected_traffic
        if 'Eco2' in slice_states and slice_states['Eco2'] == 1:
            effective_traffic['Eco2'] = effective_traffic.get('Eco2', 0.0)

        # Calcul du ratio de charge \rho_{i,b}
        rho = {}
        for s, eff_t in effective_traffic.items():
            rho[s] = eff_t / total_traffic if total_traffic > 0 else 0.0

        return rho, effective_traffic

    def compute_energy_consumption(
        self,
        rho: Dict[str, float],
        slice_states: Dict[str, int]
    ) -> float:
        r"""
        Calcule la puissance électrique totale consommée par la station en Watts :
        f_b(t) = P_static + sum_i c_i^t * [ \rho_{i,b}^t * \psi_i * P_dynamic + \psi_i * P_fixed ]

        :param rho: Ratios de charge par tranche.
        :param slice_states: État d'activation des tranches.
        :return: Consommation totale en Watts (f_b).
        """
        total_energy = self.P_static

        for s, rho_val in rho.items():
            state = slice_states.get(s, 0 if s not in ['Eco1'] else 1)
            if state == 1:
                psi_val = self.psi.get(s, 1.0)
                e_slice = (rho_val * psi_val * self.P_dynamic) + (psi_val * self.P_fixed)
                total_energy += e_slice

        return total_energy

    def compute_qos_satisfaction(
        self,
        slice_traffic: Dict[str, float],
        slice_states: Dict[str, int]
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        r"""
        Évalue la latence subie \delta_i et le score de satisfaction QoS \eta_i par tranche,
        puis calcule la satisfaction globale \eta_b^t (moyenne arithmétique).

        Pourquoi cette logique ?
        - Tranche active (c_i=1) : latence nominale \delta_{\text{active}} \le d_u \implies \eta_i = 1.0.
        - Tranche éteinte (c_i=0) : trafic déversé sur EcoSlice avec latence \delta_{\text{eco}}.
          Si \delta_{\text{eco}} > d_u, la satisfaction se dégrade de façon linéaire.

        :param slice_traffic: Trafic entrant.
        :param slice_states: État d'activation.
        :return: Tuple (score QoS moyen \eta_b, dict des scores QoS par tranche, dict des latences par tranche).
        """
        active_slices = [s for s in slice_traffic.keys() if s not in ['Eco1', 'Eco2']]
        if len(active_slices) == 0:
            return 1.0, {s: 1.0 for s in slice_traffic.keys()}, {s: 0.0 for s in slice_traffic.keys()}

        qos_slice = {}
        delta_slice = {}
        for s in active_slices:
            traffic = slice_traffic[s]
            is_active = slice_states.get(s, 1) == 1
            achieved_delay = self.delta_active.get(s, 10.0) if is_active else self.delta_eco
            req_delay = self.d_u_requirements.get(s, 20.0)

            delta_slice[s] = achieved_delay

            if traffic <= 0:
                qos_slice[s] = 1.0
                continue

            # Évaluation du respect des exigences SLA de latence
            if achieved_delay <= req_delay:
                satisfaction = 1.0
            else:
                satisfaction = max(0.0, 1.0 - (achieved_delay - req_delay) / req_delay)

            qos_slice[s] = satisfaction

        # Prise en compte facultative de l'EcoSlice dans la moyenne (Option V9)
        if self.include_ecoslice_in_qos:
            for eco_name in ['Eco1', 'Eco2']:
                is_eco_active = slice_states.get(eco_name, 1 if eco_name == 'Eco1' else 0) == 1
                if not is_eco_active:
                    continue
                req_delay = self.d_u_requirements.get(eco_name, 50.0)
                satisfaction = 1.0 if self.delta_eco <= req_delay else max(
                    0.0, 1.0 - (self.delta_eco - req_delay) / req_delay)
                qos_slice[eco_name] = satisfaction
                delta_slice[eco_name] = self.delta_eco

        # Moyenne arithmétique équitable de la satisfaction QoS sur les tranches evaluated
        overall_qos = float(np.mean(list(qos_slice.values()))) if len(qos_slice) > 0 else 1.0
        return overall_qos, qos_slice, delta_slice

    def step(
        self,
        timestamp: Any,
        subnet_id: int,
        slice_traffic: Dict[str, float],
        slice_states: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Simule une étape temporelle de 10 minutes et renvoie le bilan énergétique et QoS.

        :param timestamp: Horodatage courant.
        :param subnet_id: Identifiant de la station RAN.
        :param slice_traffic: Trafic par tranche.
        :param slice_states: Actions d'activation décidées.
        :return: Dictionnaire des résultats physiques (f_b, eta_b, rho, qos_slice, delta_slice).
        """
        # EcoSlice 1 est maintenue active par défaut comme tranche de secours principale
        slice_states_adj = slice_states.copy()
        slice_states_adj['Eco1'] = 1

        rho, effective_traffic = self.compute_bandwidth_ratio(slice_traffic, slice_states_adj)
        f_b = self.compute_energy_consumption(rho, slice_states_adj)
        eta_b, qos_slice, delta_slice = self.compute_qos_satisfaction(slice_traffic, slice_states_adj)

        return {
            'timestamp': timestamp,
            'subnet_id': subnet_id,
            'f_b': f_b,
            'eta_b': eta_b,
            'rho': rho,
            'effective_traffic': effective_traffic,
            'qos_slice': qos_slice,
            'delta_slice': delta_slice
        }