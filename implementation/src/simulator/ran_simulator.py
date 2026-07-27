#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/simulator/ran_simulator.py
 OBJET  : Moteur Physique & Énergétique du Réseau d'Accès Radio (RAN) 5G/6G
====================================================================================================

DESCRIPTION DÉTAILLÉE :
-----------------------
Ce module simule la couche physique et énergétique d'une station de base (gNodeB) 5G/6G
supportant le découpage en tranches réseau (Network Slicing). Il calcule à chaque pas de temps 
tau = 10 minutes (SADI : Slice Activation/Deactivation Interval) :
  1. La répartition dynamique du trafic et de la bande passante (ratio rho_{i,b}).
  2. La consommation électrique globale en Watts (f_b) selon le modèle de puissance 5G RAN (Phyu et al., 2023).
  3. La latence atteinte (delta_i) par tranche et le taux de satisfaction QoS moyen (eta_b)
     calculé par moyenne arithmétique non pondérée selon l'Étape 4 (Ligne 17 du PDF des Algorithmes) :
       eta_b^t = (1 / |I_b|) * sum_{i in I_b} eta_{i,b}^t

FORMULES MATHÉMATIQUES & CONSTANTES PHYSIQUES :
----------------------------------------------
  - Consommation par tranche active i :
      E_{i,b}^t = c_i^t * (rho_{i,b}^t * psi_i * P_b^dynamic + psi_i * P_b^fixed)
      avec :
        * P_static  = 18.0 W   (Puissance statique permanente)
        * P_fixed   = 139.0 W  (Surcoût fixe par tranche active VNF/CNF)
        * P_dynamic = 742.0 W  (Puissance dynamique maximale à 100% de charge)
        * psi       = {URLLC: 1.4, mMTC: 1.2, eMBB: 1.6, Eco: 1.0} (Coefficients d'amplification)

  - Satisfaction de Latence (eta) — CORRIGÉ (Option A, fidèle à Phyu et al., Section III-A & Eq. 4) :
      delta_i est une latence PRÉDÉFINIE, FIXE par slice (« a predefined achievable delay »),
      PAS une fonction de rho. Deux valeurs fixes par slice :
        * delta_active[i]  : latence obtenue quand la slice i est ACTIVE (c_i = 1)
        * delta_eco        : latence obtenue quand le trafic de i est redirigé vers l'EcoSlice
                             (= 11 ms, valeur exacte de la Table I de l'article pour l'EcoSlice)
      eta_u = 1.0 si delta_i <= d_u (exigence de latence du service) sinon dégradation graduelle
      (l'article utilise un seuil strictement binaire, Eq. 4 ; la dégradation graduelle ci-dessous
      est un choix de shaping de récompense plus doux pour l'entraînement RL, PAS une exigence de
      l'article — à retirer si vous voulez suivre l'Eq. 4 à la lettre).
      Exigences d_u : URLLC = 1.0 ms, URLLC_eMBB_MIX = 5.0 ms, eMBB = 10.0 ms, mMTC = 20.0 ms.

====================================================================================================
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class RAN_Simulator:
    """
    Simulateur de Couche Physique et Énergétique RAN 5G/6G.
    """

    def __init__(
        self,
        tau_minutes: int = 10,
        P_static: float = 18.0,
        P_fixed: float = 139.0,
        P_dynamic: float = 742.0,
        delta_active: Optional[Dict[str, float]] = None,
        delta_eco: float = 11.0,
    ):
        self.tau_minutes = tau_minutes
        self.P_static = P_static
        self.P_fixed = P_fixed
        self.P_dynamic = P_dynamic

        # Coefficients d'amplification énergétique par tranche (psi_i)
        self.psi = {
            'URLLC': 1.4,
            'mMTC': 1.2,
            'eMBB': 1.6,
            'URLLC_eMBB_MIX': 1.3,
            'Eco1': 1.0,
            'Eco2': 1.0
        }

        # Exigences de latence SLA utilisateur d_u (ms) — le seuil à respecter
        self.d_u_requirements = {
            'URLLC': 1.0,
            'URLLC_eMBB_MIX': 5.0,
            'eMBB': 10.0,
            'mMTC': 20.0,
            'Eco1': 50.0,
            'Eco2': 50.0
        }

        # ── CORRECTION Option A ──
        # Latence PRÉDÉFINIE et FIXE quand la slice est active (delta_i de l'article,
        # Section III-A : "a predefined achievable delay"). Choisie confortablement en
        # dessous de l'exigence d_u correspondante, pour que la slice soit satisfaite
        # tant qu'elle reste active — exactement l'intention de l'article.
        self.delta_active = delta_active or {
            'URLLC': 0.8,           # < exigence 1.0 ms
            'URLLC_eMBB_MIX': 4.0,  # < exigence 5.0 ms
            'eMBB': 8.0,            # < exigence 10.0 ms
            'mMTC': 15.0,           # < exigence 20.0 ms
        }

        # Latence de l'EcoSlice : fixe, indépendante du rho — 11 ms est la valeur EXACTE
        # de la Table I de l'article pour l'EcoSlice (Facebook/YouTube/Google/EcoSlice = [10,1,15,11]).
        self.delta_eco = delta_eco

    def compute_bandwidth_ratio(
        self,
        slice_traffic: Dict[str, float],
        slice_states: Dict[str, int]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calcule l'allocation effective du trafic et les ratios de bande passante rho_{i,b}.
        Le trafic des tranches désactivées (c_i = 0) est basculé vers EcoSlice 1.
        Utilisé UNIQUEMENT pour le modèle d'énergie (Eq. 2-3 de l'article) — plus du tout
        pour la latence/QoS depuis la correction Option A.
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

        # EcoSlice 1 absorbe le trafic redirigé
        effective_traffic['Eco1'] = effective_traffic.get('Eco1', 0.0) + redirected_traffic
        if 'Eco2' in slice_states and slice_states['Eco2'] == 1:
            effective_traffic['Eco2'] = effective_traffic.get('Eco2', 0.0)

        rho = {}
        for s, eff_t in effective_traffic.items():
            rho[s] = eff_t / total_traffic if total_traffic > 0 else 0.0

        return rho, effective_traffic

    def compute_energy_consumption(
        self,
        rho: Dict[str, float],
        slice_states: Dict[str, int]
    ) -> float:
        """
        Calcule la puissance électrique totale consommée en Watts : f_b(t) = sum_i E_{i,b}^t + P_static
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
        """
        Évalue la latence delta_i et le taux de satisfaction QoS eta_b^t.

        CORRIGÉ (Option A) : delta_i est une constante prédéfinie par slice (Section III-A
        de l'article), PAS une fonction de rho. Ne dépend que de l'état ON/OFF de la slice :
          - slice ACTIVE   -> delta_i = self.delta_active[i]  (fixe, < exigence par construction)
          - slice INACTIVE -> delta_i = self.delta_eco = 11 ms (redirigée vers l'EcoSlice,
                               valeur exacte de la Table I de l'article)

        Formule finale (Eq. 4 de l'article) :
          eta_b^t = (1 / |I_b|) * sum_{i in I_b} eta_{i,b}^t
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

            if achieved_delay <= req_delay:
                satisfaction = 1.0
            else:
                satisfaction = max(0.0, 1.0 - (achieved_delay - req_delay) / req_delay)

            qos_slice[s] = satisfaction

        # Ligne 17 du PDF : Moyenne arithmétique équitable entre les slices
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
        Avance d'un pas de temps de 10 minutes et retourne le bilan d'énergie et de QoS.
        """
        # S'assurer que EcoSlice 1 est toujours activée
        slice_states_adj = slice_states.copy()
        slice_states_adj['Eco1'] = 1

        rho, effective_traffic = self.compute_bandwidth_ratio(slice_traffic, slice_states_adj)
        f_b = self.compute_energy_consumption(rho, slice_states_adj)
        # CORRIGÉ : compute_qos_satisfaction prend maintenant slice_states (pas rho)
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