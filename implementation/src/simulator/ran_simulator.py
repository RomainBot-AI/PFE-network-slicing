#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical and energy model of a 5G/6G RAN base station (gNodeB) with slicing.

Following the 5G RAN power model of Phyu et al. (2023), each step (tau = 10 min)
computes:
  1. Dynamic traffic/bandwidth allocation (ratio rho_{i,b}).
  2. Total power draw in Watts, f_b.
  3. Per-slice latency (delta_i) and the mean QoS satisfaction eta_b, the
     unweighted mean over slices: eta_b^t = (1/|I_b|) * sum_i eta_{i,b}^t.

Energy per active slice i:
    E_{i,b}^t = c_i^t * (rho_{i,b}^t * psi_i * P_dynamic + psi_i * P_fixed)
  P_static = 18 W (always on), P_fixed = 139 W (per active VNF/CNF slice),
  P_dynamic = 742 W (peak at 100% load), psi = amplification per slice.

Latency (Option A, faithful to Phyu et al., Section III-A / Eq. 4): delta_i is a
predefined constant per slice, not a function of rho -- delta_active[i] when the
slice is active, delta_eco (= 11 ms, the paper's EcoSlice value) when its traffic
is redirected to the EcoSlice. eta = 1 if delta_i <= d_u else a graceful decay.
The paper uses a strict binary threshold (Eq. 4); the decay below is softer
reward shaping for RL and can be removed to match Eq. 4 exactly. Requirements
d_u: URLLC 1 ms, URLLC_eMBB_MIX 5 ms, eMBB 10 ms, mMTC 20 ms.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class RAN_Simulator:
    """5G/6G RAN physical and energy layer simulator."""

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

        # Per-slice energy amplification coefficients (psi_i).
        self.psi = {
            'URLLC': 1.4,
            'mMTC': 1.2,
            'eMBB': 1.6,
            'URLLC_eMBB_MIX': 1.3,
            'Eco1': 1.0,
            'Eco2': 1.0
        }

        # User SLA latency requirements d_u (ms) -- the threshold to meet.
        self.d_u_requirements = {
            'URLLC': 1.0,
            'URLLC_eMBB_MIX': 5.0,
            'eMBB': 10.0,
            'mMTC': 20.0,
            'Eco1': 50.0,
            'Eco2': 50.0
        }

        # Option A: fixed predefined latency when the slice is active (paper's
        # delta_i, Section III-A). Chosen comfortably below the matching d_u so an
        # active slice stays satisfied -- the paper's intent.
        self.delta_active = delta_active or {
            'URLLC': 0.8,           # < requirement 1.0 ms
            'URLLC_eMBB_MIX': 4.0,  # < requirement 5.0 ms
            'eMBB': 8.0,            # < requirement 10.0 ms
            'mMTC': 15.0,           # < requirement 20.0 ms
        }

        # EcoSlice latency: fixed, independent of rho. 11 ms is the paper's
        # exact EcoSlice value (Table I).
        self.delta_eco = delta_eco

    def compute_bandwidth_ratio(
        self,
        slice_traffic: Dict[str, float],
        slice_states: Dict[str, int]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute effective traffic allocation and bandwidth ratios rho_{i,b}.

        Traffic of deactivated slices (c_i = 0) is redirected to EcoSlice 1. Used
        only for the energy model (Eq. 2-3); no longer for latency/QoS since
        Option A.
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

        # EcoSlice 1 absorbs the redirected traffic.
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
        """Total power draw in Watts: f_b(t) = sum_i E_{i,b}^t + P_static."""
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
        """Evaluate per-slice latency delta_i and QoS satisfaction eta_b^t.

        Option A: delta_i is a predefined per-slice constant (Section III-A), not
        a function of rho. It depends only on the slice ON/OFF state:
          - active   -> delta_i = self.delta_active[i] (fixed, < requirement)
          - inactive -> delta_i = self.delta_eco = 11 ms (redirected to EcoSlice)

        Final formula (Eq. 4): eta_b^t = (1/|I_b|) * sum_{i in I_b} eta_{i,b}^t.
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

        # Eq. 4: unweighted arithmetic mean across slices.
        overall_qos = float(np.mean(list(qos_slice.values()))) if len(qos_slice) > 0 else 1.0
        return overall_qos, qos_slice, delta_slice

    def step(
        self,
        timestamp: Any,
        subnet_id: int,
        slice_traffic: Dict[str, float],
        slice_states: Dict[str, int]
    ) -> Dict[str, Any]:
        """Advance one 10-minute step and return the energy and QoS summary."""
        # EcoSlice 1 is always on.
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