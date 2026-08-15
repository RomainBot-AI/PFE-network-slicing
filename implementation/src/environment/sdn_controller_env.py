#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SDN double-controller RL environment (Gym-style, multi-RAN / multi-subnet).

Implements the four SDN double-controller algorithms for one or more RAN
stations:

  - Algorithm 1 (PPO decision): take the agent's binary action c_init for the
    specialised slices; EcoSlice 1 is always on (c_eco1 = 1).
  - Algorithm 2 (SDN 1 threshold filter): switch off slices whose traffic is
    below the per-station shutdown threshold l_{i,b} < alpha * Max_{i,b}
    (Max_{i,b} = causal 95th percentile of subnet b); redirect their traffic.
  - Algorithm 3 (SDN 2 priority routing & Eco2): sort redirected traffic by
    priority (URLLC > URLLC_eMBB_MIX > mMTC > eMBB), fill EcoSlice 1 up to
    capacity; above 75% activate EcoSlice 2; if it also overflows, flag overload.
  - Algorithm 4 (SDN 2 safety fallback, allocations & loss): on overload or a
    critical URLLC hit, re-activate the heaviest slice; compute the energy gain
    Delta E(t), loss rate L(t), and QoS satisfaction eta_b^t.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from src.simulator.ran_simulator import RAN_Simulator
from src.models.base_predictor import BaseTrafficPredictor
from src.models.passthrough_predictor import PassthroughTrafficPredictor


class SDN_DoubleController_Env:
    """SDN double-controller environment for 5G/6G RAN slicing (multi-RAN)."""

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
        energy_reward_scale: float = 1000.0,
        seed: int = 42,
        ran_ids: Optional[List[int]] = None
    ):
        self.df_traffic = df_traffic.copy()
        self.alpha_seuil = alpha_seuil
        self.beta = beta
        self.lambda_loss = lambda_loss
        self.energy_reward_scale = energy_reward_scale
        self.seed = seed

        np.random.seed(seed)
        self.ran_sim = RAN_Simulator(include_ecoslice_in_qos=True)

        # Pivot the raw long-format frame to one column per slice.
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

        # EcoSlice capacities default to the 95th percentile of the total traffic
        # carried in one step, so they scale with whatever dataset is replayed
        # instead of assuming a fixed byte budget.
        auto_capacity = max(float(self.pivoted[self.slice_names].sum(axis=1).quantile(0.95)), 1.0)
        self.capacity_eco1 = auto_capacity if capacity_eco1 is None else capacity_eco1
        self.capacity_eco2 = auto_capacity if capacity_eco2 is None else capacity_eco2
        self.seuil_75 = seuil_75_ratio * self.capacity_eco1

        # Causal Max_i, with no future leakage: instead of a single 95th
        # percentile over the whole panel (train + test), Max_i is a time-varying
        # series computed over a rolling window of PAST steps only (shifted by one
        # step), per subnet -- what a deployed controller could actually know.
        self.max_i_window = 1008  # ~7 days at 10 min; adjust for a different step
        self.Max_i_subnet = {}
        unique_subnets = self.pivoted['id_institution_subnet'].unique()
        for sub_id in unique_subnets:
            sub_df = (self.pivoted[self.pivoted['id_institution_subnet'] == sub_id]
                      .set_index('ds').sort_index())
            rolling_max_i = sub_df[self.slice_names].rolling(
                window=self.max_i_window, min_periods=1
            ).quantile(0.95)
            # Shift by one step: at step t we only know peaks up to t-1.
            rolling_max_i = rolling_max_i.shift(1).fillna(sub_df[self.slice_names].iloc[0])
            self.Max_i_subnet[sub_id] = rolling_max_i

        # Stable station ordering for the one-hot RAN identity in the state. It is
        # passed in explicitly so the train and test environments agree on the
        # layout even when a split does not contain every station.
        self.unique_subnet_ids = sorted(int(s) for s in (ran_ids if ran_ids is not None else unique_subnets))
        self.subnet_id_to_idx = {sid: i for i, sid in enumerate(self.unique_subnet_ids)}

        # Default to the passthrough (oracle) predictor when none is given.
        if predictor is None:
            self.predictor = PassthroughTrafficPredictor()
            self.predictor.fit(self.pivoted)
        else:
            self.predictor = predictor

        # Precompute traffic predictions (with optional sliding context).
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

    def _station_max_i(self, subnet_id: int, ts) -> Dict[str, float]:
        """Causal Max_i peaks for one station at time ``ts``."""
        series = self.Max_i_subnet.get(subnet_id)
        if series is not None and ts in series.index:
            return series.loc[ts].to_dict()
        # Fallback if the timestamp is not found (should not happen in normal use).
        return next(iter(self.Max_i_subnet.values())).iloc[0].to_dict()

    def _get_state(self, step_idx: int) -> np.ndarray:
        row_pred = self.pivoted_pred.iloc[step_idx]
        row_real = self.pivoted.iloc[step_idx]
        subnet_id = int(row_real['id_institution_subnet'])
        station_max_i = self._station_max_i(subnet_id, row_real['ds'])

        # Traffic features are scaled by the station's own causal peak so that
        # stations of very different volumes stay on a comparable scale.
        pred_traffic_vector = [
            float(row_pred.get(f'pred_{s}', row_pred.get(s, 0.0))) / max(station_max_i.get(s, 1.0), 1.0)
            for s in self.slice_names
        ]
        ran_onehot = [0.0] * len(self.unique_subnet_ids)
        ran_idx = self.subnet_id_to_idx.get(subnet_id)
        if ran_idx is not None:
            ran_onehot[ran_idx] = 1.0

        state = [self.past_f_b / 2000.0, self.past_eta_b] + pred_traffic_vector + ran_onehot
        return np.array(state, dtype=np.float32)

    def step_controller(self, raw_action_dict: Dict[str, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        row_real = self.pivoted.iloc[self.current_step_idx]
        ts = row_real['ds']
        subnet_id = int(row_real['id_institution_subnet'])

        l_real = {s: float(row_real[s]) for s in self.slice_names}
        row_pred = self.pivoted_pred.iloc[self.current_step_idx]
        l_pred = {s: float(row_pred.get(f'pred_{s}', row_pred.get(s, 0.0))) for s in self.slice_names}

        station_max_i = self._station_max_i(subnet_id, ts)

        # ---------------------------------------------------------------------
        # Algorithm 1: PPO decision c_init (Eco1 always on).
        # ---------------------------------------------------------------------
        c_init = {s: raw_action_dict.get(s, 1) for s in self.slice_names}
        c_init['Eco1'] = 1

        # ---------------------------------------------------------------------
        # Algorithm 2: SDN 1 threshold filter (alpha_seuil * Max_{i,b}).
        # ---------------------------------------------------------------------
        # The filter reads the forecast, not the realised load: the controller
        # must decide before the traffic of the step has arrived.
        c_filtre = {}
        V_rediriger = {}

        for s in self.slice_names:
            max_i_s = station_max_i.get(s, 1.0)
            threshold = self.alpha_seuil * max_i_s
            if c_init[s] == 0 or l_pred[s] <= threshold or max_i_s <= 0:
                c_filtre[s] = 0
                V_rediriger[s] = l_real[s]
            else:
                c_filtre[s] = 1

        c_filtre['Eco1'] = 1

        # URLLC protection: keep URLLC up whenever it actually carries traffic,
        # applied after the filter so the threshold cannot switch it back off.
        if l_real.get('URLLC', 0.0) > 10.0 and c_filtre.get('URLLC', 1) == 0:
            c_filtre['URLLC'] = 1
            V_rediriger.pop('URLLC', None)

        # ---------------------------------------------------------------------
        # Algorithm 3: routing and overflow (SDN controller 2).
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
        # Algorithm 4: safety and reallocation (SDN controller 2).
        # ---------------------------------------------------------------------
        c_final = c_filtre.copy()
        c_final['Eco1'] = 1
        c_final['Eco2'] = c_eco2

        # Emergency re-activation of the heaviest flow on overload.
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

        # RAN physical engine.
        res_step = self.ran_sim.step(ts, subnet_id, l_real, c_final)
        f_b_t = res_step['f_b']
        eta_b_t = res_step['eta_b']

        # All-active baseline, used to compute Delta E(t).
        all_active = {s: 1 for s in self.slice_names}
        all_active['Eco1'] = 1
        all_active['Eco2'] = 0
        res_base = self.ran_sim.step(ts, subnet_id, l_real, all_active)
        f_b_base = res_base['f_b']

        # Energy gain Delta E(t).
        delta_E_t = max(0.0, (f_b_base - f_b_t) / f_b_base) if f_b_base > 0 else 0.0

        # Loss rate L(t).
        total_eco_traffic = l_eco1_b + l_eco2_b
        if surcharge and total_eco_traffic > 0:
            L_t = max(0.0, (total_eco_traffic - (self.capacity_eco1 + self.capacity_eco2)) / total_eco_traffic)
        else:
            L_t = 0.0

        qos_violation = (eta_b_t < 1.0) or (L_t > 0.0)

        # PPO reward r_t. The energy term is scaled because f_b is in watts
        # (order 1e3), so an unscaled 1/f_b would be negligible next to the QoS
        # and loss terms and the agent would have no incentive to save energy.
        reward = (self.energy_reward_scale / f_b_t) + (self.beta * eta_b_t) - (self.lambda_loss * L_t)

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