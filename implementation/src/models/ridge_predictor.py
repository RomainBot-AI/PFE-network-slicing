#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/models/ridge_predictor.py
 OBJET  : Prédicteur Supervisé Multi-Échelle par Régression Ridge
====================================================================================================

DESCRIPTION DÉTAILLÉE :
-----------------------
Modèle Ridge basé sur 3 échelles temporelles (court terme 30m, 24h, 7j) prédisant la demande
de la prochaine heure (6 pas de 10 min).

====================================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
from src.models.base_predictor import BaseTrafficPredictor


class MLTrafficPredictor(BaseTrafficPredictor):
    """
    Prédicteur de Trafic Multi-Échelle basé sur la Régression Ridge.
    """

    def __init__(self, horizon: int = 6, short_lags: int = 3):
        super().__init__()
        self.horizon = horizon
        self.short_lags = short_lags
        self.lag_24h = 144
        self.lag_7d = 1008
        self.models: Dict[str, Ridge] = {}

    def extract_features_for_series(
        self,
        series: np.ndarray,
        is_train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(series)
        X_list, y_list = [], []

        min_idx = self.short_lags
        max_idx = (n - self.horizon) if is_train else n

        for i in range(min_idx, max_idx):
            feat = []

            # 1. Court terme (30m)
            for l in range(self.short_lags):
                idx_lag = i - l
                feat.append(float(series[idx_lag]) if idx_lag >= 0 else float(series[i]))

            # 2. 24h
            idx_24h = i - self.lag_24h
            feat.append(float(series[idx_24h]) if idx_24h >= 0 else float(series[i]))

            start_24h = max(0, i - self.lag_24h + 1)
            window_24h = series[start_24h : i + 1]
            feat.append(float(np.mean(window_24h)) if len(window_24h) > 0 else float(series[i]))

            # 3. 7j
            idx_7d = i - self.lag_7d
            if idx_7d >= 0:
                feat.append(float(series[idx_7d]))
            elif idx_24h >= 0:
                feat.append(float(series[idx_24h]))
            else:
                feat.append(float(series[i]))

            X_list.append(feat)

            if is_train:
                future_window = series[i + 1 : i + 1 + self.horizon]
                target_val = float(np.max(future_window)) if len(future_window) > 0 else float(series[i])
                y_list.append(target_val)

        X_arr = np.array(X_list, dtype=np.float32) if len(X_list) > 0 else np.empty((0, 6), dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32) if len(y_list) > 0 else np.empty((0,), dtype=np.float32)

        return X_arr, y_arr

    def fit(self, df_train_pivoted: pd.DataFrame):
        df_piv = df_train_pivoted.copy().reset_index(drop=True)
        self.slice_names = [c for c in df_piv.columns if c not in ['ds', 'id_institution_subnet']]
        has_st = 'id_institution_subnet' in df_piv.columns
        stations = df_piv['id_institution_subnet'].unique() if has_st else [None]

        for slice_name in self.slice_names:
            X_all_stations, y_all_stations = [], []

            for st in stations:
                if st is not None:
                    df_st = df_piv[df_piv['id_institution_subnet'] == st]
                else:
                    df_st = df_piv

                series_st = df_st[slice_name].values
                X_st, y_st = self.extract_features_for_series(series_st, is_train=True)
                if len(X_st) > 0:
                    X_all_stations.append(X_st)
                    y_all_stations.append(y_st)

            if len(X_all_stations) > 0:
                X_train = np.vstack(X_all_stations)
                y_train = np.concatenate(y_all_stations)

                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                self.models[slice_name] = model

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        df_piv = df_pivoted.copy().reset_index(drop=True)
        df_res = df_piv.copy()

        if not self.slice_names:
            self.slice_names = [c for c in df_piv.columns if c not in ['ds', 'id_institution_subnet']]

        has_st = 'id_institution_subnet' in df_piv.columns
        stations = df_piv['id_institution_subnet'].unique() if has_st else [None]
        df_ctx_clean = df_context.copy().reset_index(drop=True) if df_context is not None else None

        for slice_name in self.slice_names:
            if slice_name not in self.models:
                df_res[f'pred_{slice_name}'] = df_res[slice_name]
                continue

            model = self.models[slice_name]

            for st in stations:
                if st is not None:
                    idx_st = df_res[df_res['id_institution_subnet'] == st].index
                    df_st = df_piv.loc[idx_st]
                    df_ctx_st = df_ctx_clean[df_ctx_clean['id_institution_subnet'] == st] if df_ctx_clean is not None and has_st else None
                else:
                    idx_st = df_res.index
                    df_st = df_piv
                    df_ctx_st = df_ctx_clean

                series = df_st[slice_name].values

                if df_ctx_st is not None and slice_name in df_ctx_st.columns:
                    ctx_series = df_ctx_st[slice_name].values
                    full_series = np.concatenate([ctx_series, series])
                    offset = len(ctx_series)
                else:
                    full_series = series
                    offset = 0

                X_pred, _ = self.extract_features_for_series(full_series, is_train=False)

                if len(X_pred) > 0:
                    raw_preds = model.predict(X_pred)
                    preds_target = raw_preds[offset:] if offset > 0 else raw_preds

                    if len(preds_target) < len(idx_st):
                        pad_len = len(idx_st) - len(preds_target)
                        first_val = preds_target[0] if len(preds_target) > 0 else series[0]
                        preds_target = np.concatenate([np.full(pad_len, first_val), preds_target])

                    df_res.loc[idx_st, f'pred_{slice_name}'] = np.maximum(0.0, preds_target[:len(idx_st)])
                else:
                    df_res.loc[idx_st, f'pred_{slice_name}'] = series

        return df_res
