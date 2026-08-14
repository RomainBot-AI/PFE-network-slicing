#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/models/ridge_predictor.py
 OBJET  : Prédicteur Supervisé Multi-Échelle par Régression Ridge
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce module implémente le modèle de baseline de régression linéaire régularisée (Ridge).
Il s'insère dans l'usine à prédicteurs (`src/models/predictor_factory.py`) comme baseline de régression
linaire supervisée à faibles coûts de calcul.

CARACTÉRISTIQUES CLÉS :
-----------------------
  - Features multi-échelles (30 min, 24h, 7 jours).
  - Entraînement séparé par couple (tranche, station Macro-RAN) avec régularisation L2 ($\alpha=1.0$).
====================================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
from src.models.base_predictor import BaseTrafficPredictor


class MLTrafficPredictor(BaseTrafficPredictor):
    """
    Prédicteur de trafic multi-échelle basé sur la Régression Ridge.
    """

    def __init__(self, horizon: int = 6, short_lags: int = 3):
        """
        :param horizon: Horizon de prédiction (6 pas de 10 min = 1 heure).
        :param short_lags: Nombre de lags récents à inclure (3 pas = 30 min).
        """
        super().__init__()
        self.horizon = horizon
        self.short_lags = short_lags
        self.lag_24h = 144
        self.lag_7d = 1008
        self.models: Dict[Tuple[str, Optional[int]], Ridge] = {}

    def extract_features_for_series(
        self,
        series: np.ndarray,
        is_train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construit la matrice de features X et le vecteur cible y pour la régression Ridge.

        :param series: Série brute.
        :param is_train: Flag d'apprentissage ou d'inférence.
        :return: Tuple (matrice X, vecteur y).
        """
        n = len(series)
        X_list, y_list = [], []

        min_idx = self.short_lags + 1
        max_idx = n

        for i in range(min_idx, max_idx):
            ref = i - 1  # Alignement causal strict (passé i-1)
            feat = []

            # 1. Court terme (30 min)
            for l in range(self.short_lags):
                idx_lag = ref - l
                feat.append(float(series[idx_lag]) if idx_lag >= 0 else float(series[ref]))

            # 2. Moyen terme (24h)
            idx_24h = ref - self.lag_24h
            feat.append(float(series[idx_24h]) if idx_24h >= 0 else float(series[ref]))

            start_24h = max(0, ref - self.lag_24h + 1)
            window_24h = series[start_24h: ref + 1]
            feat.append(float(np.mean(window_24h)) if len(window_24h) > 0 else float(series[ref]))

            # 3. Long terme (7 jours)
            idx_7d = ref - self.lag_7d
            if idx_7d >= 0:
                feat.append(float(series[idx_7d]))
            elif idx_24h >= 0:
                feat.append(float(series[idx_24h]))
            else:
                feat.append(float(series[ref]))

            X_list.append(feat)

            if is_train:
                y_list.append(float(series[i]))

        X_arr = np.array(X_list, dtype=np.float32) if len(X_list) > 0 else np.empty((0, 6), dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32) if len(y_list) > 0 else np.empty((0,), dtype=np.float32)

        return X_arr, y_arr

    def fit(self, df_train_pivoted: pd.DataFrame) -> None:
        """
        Ajuste un modèle Ridge indépendant par couple (tranche, station Macro-RAN).
        """
        df_piv = df_train_pivoted.copy().reset_index(drop=True)
        self.slice_names = [c for c in df_piv.columns if c not in ['ds', 'id_institution_subnet']]
        has_st = 'id_institution_subnet' in df_piv.columns
        stations = df_piv['id_institution_subnet'].unique() if has_st else [None]

        for slice_name in self.slice_names:
            for st in stations:
                if st is not None:
                    df_st = df_piv[df_piv['id_institution_subnet'] == st]
                else:
                    df_st = df_piv

                series_st = df_st[slice_name].values
                X_st, y_st = self.extract_features_for_series(series_st, is_train=True)
                if len(X_st) < 2:
                    continue

                model = Ridge(alpha=1.0)
                model.fit(X_st, y_st)
                self.models[(slice_name, st)] = model

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Génère la prédiction de trafic par régression Ridge.
        """
        df_piv = df_pivoted.copy().reset_index(drop=True)
        df_res = df_piv.copy()

        if not self.slice_names:
            self.slice_names = [c for c in df_piv.columns if c not in ['ds', 'id_institution_subnet']]

        has_st = 'id_institution_subnet' in df_piv.columns
        stations = df_piv['id_institution_subnet'].unique() if has_st else [None]
        df_ctx_clean = df_context.copy().reset_index(drop=True) if df_context is not None else None

        for slice_name in self.slice_names:
            for st in stations:
                if st is not None:
                    idx_st = df_res[df_res['id_institution_subnet'] == st].index
                    df_st = df_piv.loc[idx_st]
                    df_ctx_st = df_ctx_clean[df_ctx_clean['id_institution_subnet'] == st] if df_ctx_clean is not None and has_st else None
                else:
                    idx_st = df_res.index
                    df_st = df_piv
                    df_ctx_st = df_ctx_clean

                model = self.models.get((slice_name, st))
                if model is None:
                    df_res.loc[idx_st, f'pred_{slice_name}'] = df_st[slice_name]
                    continue

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