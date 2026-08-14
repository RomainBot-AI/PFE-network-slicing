#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
====================================================================================================
 MODULE : src/models/lightgbm_predictor.py
 OBJET  : Prédicteur Multi-Échelle de Trafic par LightGBM (Gradient Boosting)
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce module implémente le modèle prédicteur basé sur LightGBM (Gradient Boosted Decision Trees).
Il s'insère dans le pipeline d'inférence (`src/models/predictor_factory.py`) et fournit au
contrôleur SDN (`src/environment/sdn_controller_env.py`) la prédiction de trafic futur \hat{l}^{t+1}.

CARACTÉRISTIQUES CLÉS & INGÉNIERIE DE FEATURES :
------------------------------------------------
  1. Combinaison de 3 échelles temporelles :
     - Court terme (30 min)  : Lags t, t-1, t-2.
     - Moyen terme (24h)     : Lag t-144, Moyenne glissante 24h, Max glissant 24h.
     - Long terme (7 jours)  : Lag t-1008 (saisonnalité hebdomadaire).
     - Features calendaires  : Heure du jour (0-23), jour de la semaine (0-6), indicateur week-end.

  2. Entraînement Indépendant par couple (tranche, station) :
     Un modèle LightGBM distinct est entraîné pour chaque station Macro-RAN afin d'éviter
     toute contamination croisée entre stations à profils de trafic divergents (ex: station rurale vs urbaine).
====================================================================================================
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from src.models.base_predictor import BaseTrafficPredictor


class LightGBMTrafficPredictor(BaseTrafficPredictor):
    """
    Prédicteur de trafic multi-échelle basé sur LightGBM.
    """

    def __init__(self, horizon: int = 6, short_lags: int = 3):
        """
        :param horizon: Horizon de prédiction (6 pas de 10 min = 1 heure).
        :param short_lags: Nombre de lags récents à inclure (3 pas = 30 minutes).
        """
        super().__init__()
        self.horizon = horizon
        self.short_lags = short_lags
        self.lag_24h = 144  # 24 heures (144 pas de 10 min)
        self.lag_7d = 1008  # 7 jours (1008 pas de 10 min)
        self.models: Dict[Tuple[str, Optional[int]], lgb.LGBMRegressor] = {}

    def _extract_calendar_features(self, timestamp) -> List[float]:
        """
        Extrait les caractéristiques temporelles [heure, jour_semaine, est_weekend] du timestamp.
        """
        try:
            ts = pd.to_datetime(timestamp)
            return [float(ts.hour), float(ts.dayofweek), 1.0 if ts.dayofweek >= 5 else 0.0]
        except Exception:
            return [0.0, 0.0, 0.0]

    def extract_features_for_series(
        self,
        series: np.ndarray,
        ds_values: Optional[np.ndarray] = None,
        is_train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construit la matrice de features X et le vecteur cible y pour une série temporelle donnée,
        en veillant à l'alignement temporel strict (prédiction de la ligne i faite à partir du passé i-1).

        :param series: Tableau 1D des valeurs de trafic.
        :param ds_values: Tableau 1D des timestamps.
        :param is_train: Flag indiquant la phase d'apprentissage ou d'inférence.
        :return: Tuple (matrice X, vecteur y).
        """
        n = len(series)
        X_list, y_list = [], []

        min_idx = self.short_lags + 1
        max_idx = n

        for i in range(min_idx, max_idx):
            ref = i - 1  # Référence stricte au passé
            feat = []

            # 1. Échelle Court Terme (30 dernières minutes : ref, ref-1, ref-2)
            for l in range(self.short_lags):
                idx_lag = ref - l
                feat.append(float(series[idx_lag]) if idx_lag >= 0 else float(series[ref]))

            # 2. Échelle Moyen Terme (24 heures)
            idx_24h = ref - self.lag_24h
            feat.append(float(series[idx_24h]) if idx_24h >= 0 else float(series[ref]))

            # Statistiques glissantes 24h (Moyenne & Max)
            start_24h = max(0, ref - self.lag_24h + 1)
            window_24h = series[start_24h: ref + 1]
            feat.append(float(np.mean(window_24h)) if len(window_24h) > 0 else float(series[ref]))
            feat.append(float(np.max(window_24h)) if len(window_24h) > 0 else float(series[ref]))

            # 3. Échelle Long Terme (7 jours)
            idx_7d = ref - self.lag_7d
            if idx_7d >= 0:
                feat.append(float(series[idx_7d]))
            elif idx_24h >= 0:
                feat.append(float(series[idx_24h]))
            else:
                feat.append(float(series[ref]))

            # 4. Features Calendaires associées au pas cible i
            if ds_values is not None and i < len(ds_values):
                cal_f = self._extract_calendar_features(ds_values[i])
                feat.extend(cal_f)
            else:
                feat.extend([0.0, 0.0, 0.0])

            X_list.append(feat)

            if is_train:
                y_list.append(float(series[i]))

        X_arr = np.array(X_list, dtype=np.float32) if len(X_list) > 0 else np.empty((0, 11), dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32) if len(y_list) > 0 else np.empty((0,), dtype=np.float32)

        return X_arr, y_arr

    def fit(self, df_train_pivoted: pd.DataFrame) -> None:
        """
        Entraîne un modèle LightGBM distinct par couple (tranche, station Macro-RAN).
        """
        df_piv = df_train_pivoted.copy().reset_index(drop=True)
        self.slice_names = [c for c in df_piv.columns if c not in ['ds', 'id_institution_subnet']]
        has_ds = 'ds' in df_piv.columns

        stations = df_piv['id_institution_subnet'].unique() if 'id_institution_subnet' in df_piv.columns else [None]

        for slice_name in self.slice_names:
            for st in stations:
                if st is not None:
                    df_st = df_piv[df_piv['id_institution_subnet'] == st]
                else:
                    df_st = df_piv

                series_st = df_st[slice_name].values
                ds_st = df_st['ds'].values if has_ds else None

                X_st, y_st = self.extract_features_for_series(series_st, ds_values=ds_st, is_train=True)
                if len(X_st) < 2:
                    continue

                min_child = min(15, len(X_st))
                model = lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.08,
                    num_leaves=63,
                    min_child_samples=min_child,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    n_jobs=-1,
                    verbose=-1,
                    importance_type='gain'
                )
                model.fit(X_st, y_st)
                self.models[(slice_name, st)] = model

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Génère les prédictions de trafic futures `pred_<slice_name>` pour chaque pas de temps.
        """
        df_piv = df_pivoted.copy().reset_index(drop=True)
        df_res = df_piv.copy()

        if not self.slice_names:
            self.slice_names = [c for c in df_piv.columns if c not in ['ds', 'id_institution_subnet']]

        has_ds = 'ds' in df_piv.columns
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
                ds_vals = df_st['ds'].values if has_ds else None

                if df_ctx_st is not None and slice_name in df_ctx_st.columns:
                    ctx_series = df_ctx_st[slice_name].values
                    full_series = np.concatenate([ctx_series, series])
                    offset = len(ctx_series)

                    if has_ds and 'ds' in df_ctx_st.columns:
                        full_ds = np.concatenate([df_ctx_st['ds'].values, ds_vals])
                    else:
                        full_ds = ds_vals
                else:
                    full_series = series
                    offset = 0
                    full_ds = ds_vals

                X_pred, _ = self.extract_features_for_series(full_series, ds_values=full_ds, is_train=False)

                if len(X_pred) > 0:
                    raw_preds = model.predict(X_pred)
                    preds_target = raw_preds[offset:] if offset > 0 else raw_preds

                    if len(preds_target) < len(idx_st):
                        pad_len = len(idx_st) - len(preds_target)
                        first_val = preds_target[0] if len(preds_target) > 0 else series[0]
                        preds_target = np.concatenate([np.full(pad_len, first_val), preds_target])

                    # Tronquage au seuil 0.0 pour éviter les prédictions de trafic négatives
                    df_res.loc[idx_st, f'pred_{slice_name}'] = np.maximum(0.0, preds_target[:len(idx_st)])
                else:
                    df_res.loc[idx_st, f'pred_{slice_name}'] = series

        return df_res