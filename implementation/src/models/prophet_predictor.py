#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/models/prophet_predictor.py
 OBJET  : Prédicteur de Trafic basé sur Prophet (Meta Time Series Model)
====================================================================================================

DESCRIPTION DÉTAILLÉE :
-----------------------
Ce module implémente un prédicteur de trafic basé sur Meta Prophet.
Prophet modélise les tendances et la saisonnalité (journalière et hebdomadaire).
Entraîne un modèle Prophet séparé par station/subnet et par slice pour une échelle exacte.

====================================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from prophet import Prophet
import warnings

from src.models.base_predictor import BaseTrafficPredictor

# Supprimer les warnings de Prophet/Stan pendant la simulation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ProphetTrafficPredictor(BaseTrafficPredictor):
    """
    Prédicteur de Trafic 5G basé sur Meta Prophet.
    Prédit le trafic maximal anticipé sur l'horizon de 1 heure (6 pas de 10 min) station par station.
    """

    def __init__(self, horizon: int = 6, max_train_samples: int = 2016):
        super().__init__()
        self.horizon = horizon
        self.max_train_samples = max_train_samples  # 2 semaines max par station pour rapidité
        self.models: Dict[Tuple[str, int], Prophet] = {}

    def fit(self, df_train_pivoted: pd.DataFrame):
        """
        Ajuste un modèle Prophet indépendant pour chaque tranche réseau (slice) et chaque station.
        """
        df_piv = df_train_pivoted.copy().reset_index(drop=True)
        self.slice_names = [c for c in df_piv.columns if c not in ['ds', 'id_institution_subnet']]
        has_ds = 'ds' in df_piv.columns

        if not has_ds:
            df_piv['ds'] = pd.date_range(start='2024-01-01', periods=len(df_piv), freq='10min')

        stations = df_piv['id_institution_subnet'].unique() if 'id_institution_subnet' in df_piv.columns else [0]

        for slice_name in self.slice_names:
            for st in stations:
                if 'id_institution_subnet' in df_piv.columns:
                    df_st = df_piv[df_piv['id_institution_subnet'] == st].copy()
                else:
                    df_st = df_piv.copy()

                df_st = df_st.sort_values(by='ds').tail(self.max_train_samples).reset_index(drop=True)

                ds_series = pd.to_datetime(df_st['ds'])
                if ds_series.dt.tz is not None:
                    ds_series = ds_series.dt.tz_localize(None)

                df_prophet = pd.DataFrame({
                    'ds': ds_series,
                    'y': df_st[slice_name].values
                })

                m = Prophet(
                    growth='flat',
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    interval_width=0.80
                )
                
                try:
                    m.fit(df_prophet)
                    self.models[(slice_name, st)] = m
                except Exception as e:
                    self.models[(slice_name, st)] = None

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Génère les prédictions de trafic anticipées station par station.
        """
        df_res = df_pivoted.copy().reset_index(drop=True)
        self.slice_names = [c for c in df_res.columns if c not in ['ds', 'id_institution_subnet']]
        has_ds = 'ds' in df_res.columns

        if not has_ds:
            df_res['ds'] = pd.date_range(start='2024-01-01', periods=len(df_res), freq='10min')

        stations = df_res['id_institution_subnet'].unique() if 'id_institution_subnet' in df_res.columns else [0]

        for slice_name in self.slice_names:
            col_pred_name = f'pred_{slice_name}'
            df_res[col_pred_name] = df_res[slice_name].values  # Valeur par défaut

            for st in stations:
                m = self.models.get((slice_name, st), None)
                if m is not None:
                    try:
                        mask = (df_res['id_institution_subnet'] == st) if 'id_institution_subnet' in df_res.columns else np.ones(len(df_res), dtype=bool)
                        df_st = df_res[mask]

                        ds_series = pd.to_datetime(df_st['ds'])
                        if ds_series.dt.tz is not None:
                            ds_series = ds_series.dt.tz_localize(None)

                        df_ds = pd.DataFrame({'ds': ds_series})
                        forecast = m.predict(df_ds)
                        pred_vals = np.maximum(0.0, forecast['yhat'].values)
                        pred_smooth = pd.Series(pred_vals, index=df_st.index).rolling(self.horizon, min_periods=1).max().values

                        df_res.loc[mask, col_pred_name] = pred_smooth
                    except Exception:
                        pass

        return df_res
