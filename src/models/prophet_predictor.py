#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
====================================================================================================
 MODULE : src/models/prophet_predictor.py
 OBJET  : Prédicteur de Trafic Réseau basé sur Meta Prophet (Time Series Model)
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce module implémente le prédicteur basé sur Meta Prophet (`ProphetTrafficPredictor`).
Il s'insère dans l'usine à modèles (`src/models/predictor_factory.py`) et fournit la prédiction
de trafic futur \hat{l}^{t+1} au contrôleur SDN.

CARACTÉRISTIQUES CLÉS & CONFIGURATION :
----------------------------------------
  1. Décomposition Additive : Modélise la tendance et les saisonnalités journalières et hebdomadaires.
  2. Option `growth='flat'` : Empêche l'extrapolation linéaire de tendance à long terme qui provoquerait
     des dérives explosives sur 7 mois d'écart temporal.
  3. Modèle Séparé par (tranche, station) : Entraîné indépendamment pour chaque sous-réseau afin de capter
     les rythmes de trafic locaux sans contamination.
====================================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from prophet import Prophet
import warnings

from src.models.base_predictor import BaseTrafficPredictor

# Ignorer les avertissements de conciliation Stan/Prophet dans les logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ProphetTrafficPredictor(BaseTrafficPredictor):
    """
    Prédicteur de trafic 5G basé sur Meta Prophet.
    """

    def __init__(self, horizon: int = 6, max_train_samples: int = 2016):
        """
        :param horizon: Horizon de prédiction (6 pas de 10 min = 1 heure).
        :param max_train_samples: Nombre maximal de pas d'historique conservés pour l'ajustement (2016 pas = 14 jours).
        """
        super().__init__()
        self.horizon = horizon
        self.max_train_samples = max_train_samples
        self.models: Dict[Tuple[str, int], Prophet] = {}

    def fit(self, df_train_pivoted: pd.DataFrame) -> None:
        """
        Ajuste un modèle Meta Prophet indépendant par couple (tranche, station Macro-RAN).
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

                if len(df_prophet) < 2:
                    self.models[(slice_name, st)] = None
                    continue

                # Modèle Prophet à croissance plate (growth='flat') pour éviter toute dérive de tendance
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
                except Exception:
                    self.models[(slice_name, st)] = None

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Génère la prédiction `pred_<slice_name>` pour chaque pas de temps via Prophet.
        """
        df_res = df_pivoted.copy().reset_index(drop=True)
        self.slice_names = [c for c in df_res.columns if c not in ['ds', 'id_institution_subnet']]
        has_ds = 'ds' in df_res.columns

        if not has_ds:
            df_res['ds'] = pd.date_range(start='2024-01-01', periods=len(df_res), freq='10min')

        stations = df_res['id_institution_subnet'].unique() if 'id_institution_subnet' in df_res.columns else [0]

        for slice_name in self.slice_names:
            col_pred_name = f'pred_{slice_name}'
            df_res[col_pred_name] = df_res[slice_name].values  # Valeur par défaut si modèle indisponible

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

                        df_res.loc[mask, col_pred_name] = pred_vals
                    except Exception:
                        pass

        return df_res