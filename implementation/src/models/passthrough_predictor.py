#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/models/passthrough_predictor.py
 OBJET  : Prédicteur Passthrough / Oracle (Prédiction Proactive sur 1 Heure)
====================================================================================================

DESCRIPTION DÉTAILLÉE :
-----------------------
Définit pred_{slice} = max(real_{slice}^{t+1..t+6}).
Mode Oracle à prédiction parfaite sur la prochaine heure (6 pas de 10 min).

====================================================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from src.models.base_predictor import BaseTrafficPredictor


class PassthroughTrafficPredictor(BaseTrafficPredictor):
    """
    Prédicteur Passthrough / Oracle Proactif (Horizon 1 heure = 6 pas).
    """

    def __init__(self, horizon: int = 6):
        super().__init__()
        self.horizon = horizon

    def fit(self, df_train_pivoted: pd.DataFrame):
        self.slice_names = [c for c in df_train_pivoted.columns if c not in ['ds', 'id_institution_subnet']]

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        df_res = df_pivoted.copy().reset_index(drop=True)
        if not self.slice_names:
            self.slice_names = [c for c in df_pivoted.columns if c not in ['ds', 'id_institution_subnet']]

        has_st = 'id_institution_subnet' in df_pivoted.columns
        stations = df_pivoted['id_institution_subnet'].unique() if has_st else [None]

        for slice_name in self.slice_names:
            for st in stations:
                if st is not None:
                    idx_st = df_res[df_res['id_institution_subnet'] == st].index
                    series = df_pivoted.loc[idx_st, slice_name].values
                else:
                    idx_st = df_res.index
                    series = df_pivoted[slice_name].values

                preds = []
                n = len(series)
                for i in range(n):
                    future_win = series[i : i + self.horizon]
                    preds.append(float(np.max(future_win)) if len(future_win) > 0 else float(series[i]))

                df_res.loc[idx_st, f'pred_{slice_name}'] = preds

        return df_res

    def evaluate(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        return {'MAE': 0.0, 'RMSE': 0.0, 'NMAE': 0.0}
