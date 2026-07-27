#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/models/base_predictor.py
 OBJET  : Classe de Base Abstraite pour tous les Prédicteurs de Trafic Réseau
====================================================================================================

DESCRIPTION DÉTAILLÉE :
-----------------------
Définit l'interface standardisée pour les modèles de prédiction de trafic.
Cette architecture abstraite permet d'interchanger en toute transparence :
  1. PassthroughPredictor (Équivalent oracle/réallocation dynamique pure)
  2. RidgePredictor (Régression Ridge supervisée)
  3. LSTMPredictor (Réseau de neurones récurrent PyTorch LSTM)
  4. NHiTSPredictor (Architecture neuronale multi-échelle N-HiTS)

FONCTIONS DE L'INTERFACE :
--------------------------
  - fit(df_train_pivoted) : Entraîne le modèle sur le Train Set sans fuite temporelle.
  - predict_pivoted(df_pivoted, df_context) : Prédit le trafic futur l_hat^{t+1} pas-à-pas.
  - evaluate(df_pivoted, df_context) : Calcule la MAE, la RMSE et la NMAE normalisée (%).

====================================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class BaseTrafficPredictor:
    """
    Interface Abstraite pour les Prédicteurs de Trafic Réseau.
    """
    def __init__(self):
        self.slice_names: List[str] = []

    def fit(self, df_train_pivoted: pd.DataFrame):
        """Entraîne le modèle sur les données pivotées d'entraînement."""
        pass

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Prédit le trafic pour tous les pas de temps dans df_pivoted."""
        raise NotImplementedError("La méthode predict_pivoted doit être implémentée.")

    def evaluate(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Évalue les métriques MAE, RMSE et NMAE (%) entre trafic réel et prédit."""
        df_pred = self.predict_pivoted(df_pivoted, df_context=df_context)
        actuals, preds = [], []

        for slice_name in self.slice_names:
            actuals.extend(df_pred[slice_name].values)
            preds.extend(df_pred[f'pred_{slice_name}'].values)

        actuals = np.array(actuals, dtype=np.float64)
        preds = np.array(preds, dtype=np.float64)

        mae = float(np.mean(np.abs(preds - actuals)))
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
        
        # Normalisation NMAE (%) par rapport à la valeur maximale observée
        max_actual = float(np.max(actuals)) if len(actuals) > 0 and np.max(actuals) > 0 else 1.0
        nmae = (mae / max_actual) * 100.0

        return {'MAE': mae, 'RMSE': rmse, 'NMAE': nmae}
