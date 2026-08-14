#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/models/base_predictor.py
 OBJET  : Classe de Base Abstraite pour tous les Prédicteurs de Trafic Réseau
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Définit l'interface standardisée pour l'ensemble des modèles de prédiction de trafic.
Tous les prédicteurs (Passthrough, Ridge, LightGBM, LSTM, N-HiTS, Prophet) héritent de cette classe.
Cette abstraction garantit :
  1. Une méthode `fit` uniforme pour l'apprentissage sur les données pivotées.
  2. Une méthode `predict_pivoted` produisant un DataFrame enrichi des colonnes `pred_<slice_name>`.
  3. Une méthode `evaluate` calculant de manière homogène les métriques d'erreur MAE, RMSE et NMAE (%).
====================================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class BaseTrafficPredictor:
    """
    Interface abstraite définissant le contrat d'exécution des modèles de prédiction de trafic.
    """
    def __init__(self):
        self.slice_names: List[str] = []

    def fit(self, df_train_pivoted: pd.DataFrame) -> None:
        """
        Entraîne le modèle sur les données pivotées d'entraînement.

        :param df_train_pivoted: DataFrame pivoté contenant l'historique de trafic par tranche.
        """
        pass

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prédit le trafic futur pas-à-pas pour l'ensemble du DataFrame d'entrée.

        :param df_pivoted: DataFrame pivoté contenant le trafic courant.
        :param df_context: Contexte historique optionnel pour les modèles tabulaires à fenêtre glissante.
        :return: DataFrame copie enrichie des colonnes de prédiction `pred_<slice_name>`.
        """
        raise NotImplementedError("La méthode predict_pivoted doit être implémentée.")

    def evaluate(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calcule les métriques d'erreur standardisées : MAE, RMSE et NMAE (%).

        Pourquoi le NMAE (%) ?
        Le Normalized Mean Absolute Error ramène l'erreur MAE en pourcentage par rapport
        au trafic maximal observé (max_actual), ce qui permet de comparer équitablement la précision
        entre des stations ou des tranches de volumes très différents.

        :param df_pivoted: DataFrame de trafic réel.
        :param df_context: Contexte historique glissant.
        :return: Dictionnaire des scores d'erreur {'MAE': ..., 'RMSE': ..., 'NMAE': ...}.
        """
        df_pred = self.predict_pivoted(df_pivoted, df_context=df_context)
        actuals, preds = [], []

        for slice_name in self.slice_names:
            actuals.extend(df_pred[slice_name].values)
            preds.extend(df_pred[f'pred_{slice_name}'].values)

        actuals = np.array(actuals, dtype=np.float64)
        preds = np.array(preds, dtype=np.float64)

        mae = float(np.mean(np.abs(preds - actuals)))
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))

        # Normalisation NMAE (%) par rapport au pic maximal observé
        max_actual = float(np.max(actuals)) if len(actuals) > 0 and np.max(actuals) > 0 else 1.0
        nmae = (mae / max_actual) * 100.0

        return {'MAE': mae, 'RMSE': rmse, 'NMAE': nmae}
