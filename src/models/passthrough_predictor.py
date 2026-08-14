#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
====================================================================================================
 MODULE : src/models/passthrough_predictor.py
 OBJET  : Prédicteur Passthrough — Allocation Dynamique Réactive
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce module implémente la baseline Passthrough (Oracle à information parfaite).
Dans la simulation, elle assigne la valeur du trafic réel instantané à la prédiction (`pred = real`).

Pourquoi cette baseline est-elle essentielle ?
  1. Elle sert de borne supérieure théorique (Upper Bound) : elle montre le score d'économie et de QoS
     qu'obtiendrait le système si les prédictions futures étaient 100% parfaites ($NMAE = 0.00\%$).
  2. Elle permet d'illustrer la différence fondamentale entre une réallocation réactive (qui arrive
     "trop tard" dans un vrai réseau mobile réel) et une réservation proactive par IA.
====================================================================================================
"""

import pandas as pd
from typing import Optional, Dict
from src.models.base_predictor import BaseTrafficPredictor


class PassthroughTrafficPredictor(BaseTrafficPredictor):
    """
    Baseline Oracle / Allocation Réactive : \hat{l}^{t+1} = l^{t+1}.
    Transmet directement la valeur réelle du trafic sans erreur de régression.
    """

    def __init__(self):
        super().__init__()

    def fit(self, df_train_pivoted: pd.DataFrame) -> None:
        """
        Stocke la liste des tranches représentées dans le DataFrame d'entraînement.
        """
        self.slice_names = [c for c in df_train_pivoted.columns if c not in ['ds', 'id_institution_subnet']]

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Génère les colonnes de prédiction en copiant exactement la charge réelle instantanée.

        :param df_pivoted: DataFrame pivoté d'entrée.
        :param df_context: Non utilisé pour Passthrough.
        :return: DataFrame copie avec colonnes `pred_<slice_name>`.
        """
        df_res = df_pivoted.copy().reset_index(drop=True)
        if not self.slice_names:
            self.slice_names = [c for c in df_pivoted.columns if c not in ['ds', 'id_institution_subnet']]

        # Assignation directe : pred = real (information parfaite / Oracle)
        for slice_name in self.slice_names:
            df_res[f'pred_{slice_name}'] = df_res[slice_name].values.copy()

        return df_res

    def evaluate(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Retourne des erreurs nulles par construction (NMAE = 0.00%).
        """
        return {'MAE': 0.0, 'RMSE': 0.0, 'NMAE': 0.0}
