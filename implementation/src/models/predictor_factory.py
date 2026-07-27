#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/models/predictor_factory.py
 OBJET  : Usine à Modèles (Factory Pattern) pour l'Instanciation des Prédicteurs
====================================================================================================

DESCRIPTION DÉTAILLÉE :
-----------------------
Permet d'instancier facilement n'importe quel prédicteur de trafic à l'aide d'une clé de texte :
  - 'passthrough' : Réallocation dynamique pure vs Baseline All-Active
  - 'lightgbm'    : LightGBM Gradient Boosting (features multi-échelles & calendaires)
  - 'lstm'        : Réseau PyTorch LSTM
  - 'nhits'       : Réseau PyTorch N-HiTS
  - 'prophet'     : Modèle Temporel Meta Prophet (tendance & saisonnalité)
  - 'ridge'       : Régression Ridge supervisée
  - 'all'         : Exécute la comparaison de TOUS les modèles !

====================================================================================================
"""

from src.models.base_predictor import BaseTrafficPredictor
from src.models.passthrough_predictor import PassthroughTrafficPredictor
from src.models.ridge_predictor import MLTrafficPredictor
from src.models.lightgbm_predictor import LightGBMTrafficPredictor
from src.models.lstm_predictor import LSTMTrafficPredictor
from src.models.nhits_predictor import NHiTSTrafficPredictor
from src.models.prophet_predictor import ProphetTrafficPredictor

AVAILABLE_MODELS = ['passthrough', 'lightgbm', 'lstm', 'nhits', 'prophet']


def get_traffic_predictor(model_name: str = "passthrough", **kwargs) -> BaseTrafficPredictor:
    """
    Instancie le prédicteur de trafic correspondant.
    """
    key = model_name.lower()
    if key == "passthrough":
        return PassthroughTrafficPredictor()
    elif key in ["lightgbm", "lgbm", "gbm"]:
        return LightGBMTrafficPredictor(**kwargs)
    elif key == "lstm":
        return LSTMTrafficPredictor(**kwargs)
    elif key in ["nhits", "nhitm"]:
        return NHiTSTrafficPredictor(**kwargs)
    elif key in ["prophet", "fbprophet"]:
        return ProphetTrafficPredictor(**kwargs)
    elif key in ["ridge", "ml"]:
        return MLTrafficPredictor(**kwargs)
    else:
        raise ValueError(
            f"Modèle inconnu: '{model_name}'. Modèles disponibles: {AVAILABLE_MODELS} ou 'all'."
        )
