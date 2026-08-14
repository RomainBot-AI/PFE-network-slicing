#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/models/predictor_factory.py
 OBJET  : Usine à Modèles (Factory Pattern) pour l'Instanciation des Prédicteurs de Trafic
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce module centralise la création et le dispatching des prédicteurs de trafic.
Il est invoqué par l'orchestrateur du projet (`src/pipeline/trainer_evaluator.py`) pour instancier
dynamiquement le modèle choisi via l'argument CLI `--model` (`passthrough`, `lightgbm`, `lstm`, `nhits`, `prophet`).
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
    Instancie le prédicteur de trafic correspondant au nom fourni.

    :param model_name: Clé du modèle ('passthrough', 'lightgbm', 'lstm', 'nhits', 'prophet', 'ridge').
    :param kwargs: Arguments optionnels passés au constructeur du modèle.
    :return: Instance d'une sous-classe de BaseTrafficPredictor.
    :raises ValueError: Si la clé du modèle n'est pas reconnue.
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
