"""
Package des modèles de prédiction du trafic réseau.
"""

from src.models.base_predictor import BaseTrafficPredictor
from src.models.passthrough_predictor import PassthroughTrafficPredictor
from src.models.ridge_predictor import MLTrafficPredictor
from src.models.lightgbm_predictor import LightGBMTrafficPredictor
from src.models.lstm_predictor import LSTMTrafficPredictor
from src.models.nhits_predictor import NHiTSTrafficPredictor
from src.models.prophet_predictor import ProphetTrafficPredictor
from src.models.predictor_factory import get_traffic_predictor, AVAILABLE_MODELS
