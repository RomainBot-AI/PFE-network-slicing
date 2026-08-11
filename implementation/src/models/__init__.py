"""Traffic-prediction models.

Only light interfaces are re-exported here. Concrete predictors are imported
lazily by :func:`get_traffic_predictor` so that heavy backends (torch, lightgbm,
prophet) are loaded only when actually used.
"""

from src.models.base_predictor import BaseTrafficPredictor
from src.models.predictor_factory import get_traffic_predictor, AVAILABLE_MODELS

__all__ = ["BaseTrafficPredictor", "get_traffic_predictor", "AVAILABLE_MODELS"]
