"""Factory for traffic predictors.

Imports are deferred to ``get_traffic_predictor`` so that selecting a light model
(passthrough, ridge) does not require the heavy optional backends
(torch, lightgbm, prophet) to be installed.
"""

from __future__ import annotations

from src.models.base_predictor import BaseTrafficPredictor

AVAILABLE_MODELS = ["passthrough", "lightgbm", "lstm", "nhits", "prophet"]


def get_traffic_predictor(model_name: str = "passthrough", **kwargs) -> BaseTrafficPredictor:
    """Instantiate the predictor matching ``model_name``."""
    key = model_name.lower()

    if key == "passthrough":
        from src.models.passthrough_predictor import PassthroughTrafficPredictor

        return PassthroughTrafficPredictor(**kwargs)
    if key in {"ridge", "ml"}:
        from src.models.ridge_predictor import RidgeTrafficPredictor

        return RidgeTrafficPredictor(**kwargs)
    if key in {"lightgbm", "lgbm", "gbm"}:
        from src.models.lightgbm_predictor import LightGBMTrafficPredictor

        return LightGBMTrafficPredictor(**kwargs)
    if key == "lstm":
        from src.models.lstm_predictor import LSTMTrafficPredictor

        return LSTMTrafficPredictor(**kwargs)
    if key in {"nhits", "nhitm"}:
        from src.models.nhits_predictor import NHiTSTrafficPredictor

        return NHiTSTrafficPredictor(**kwargs)
    if key in {"prophet", "fbprophet"}:
        from src.models.prophet_predictor import ProphetTrafficPredictor

        return ProphetTrafficPredictor(**kwargs)

    raise ValueError(f"Unknown model: '{model_name}'. Available: {AVAILABLE_MODELS} or 'all'.")
