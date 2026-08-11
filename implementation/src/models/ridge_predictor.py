"""Ridge-regression traffic predictor (multi-scale lag features, no calendar)."""

from __future__ import annotations

from sklearn.linear_model import Ridge

from src.models.tabular_predictor import TabularLagPredictor


class RidgeTrafficPredictor(TabularLagPredictor):
    """Supervised multi-scale predictor using ridge regression."""

    use_24h_max = False
    use_calendar = False

    def __init__(self, horizon: int = 6, short_lags: int = 3, alpha: float = 1.0):
        super().__init__(horizon=horizon, short_lags=short_lags)
        self.alpha = alpha

    def _make_model(self) -> Ridge:
        return Ridge(alpha=self.alpha)


# Backwards-compatible alias (the class used to be named MLTrafficPredictor).
MLTrafficPredictor = RidgeTrafficPredictor
