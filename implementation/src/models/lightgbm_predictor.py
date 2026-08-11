"""LightGBM traffic predictor with multi-scale lag and calendar features.

Combines three temporal scales -- short term (last 30 min), medium term (24 h
lag, rolling mean and max), long term (7-day lag) -- plus calendar features
(hour, day of week, weekend) to predict the maximum demand of the next hour.
"""

from __future__ import annotations

import lightgbm as lgb

from src.models.tabular_predictor import TabularLagPredictor


class LightGBMTrafficPredictor(TabularLagPredictor):
    """Multi-scale gradient-boosting predictor."""

    use_24h_max = True
    use_calendar = True

    def _make_model(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.08,
            num_leaves=63,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            verbose=-1,
            importance_type="gain",
        )
