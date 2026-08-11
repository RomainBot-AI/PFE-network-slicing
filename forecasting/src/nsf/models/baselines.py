"""Simple deterministic baseline forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nsf.models.base import Forecaster, ForecastHorizon


class PersistenceForecaster(Forecaster):
    name = "persistence"

    def fit(self, y: pd.Series) -> "PersistenceForecaster":
        if y.empty:
            raise ValueError("Cannot fit persistence on an empty series")
        self.history_ = y.astype(float).copy()
        return self

    def predict(self, horizon: ForecastHorizon) -> pd.Series:
        value = float(self.history_.iloc[-1])
        return pd.Series(np.repeat(value, horizon.steps), name=self.name)


class SeasonalNaiveForecaster(Forecaster):
    def __init__(self, seasonal_period: int, name: str | None = None):
        self.seasonal_period = seasonal_period
        self.name = name or f"seasonal_naive_{seasonal_period}"

    def fit(self, y: pd.Series) -> "SeasonalNaiveForecaster":
        if y.empty:
            raise ValueError("Cannot fit seasonal naive on an empty series")
        self.history_ = y.astype(float).copy()
        return self

    def predict(self, horizon: ForecastHorizon) -> pd.Series:
        if len(self.history_) < self.seasonal_period:
            value = float(self.history_.iloc[-1])
            values = np.repeat(value, horizon.steps)
        else:
            season = self.history_.iloc[-self.seasonal_period :].to_numpy(dtype=float)
            reps = int(np.ceil(horizon.steps / len(season)))
            values = np.tile(season, reps)[: horizon.steps]
        return pd.Series(values, name=self.name)


class MovingAverageForecaster(Forecaster):
    def __init__(self, window: int, name: str | None = None):
        self.window = window
        self.name = name or f"moving_average_{window}"

    def fit(self, y: pd.Series) -> "MovingAverageForecaster":
        if y.empty:
            raise ValueError("Cannot fit moving average on an empty series")
        self.history_ = y.astype(float).copy()
        return self

    def predict(self, horizon: ForecastHorizon) -> pd.Series:
        window = max(1, min(self.window, len(self.history_)))
        value = float(self.history_.iloc[-window:].mean())
        return pd.Series(np.repeat(value, horizon.steps), name=self.name)
