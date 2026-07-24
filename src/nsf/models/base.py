"""Common forecaster contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastHorizon:
    steps: int
    freq: str = "10min"


class Forecaster(ABC):
    """Minimal model interface used by future backtest engines."""

    name: str

    @abstractmethod
    def fit(self, y: pd.Series) -> "Forecaster":
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: ForecastHorizon) -> pd.Series:
        raise NotImplementedError
