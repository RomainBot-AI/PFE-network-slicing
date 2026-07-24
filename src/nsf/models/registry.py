"""Model registry for config-driven experiments."""

from __future__ import annotations

from nsf.models.baselines import MovingAverageForecaster, PersistenceForecaster, SeasonalNaiveForecaster


def make_forecaster(name: str, **kwargs):
    if name in {"persistence", "naive"}:
        return PersistenceForecaster()
    if name in {"seasonal_naive", "seasonal_naive_daily"}:
        return SeasonalNaiveForecaster(seasonal_period=int(kwargs.get("seasonal_period", 144)), name=name)
    if name == "seasonal_naive_weekly":
        return SeasonalNaiveForecaster(seasonal_period=int(kwargs.get("seasonal_period", 1008)), name=name)
    if name == "moving_average":
        return MovingAverageForecaster(window=int(kwargs.get("window", 12)), name=name)
    raise KeyError(f"Unknown forecaster: {name}")
