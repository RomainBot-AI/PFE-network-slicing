"""Prophet traffic predictor (Meta's additive time-series model).

Fits one Prophet model per (slice, station) with daily and weekly seasonality,
then smooths the forecast with a rolling max over the horizon to match the
"max demand over the next hour" target used by the other predictors.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

from src.models.base_predictor import BaseTrafficPredictor, feature_columns

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ProphetTrafficPredictor(BaseTrafficPredictor):
    """Per-station, per-slice Prophet predictor over a 1-hour horizon."""

    def __init__(self, horizon: int = 6, max_train_samples: int = 2016):
        super().__init__()
        self.horizon = horizon
        self.max_train_samples = max_train_samples  # ~2 weeks per station, for speed
        self.models: Dict[Tuple[str, int], Optional[Prophet]] = {}

    @staticmethod
    def _naive_ds(values: pd.Series) -> pd.Series:
        ds = pd.to_datetime(values)
        if ds.dt.tz is not None:
            ds = ds.dt.tz_localize(None)
        return ds

    def _stations(self, df: pd.DataFrame):
        return df["id_institution_subnet"].unique() if "id_institution_subnet" in df.columns else [0]

    def fit(self, df_train_pivoted: pd.DataFrame) -> None:
        df_piv = df_train_pivoted.copy().reset_index(drop=True)
        self.slice_names = feature_columns(df_piv)
        if "ds" not in df_piv.columns:
            df_piv["ds"] = pd.date_range(start="2024-01-01", periods=len(df_piv), freq="10min")

        for slice_name in self.slice_names:
            for station in self._stations(df_piv):
                if "id_institution_subnet" in df_piv.columns:
                    df_st = df_piv[df_piv["id_institution_subnet"] == station].copy()
                else:
                    df_st = df_piv.copy()

                df_st = df_st.sort_values(by="ds").tail(self.max_train_samples).reset_index(drop=True)
                df_prophet = pd.DataFrame({"ds": self._naive_ds(df_st["ds"]), "y": df_st[slice_name].values})

                model = Prophet(
                    growth="flat",
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    interval_width=0.80,
                )
                try:
                    model.fit(df_prophet)
                    self.models[(slice_name, station)] = model
                except Exception:
                    self.models[(slice_name, station)] = None

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df_res = df_pivoted.copy().reset_index(drop=True)
        self.slice_names = feature_columns(df_res)
        if "ds" not in df_res.columns:
            df_res["ds"] = pd.date_range(start="2024-01-01", periods=len(df_res), freq="10min")

        has_st = "id_institution_subnet" in df_res.columns

        for slice_name in self.slice_names:
            col_pred = f"pred_{slice_name}"
            df_res[col_pred] = df_res[slice_name].values

            for station in self._stations(df_res):
                model = self.models.get((slice_name, station))
                if model is None:
                    continue
                try:
                    mask = (df_res["id_institution_subnet"] == station) if has_st else np.ones(len(df_res), dtype=bool)
                    df_st = df_res[mask]

                    forecast = model.predict(pd.DataFrame({"ds": self._naive_ds(df_st["ds"])}))
                    pred_vals = np.maximum(0.0, forecast["yhat"].values)
                    pred_smooth = pd.Series(pred_vals, index=df_st.index).rolling(self.horizon, min_periods=1).max().values

                    df_res.loc[mask, col_pred] = pred_smooth
                except Exception:
                    pass

        return df_res
