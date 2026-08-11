"""Passthrough / oracle predictor.

Sets ``pred_<slice>`` to the true maximum demand over the next hour
(``horizon`` steps). It is a perfect-foresight upper bound used to isolate the
control policy's quality from forecasting error, so ``evaluate`` reports zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional

from src.models.base_predictor import BaseTrafficPredictor, feature_columns, stations_of


class PassthroughTrafficPredictor(BaseTrafficPredictor):
    """Perfect-foresight predictor over a 1-hour horizon (6 steps)."""

    def __init__(self, horizon: int = 6):
        super().__init__()
        self.horizon = horizon

    def fit(self, df_train_pivoted: pd.DataFrame) -> None:
        self.slice_names = feature_columns(df_train_pivoted)

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df_res = df_pivoted.copy().reset_index(drop=True)
        if not self.slice_names:
            self.slice_names = feature_columns(df_pivoted)

        for slice_name in self.slice_names:
            for station in stations_of(df_res):
                if station is not None:
                    idx = df_res[df_res["id_institution_subnet"] == station].index
                    series = df_res.loc[idx, slice_name].values
                else:
                    idx = df_res.index
                    series = df_res[slice_name].values

                preds = [
                    float(np.max(series[i : i + self.horizon])) if i < len(series) else float(series[i])
                    for i in range(len(series))
                ]
                df_res.loc[idx, f"pred_{slice_name}"] = preds

        return df_res

    def evaluate(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        return {"MAE": 0.0, "RMSE": 0.0, "NMAE": 0.0}
