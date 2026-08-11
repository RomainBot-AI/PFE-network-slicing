"""Abstract interfaces shared by all traffic predictors.

``BaseTrafficPredictor`` defines the minimal contract used by the environment
(``fit`` / ``predict_pivoted`` / ``evaluate``). ``StationWindowPredictor`` adds
the per-station training and inference loop shared by every windowed model
(Ridge, LightGBM, LSTM, N-HiTS): iterate over slices and stations, concatenate
optional context history, run the model, then align, pad, and clamp the
predictions back onto the input frame.

All predictors write one ``pred_<slice>`` column per slice and predict the
maximum demand over the next hour (6 steps of 10 minutes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

META_COLUMNS = ("ds", "id_institution_subnet")


class BaseTrafficPredictor:
    """Standard interface for network-traffic predictors."""

    def __init__(self) -> None:
        self.slice_names: List[str] = []

    def fit(self, df_train_pivoted: pd.DataFrame) -> None:
        """Fit the model on the pivoted training frame."""
        raise NotImplementedError

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return ``df_pivoted`` with an added ``pred_<slice>`` column per slice."""
        raise NotImplementedError

    def evaluate(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Compute MAE, RMSE, and NMAE (%) between actual and predicted traffic."""
        df_pred = self.predict_pivoted(df_pivoted, df_context=df_context)
        actuals, preds = [], []
        for slice_name in self.slice_names:
            actuals.extend(df_pred[slice_name].values)
            preds.extend(df_pred[f"pred_{slice_name}"].values)

        actuals = np.array(actuals, dtype=np.float64)
        preds = np.array(preds, dtype=np.float64)

        mae = float(np.mean(np.abs(preds - actuals)))
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
        max_actual = float(np.max(actuals)) if len(actuals) > 0 and np.max(actuals) > 0 else 1.0
        nmae = (mae / max_actual) * 100.0
        return {"MAE": mae, "RMSE": rmse, "NMAE": nmae}


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Slice columns of a pivoted frame (everything except the meta columns)."""
    return [c for c in df.columns if c not in META_COLUMNS]


def stations_of(df: pd.DataFrame) -> np.ndarray | List[None]:
    """Unique station ids, or ``[None]`` when the frame is not station-aware."""
    if "id_institution_subnet" in df.columns:
        return df["id_institution_subnet"].unique()
    return [None]


class StationWindowPredictor(BaseTrafficPredictor):
    """Base class for models trained and applied per station on sliding windows.

    Subclasses implement the model-specific hooks and inherit the shared
    train/predict orchestration:

    - ``_prepare_slice``     : per-slice setup before the station loop (e.g. scaler).
    - ``_build_train_arrays``: build ``(X, y)`` from one station's train series.
    - ``_fit_slice``         : fit and store a model for one slice.
    - ``_build_infer_arrays``: build ``X`` from one station's (context+eval) series.
    - ``_predict_batch``     : run the stored model and return unscaled predictions.
    - ``_has_model``         : whether a usable model exists for a slice.
    """

    def _prepare_slice(self, slice_name: str, df_pivoted: pd.DataFrame) -> None:
        return None

    def _build_train_arrays(
        self, slice_name: str, series: np.ndarray, ds_values: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _fit_slice(self, slice_name: str, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def _build_infer_arrays(
        self, slice_name: str, series: np.ndarray, ds_values: Optional[np.ndarray]
    ) -> np.ndarray:
        raise NotImplementedError

    def _predict_batch(self, slice_name: str, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _has_model(self, slice_name: str) -> bool:
        raise NotImplementedError

    def fit(self, df_train_pivoted: pd.DataFrame) -> None:
        df_piv = df_train_pivoted.copy().reset_index(drop=True)
        self.slice_names = feature_columns(df_piv)
        has_ds = "ds" in df_piv.columns

        for slice_name in self.slice_names:
            self._prepare_slice(slice_name, df_piv)
            x_parts, y_parts = [], []
            for station in stations_of(df_piv):
                df_st = df_piv if station is None else df_piv[df_piv["id_institution_subnet"] == station]
                series = df_st[slice_name].values
                ds_values = df_st["ds"].values if has_ds else None
                x_st, y_st = self._build_train_arrays(slice_name, series, ds_values)
                if len(x_st) > 0:
                    x_parts.append(x_st)
                    y_parts.append(y_st)

            if x_parts:
                self._fit_slice(slice_name, np.vstack(x_parts), np.concatenate(y_parts))

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df_piv = df_pivoted.copy().reset_index(drop=True)
        df_res = df_piv.copy()
        if not self.slice_names:
            self.slice_names = feature_columns(df_piv)

        has_ds = "ds" in df_piv.columns
        has_st = "id_institution_subnet" in df_piv.columns
        ctx = df_context.copy().reset_index(drop=True) if df_context is not None else None

        for slice_name in self.slice_names:
            if not self._has_model(slice_name):
                df_res[f"pred_{slice_name}"] = df_res[slice_name]
                continue

            for station in stations_of(df_piv):
                if station is not None:
                    idx = df_res[df_res["id_institution_subnet"] == station].index
                    df_st = df_piv.loc[idx]
                    ctx_st = ctx[ctx["id_institution_subnet"] == station] if ctx is not None and has_st else None
                else:
                    idx = df_res.index
                    df_st = df_piv
                    ctx_st = ctx

                series = df_st[slice_name].values
                ds_values = df_st["ds"].values if has_ds else None

                if ctx_st is not None and slice_name in ctx_st.columns:
                    full_series = np.concatenate([ctx_st[slice_name].values, series])
                    offset = len(ctx_st)
                    if has_ds and "ds" in ctx_st.columns:
                        full_ds = np.concatenate([ctx_st["ds"].values, ds_values])
                    else:
                        full_ds = ds_values
                else:
                    full_series = series
                    offset = 0
                    full_ds = ds_values

                x_pred = self._build_infer_arrays(slice_name, full_series, full_ds)
                if len(x_pred) == 0:
                    df_res.loc[idx, f"pred_{slice_name}"] = series
                    continue

                raw_preds = self._predict_batch(slice_name, x_pred)
                preds = raw_preds[offset:] if offset > 0 else raw_preds
                if len(preds) < len(idx):
                    pad_len = len(idx) - len(preds)
                    first_val = preds[0] if len(preds) > 0 else series[0]
                    preds = np.concatenate([np.full(pad_len, first_val), preds])

                df_res.loc[idx, f"pred_{slice_name}"] = np.maximum(0.0, preds[: len(idx)])

        return df_res
