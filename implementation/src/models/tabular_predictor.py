"""Multi-scale tabular lag predictors (Ridge, LightGBM).

Both models turn each series into lag/rolling features over three horizons
(short term ~30 min, medium term 24 h, long term 7 days) and regress the maximum
demand of the next hour. The only differences are the feature set and the
regressor, expressed here as ``use_24h_max`` / ``use_calendar`` flags and the
``_make_model`` hook, so the windowing lives once in ``TabularLagPredictor``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.models.base_predictor import StationWindowPredictor


class TabularLagPredictor(StationWindowPredictor):
    """Shared multi-scale feature engineering for tabular regressors."""

    use_24h_max: bool = False
    use_calendar: bool = False

    def __init__(self, horizon: int = 6, short_lags: int = 3):
        super().__init__()
        self.horizon = horizon
        self.short_lags = short_lags
        self.lag_24h = 144
        self.lag_7d = 1008
        self.models: Dict[str, object] = {}

    def _make_model(self):
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        return self.short_lags + 2 + int(self.use_24h_max) + 1 + (3 if self.use_calendar else 0)

    @staticmethod
    def _calendar_features(timestamp) -> List[float]:
        try:
            ts = pd.to_datetime(timestamp)
            return [float(ts.hour), float(ts.dayofweek), 1.0 if ts.dayofweek >= 5 else 0.0]
        except Exception:
            return [0.0, 0.0, 0.0]

    def _extract(
        self,
        series: np.ndarray,
        ds_values: Optional[np.ndarray],
        is_train: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(series)
        x_list, y_list = [], []
        min_idx = self.short_lags
        max_idx = (n - self.horizon) if is_train else n

        for i in range(min_idx, max_idx):
            feat: List[float] = []

            for lag in range(self.short_lags):
                idx_lag = i - lag
                feat.append(float(series[idx_lag]) if idx_lag >= 0 else float(series[i]))

            idx_24h = i - self.lag_24h
            feat.append(float(series[idx_24h]) if idx_24h >= 0 else float(series[i]))

            start_24h = max(0, i - self.lag_24h + 1)
            window_24h = series[start_24h : i + 1]
            feat.append(float(np.mean(window_24h)) if len(window_24h) > 0 else float(series[i]))
            if self.use_24h_max:
                feat.append(float(np.max(window_24h)) if len(window_24h) > 0 else float(series[i]))

            idx_7d = i - self.lag_7d
            if idx_7d >= 0:
                feat.append(float(series[idx_7d]))
            elif idx_24h >= 0:
                feat.append(float(series[idx_24h]))
            else:
                feat.append(float(series[i]))

            if self.use_calendar:
                if ds_values is not None and i < len(ds_values):
                    feat.extend(self._calendar_features(ds_values[i]))
                else:
                    feat.extend([0.0, 0.0, 0.0])

            x_list.append(feat)

            if is_train:
                future_window = series[i + 1 : i + 1 + self.horizon]
                target = float(np.max(future_window)) if len(future_window) > 0 else float(series[i])
                y_list.append(target)

        x_arr = np.array(x_list, dtype=np.float32) if x_list else np.empty((0, self.n_features), dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32) if y_list else np.empty((0,), dtype=np.float32)
        return x_arr, y_arr

    def _build_train_arrays(self, slice_name, series, ds_values):
        return self._extract(series, ds_values, is_train=True)

    def _build_infer_arrays(self, slice_name, series, ds_values):
        x_pred, _ = self._extract(series, ds_values, is_train=False)
        return x_pred

    def _fit_slice(self, slice_name, x, y):
        model = self._make_model()
        model.fit(x, y)
        self.models[slice_name] = model

    def _predict_batch(self, slice_name, x):
        return self.models[slice_name].predict(x)

    def _has_model(self, slice_name):
        return slice_name in self.models
