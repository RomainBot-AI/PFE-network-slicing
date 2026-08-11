"""Shared base for PyTorch sequence predictors (LSTM, N-HiTS).

Both models scale each slice by its training maximum, slide a fixed-length
window over the series, and regress the scaled maximum demand of the next hour.
The scaling, windowing, and training loop live here; subclasses only provide the
``nn.Module`` and whether the input carries an explicit channel dimension.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple

from src.models.base_predictor import StationWindowPredictor


class SequencePredictor(StationWindowPredictor):
    """Base class for windowed neural sequence predictors."""

    input_is_3d: bool = False

    def __init__(
        self,
        sequence_length: int = 12,
        horizon: int = 6,
        hidden_dim: int = 32,
        epochs: int = 15,
        lr: float = 0.005,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.models: Dict[str, nn.Module] = {}
        self.scalers: Dict[str, float] = {}

    def _make_module(self) -> nn.Module:
        raise NotImplementedError

    def _extract(self, series: np.ndarray, max_val: float, is_train: bool) -> Tuple[np.ndarray, np.ndarray]:
        scaled = series / max_val
        n = len(scaled)
        x_list, y_list = [], []
        min_idx = self.sequence_length
        max_idx = (n - self.horizon) if is_train else n

        for i in range(min_idx, max_idx):
            x_list.append(scaled[i - self.sequence_length : i])
            if is_train:
                future_window = scaled[i + 1 : i + 1 + self.horizon]
                target = float(np.max(future_window)) if len(future_window) > 0 else float(scaled[i])
                y_list.append(target)

        if x_list:
            x_arr = np.array(x_list, dtype=np.float32)
            if self.input_is_3d:
                x_arr = x_arr[:, :, np.newaxis]
        else:
            shape = (0, self.sequence_length, 1) if self.input_is_3d else (0, self.sequence_length)
            x_arr = np.empty(shape, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32) if y_list else np.empty((0,), dtype=np.float32)
        return x_arr, y_arr

    def _prepare_slice(self, slice_name, df_pivoted):
        series = df_pivoted[slice_name].values
        max_val = float(np.max(series)) if len(series) > 0 and np.max(series) > 0 else 1.0
        self.scalers[slice_name] = max_val

    def _build_train_arrays(self, slice_name, series, ds_values):
        return self._extract(series, self.scalers[slice_name], is_train=True)

    def _build_infer_arrays(self, slice_name, series, ds_values):
        x_pred, _ = self._extract(series, self.scalers.get(slice_name, 1.0), is_train=False)
        return x_pred

    def _fit_slice(self, slice_name, x, y):
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        model = self._make_module()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(model(x_t), y_t)
            loss.backward()
            optimizer.step()
        model.eval()
        self.models[slice_name] = model

    def _predict_batch(self, slice_name, x):
        model = self.models[slice_name]
        max_val = self.scalers.get(slice_name, 1.0)
        with torch.no_grad():
            scaled_preds = model(torch.tensor(x, dtype=torch.float32)).numpy()
        return scaled_preds * max_val

    def _has_model(self, slice_name):
        return slice_name in self.models
