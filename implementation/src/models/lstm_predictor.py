"""LSTM traffic predictor (PyTorch recurrent sequence model)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.sequence_predictor import SequencePredictor


class PyTorchLSTMModule(nn.Module):
    """Single-layer LSTM regressing the last hidden state to one value."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


class LSTMTrafficPredictor(SequencePredictor):
    """Recurrent predictor; input windows carry an explicit channel dimension."""

    input_is_3d = True

    def _make_module(self) -> nn.Module:
        return PyTorchLSTMModule(input_dim=1, hidden_dim=self.hidden_dim)
