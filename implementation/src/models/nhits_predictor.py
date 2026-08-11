"""N-HiTS traffic predictor (PyTorch multi-rate hierarchical model)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.sequence_predictor import SequencePredictor


class NHiTSBlock(nn.Module):
    """Hierarchical block: average-pool the input, then project through an MLP."""

    def __init__(self, input_len: int, pool_size: int, hidden_dim: int = 32):
        super().__init__()
        self.pool_size = pool_size
        pooled_len = max(1, input_len // pool_size)
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size) if pool_size > 1 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(pooled_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_size > 1 and x.shape[-1] >= self.pool_size:
            x_pooled = self.pool(x.unsqueeze(1)).squeeze(1)
        else:
            x_pooled = x
        return self.mlp(x_pooled).squeeze(-1)


class PyTorchNHiTSModule(nn.Module):
    """Sum of a low-frequency (pooled) block and a full-resolution block."""

    def __init__(self, input_len: int = 12, hidden_dim: int = 32):
        super().__init__()
        self.block1 = NHiTSBlock(input_len=input_len, pool_size=2, hidden_dim=hidden_dim)
        self.block2 = NHiTSBlock(input_len=input_len, pool_size=1, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block1(x) + self.block2(x)


class NHiTSTrafficPredictor(SequencePredictor):
    """Multi-rate hierarchical predictor; input windows are 2-D."""

    input_is_3d = False

    def _make_module(self) -> nn.Module:
        return PyTorchNHiTSModule(input_len=self.sequence_length, hidden_dim=self.hidden_dim)
