#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/models/nhits_predictor.py
 OBJET  : Prédicteur Multi-Échelle de Trafic basé sur N-HiTS (PyTorch)
====================================================================================================

DESCRIPTION DÉTAILLÉE :
-----------------------
Implémente la classe `NHiTSTrafficPredictor` basée sur N-HiTS.
- Horizon d'entrée : Séquence temporelle étendue (sequence_length = 12 pas = 2 heures)
- Target de sortie  : Demande maximale de la prochaine heure (horizon = 6 pas de 10 min)
- Traitement station-aware séparé par id_institution_subnet.

====================================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.models.base_predictor import BaseTrafficPredictor


class NHiTSBlock(nn.Module):
    """
    Bloc Hiérarchique N-HiTS avec sous-échantillonnage (pooling) et projection résiduelle.
    """
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
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_size > 1 and x.shape[-1] >= self.pool_size:
            x_pooled = self.pool(x.unsqueeze(1)).squeeze(1)
        else:
            x_pooled = x
        out = self.mlp(x_pooled)
        return out.squeeze(-1)


class PyTorchNHiTSModule(nn.Module):
    """
    Module Multi-Échelle N-HiTS combinant un bloc de tendance basse fréquence et un bloc haute fréquence.
    """
    def __init__(self, input_len: int = 12, hidden_dim: int = 32):
        super().__init__()
        self.block1 = NHiTSBlock(input_len=input_len, pool_size=2, hidden_dim=hidden_dim)
        self.block2 = NHiTSBlock(input_len=input_len, pool_size=1, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.block1(x)
        out2 = self.block2(x)
        return out1 + out2


class NHiTSTrafficPredictor(BaseTrafficPredictor):
    """
    Prédicteur de Trafic Réseau basé sur l'Architecture Neuronale N-HiTS.
    """

    def __init__(self, sequence_length: int = 12, horizon: int = 6, hidden_dim: int = 32, epochs: int = 15, lr: float = 0.005):
        super().__init__()
        self.sequence_length = sequence_length  # 12 pas de 10 min = 2h d'historique direct
        self.horizon = horizon                  # 6 pas de 10 min = 1h de prédiction
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.models: Dict[str, PyTorchNHiTSModule] = {}
        self.scalers: Dict[str, float] = {}

    def extract_sequences_for_series(
        self,
        series: np.ndarray,
        max_val: float,
        is_train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        scaled_series = series / max_val
        n = len(scaled_series)
        X_list, y_list = [], []

        min_idx = self.sequence_length
        max_idx = (n - self.horizon) if is_train else n

        for i in range(min_idx, max_idx):
            X_list.append(scaled_series[i - self.sequence_length : i])

            if is_train:
                future_window = scaled_series[i + 1 : i + 1 + self.horizon]
                target_val = float(np.max(future_window)) if len(future_window) > 0 else float(scaled_series[i])
                y_list.append(target_val)

        X_arr = np.array(X_list, dtype=np.float32) if len(X_list) > 0 else np.empty((0, self.sequence_length), dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32) if len(y_list) > 0 else np.empty((0,), dtype=np.float32)

        return X_arr, y_arr

    def fit(self, df_train_pivoted: pd.DataFrame):
        df_piv = df_train_pivoted.copy().reset_index(drop=True)
        self.slice_names = [c for c in df_piv.columns if c not in ['ds', 'id_institution_subnet']]
        has_st = 'id_institution_subnet' in df_piv.columns
        stations = df_piv['id_institution_subnet'].unique() if has_st else [None]

        for slice_name in self.slice_names:
            full_series_all = df_piv[slice_name].values
            max_val = float(np.max(full_series_all)) if np.max(full_series_all) > 0 else 1.0
            self.scalers[slice_name] = max_val

            X_all, y_all = [], []
            for st in stations:
                if st is not None:
                    df_st = df_piv[df_piv['id_institution_subnet'] == st]
                else:
                    df_st = df_piv

                series_st = df_st[slice_name].values
                X_st, y_st = self.extract_sequences_for_series(series_st, max_val, is_train=True)
                if len(X_st) > 0:
                    X_all.append(X_st)
                    y_all.append(y_st)

            if len(X_all) > 0:
                X_t = torch.tensor(np.vstack(X_all), dtype=torch.float32)
                y_t = torch.tensor(np.concatenate(y_all), dtype=torch.float32)

                model = PyTorchNHiTSModule(input_len=self.sequence_length, hidden_dim=self.hidden_dim)
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
                criterion = nn.MSELoss()

                model.train()
                for epoch in range(self.epochs):
                    optimizer.zero_grad()
                    output = model(X_t)
                    loss = criterion(output, y_t)
                    loss.backward()
                    optimizer.step()

                model.eval()
                self.models[slice_name] = model

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        df_piv = df_pivoted.copy().reset_index(drop=True)
        df_res = df_piv.copy()

        if not self.slice_names:
            self.slice_names = [c for c in df_piv.columns if c not in ['ds', 'id_institution_subnet']]

        has_st = 'id_institution_subnet' in df_piv.columns
        stations = df_piv['id_institution_subnet'].unique() if has_st else [None]
        df_ctx_clean = df_context.copy().reset_index(drop=True) if df_context is not None else None

        for slice_name in self.slice_names:
            if slice_name not in self.models:
                df_res[f'pred_{slice_name}'] = df_res[slice_name]
                continue

            model = self.models[slice_name]
            max_val = self.scalers.get(slice_name, 1.0)

            for st in stations:
                if st is not None:
                    idx_st = df_res[df_res['id_institution_subnet'] == st].index
                    df_st = df_piv.loc[idx_st]
                    df_ctx_st = df_ctx_clean[df_ctx_clean['id_institution_subnet'] == st] if df_ctx_clean is not None and has_st else None
                else:
                    idx_st = df_res.index
                    df_st = df_piv
                    df_ctx_st = df_ctx_clean

                series = df_st[slice_name].values.astype(np.float32)

                if df_ctx_st is not None and slice_name in df_ctx_st.columns:
                    ctx_series = df_ctx_st[slice_name].values.astype(np.float32)
                    full_series = np.concatenate([ctx_series, series])
                    offset = len(ctx_series)
                else:
                    full_series = series
                    offset = 0

                X_pred, _ = self.extract_sequences_for_series(full_series, max_val, is_train=False)

                if len(X_pred) > 0:
                    with torch.no_grad():
                        raw_preds_scaled = model(torch.tensor(X_pred, dtype=torch.float32)).numpy()

                    raw_preds = raw_preds_scaled * max_val
                    preds_target = raw_preds[offset:] if offset > 0 else raw_preds

                    if len(preds_target) < len(idx_st):
                        pad_len = len(idx_st) - len(preds_target)
                        first_val = preds_target[0] if len(preds_target) > 0 else series[0]
                        preds_target = np.concatenate([np.full(pad_len, first_val), preds_target])

                    df_res.loc[idx_st, f'pred_{slice_name}'] = np.maximum(0.0, preds_target[:len(idx_st)])
                else:
                    df_res.loc[idx_st, f'pred_{slice_name}'] = series

        return df_res
