#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/models/nhits_predictor.py
 OBJET  : Prédicteur Multi-Échelle de Trafic basé sur N-HiTS (Neural Hierarchical Interpolation)
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce module implémente le modèle N-HiTS en PyTorch (`NHiTSTrafficPredictor`).
Il fait partie des modèles de prévision de séries temporelles (`src/models/predictor_factory.py`).

CARACTÉRISTIQUES CLÉS & SPÉCIFICITÉS :
-------------------------------------
  1. Interpolation Hiérarchique Multi-Résolution : Combine deux blocs d'agrégation temporelle
     (bloc basse fréquence avec pooling de taille 2 pour la tendance, et bloc haute fréquence avec pooling 1).
  2. Normalisation `log1p` : Stabilise la régresssion sur les séries à fortes amplitudes et valeurs nulles.
  3. Apprentissage Séparé par (tranche, station) : Garantit que chaque station Macro-RAN possède son propre
     modèle N-HiTS sans biais causé par d'autres sous-réseaux.
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
    Bloc d'agrégation temporelle N-HiTS avec sous-échantillonnage (pooling) et couche MLP.
    """
    def __init__(self, input_len: int, pool_size: int, hidden_dim: int = 32):
        """
        :param input_len: Longueur de la séquence d'entrée.
        :param pool_size: Facteur de sous-échantillonnage moyen (pooling).
        :param hidden_dim: Nombre de neurones dans les couches cachées du MLP.
        """
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
        """
        Application du sous-échantillonnage et de la projection MLP.
        """
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
        """
        Somme résiduelle des prédictions des deux échelles temporelles.
        """
        out1 = self.block1(x)
        out2 = self.block2(x)
        return out1 + out2


class NHiTSTrafficPredictor(BaseTrafficPredictor):
    """
    Prédicteur de trafic réseau basé sur l'architecture neuronale N-HiTS.
    """

    def __init__(self, sequence_length: int = 12, hidden_dim: int = 32,
                 epochs: int = 25, lr: float = 0.003, batch_size: int = 256):
        """
        :param sequence_length: Longueur de l'historique d'entrée (12 pas de 10 min = 2 heures).
        :param hidden_dim: Dimension cachée du réseau.
        :param epochs: Nombre d'époques d'entraînement.
        :param lr: Taux d'apprentissage.
        :param batch_size: Taille des mini-batchs.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.models: Dict[Tuple[str, Optional[int]], PyTorchNHiTSModule] = {}
        self.scalers: Dict[str, float] = {}

    def _normalize(self, series: np.ndarray, log_scale: float) -> np.ndarray:
        """Transformation log1p normalisée."""
        return np.log1p(np.clip(series, 0, None)) / max(log_scale, 1e-6)

    def _denormalize(self, values: np.ndarray, log_scale: float) -> np.ndarray:
        """Inversion de la transformation log1p normalisée."""
        return np.expm1(np.clip(values, 0, None) * log_scale)

    def extract_sequences_for_series(
        self,
        series: np.ndarray,
        log_scale: float,
        is_train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construit les fenêtres glissantes et les cibles cibles pour N-HiTS.

        :param series: Série brute.
        :param log_scale: Facteur d'échelle log1p.
        :param is_train: Flag d'entraînement ou d'inférence.
        :return: Tuple (matrice X, vecteur y).
        """
        scaled_series = self._normalize(series, log_scale)
        n = len(scaled_series)
        X_list, y_list = [], []

        min_idx = self.sequence_length
        max_idx = n

        for i in range(min_idx, max_idx):
            X_list.append(scaled_series[i - self.sequence_length: i])
            if is_train:
                y_list.append(scaled_series[i])

        X_arr = np.array(X_list, dtype=np.float32) if len(X_list) > 0 \
            else np.empty((0, self.sequence_length), dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32) if len(y_list) > 0 else np.empty((0,), dtype=np.float32)

        return X_arr, y_arr

    def fit(self, df_train_pivoted: pd.DataFrame) -> None:
        """
        Entraîne un modèle N-HiTS indépendant par (tranche, station Macro-RAN).
        """
        df_piv = df_train_pivoted.copy().reset_index(drop=True)
        self.slice_names = [c for c in df_piv.columns if c not in ['ds', 'id_institution_subnet']]
        has_st = 'id_institution_subnet' in df_piv.columns
        stations = df_piv['id_institution_subnet'].unique() if has_st else [None]

        for slice_name in self.slice_names:
            full_series_all = df_piv[slice_name].values
            log_scale = float(np.log1p(np.max(full_series_all))) if np.max(full_series_all) > 0 else 1.0
            self.scalers[slice_name] = log_scale

            for st in stations:
                df_st = df_piv[df_piv['id_institution_subnet'] == st] if st is not None else df_piv
                series_st = df_st[slice_name].values
                X_st, y_st = self.extract_sequences_for_series(series_st, log_scale, is_train=True)
                if len(X_st) < 2:
                    continue

                n_samples = len(X_st)
                model = PyTorchNHiTSModule(input_len=self.sequence_length, hidden_dim=self.hidden_dim)
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
                criterion = nn.MSELoss()

                model.train()
                for epoch in range(self.epochs):
                    perm = np.random.permutation(n_samples)
                    for start in range(0, n_samples, self.batch_size):
                        idx = perm[start:start + self.batch_size]
                        xb = torch.tensor(X_st[idx], dtype=torch.float32)
                        yb = torch.tensor(y_st[idx], dtype=torch.float32)

                        optimizer.zero_grad()
                        output = model(xb)
                        loss = criterion(output, yb)
                        loss.backward()
                        optimizer.step()

                model.eval()
                self.models[(slice_name, st)] = model

    def predict_pivoted(
        self,
        df_pivoted: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Génère les prédictions de trafic par inférence N-HiTS multi-échelle.
        """
        df_piv = df_pivoted.copy().reset_index(drop=True)
        df_res = df_piv.copy()

        if not self.slice_names:
            self.slice_names = [c for c in df_piv.columns if c not in ['ds', 'id_institution_subnet']]

        has_st = 'id_institution_subnet' in df_piv.columns
        stations = df_piv['id_institution_subnet'].unique() if has_st else [None]
        df_ctx_clean = df_context.copy().reset_index(drop=True) if df_context is not None else None

        for slice_name in self.slice_names:
            log_scale = self.scalers.get(slice_name, 1.0)

            for st in stations:
                if st is not None:
                    idx_st = df_res[df_res['id_institution_subnet'] == st].index
                    df_st = df_piv.loc[idx_st]
                    df_ctx_st = df_ctx_clean[df_ctx_clean['id_institution_subnet'] == st] if df_ctx_clean is not None and has_st else None
                else:
                    idx_st = df_res.index
                    df_st = df_piv
                    df_ctx_st = df_ctx_clean

                model = self.models.get((slice_name, st))
                if model is None:
                    df_res.loc[idx_st, f'pred_{slice_name}'] = df_st[slice_name]
                    continue

                series = df_st[slice_name].values.astype(np.float32)

                if df_ctx_st is not None and slice_name in df_ctx_st.columns:
                    ctx_series = df_ctx_st[slice_name].values.astype(np.float32)
                    full_series = np.concatenate([ctx_series, series])
                    offset = len(ctx_series)
                else:
                    full_series = series
                    offset = 0

                X_pred, _ = self.extract_sequences_for_series(full_series, log_scale, is_train=False)

                if len(X_pred) > 0:
                    with torch.no_grad():
                        raw_preds_scaled = model(torch.tensor(X_pred, dtype=torch.float32)).numpy()

                    raw_preds = self._denormalize(raw_preds_scaled, log_scale)
                    preds_target = raw_preds[offset:] if offset > 0 else raw_preds

                    if len(preds_target) < len(idx_st):
                        pad_len = len(idx_st) - len(preds_target)
                        first_val = preds_target[0] if len(preds_target) > 0 else series[0]
                        preds_target = np.concatenate([np.full(pad_len, first_val), preds_target])

                    df_res.loc[idx_st, f'pred_{slice_name}'] = np.maximum(0.0, preds_target[:len(idx_st)])
                else:
                    df_res.loc[idx_st, f'pred_{slice_name}'] = series

        return df_res