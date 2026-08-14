#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
====================================================================================================
 MODULE : src/models/lstm_predictor.py
 OBJET  : Prédicteur de Trafic Réseau basé sur un Réseau Neuronal Récurrent (PyTorch LSTM)
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce module implémente la classe `LSTMTrafficPredictor` basée sur un réseau de neurones récurrent LSTM (PyTorch).
Il s'insère dans la suite de prédicteurs (`src/models/predictor_factory.py`) pour alimenter le contrôleur SDN.

CARACTÉRISTIQUES CLÉS & SPÉCIFICITÉS :
-------------------------------------
  1. Horizon de Séquence : Exploite une fenêtre glissante (sequence_length = 12 pas = 2 heures) pour prédire \hat{l}^{t+1}.
  2. Normalisation `log1p` : Réduit l'asymétrie forte des volumes de trafic (pics d'eMBB) afin d'éviter
     l'écrasement des petites séries et de stabiliser l'apprentissage par descente de gradient.
  3. Modèles Indépendants par couple (tranche, station) : Empêche la contamination croisée entre stations
     à profils énergétiques ou de trafic divergents (ex: Macro-RAN 3 rurale sans eMBB).
====================================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.models.base_predictor import BaseTrafficPredictor


class PyTorchLSTMModule(nn.Module):
    """
    Module PyTorch LSTM à 1 couche cachée suivie d'une couche linéaire dense.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 1):
        """
        :param input_dim: Dimension d'entrée au temps t (1 feature de trafic).
        :param hidden_dim: Nombre d'unités récurrentes dans l'état caché.
        :param num_layers: Nombre de couches LSTM empilées.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant calculant la prédiction à partir du dernier état caché de la séquence.
        """
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


class LSTMTrafficPredictor(BaseTrafficPredictor):
    """
    Prédicteur de trafic basé sur le module PyTorch LSTM.
    """

    def __init__(self, sequence_length: int = 12, hidden_dim: int = 32,
                 epochs: int = 25, lr: float = 0.003, batch_size: int = 256):
        """
        :param sequence_length: Longueur de la séquence passée d'entrée (12 pas de 10 min = 2 heures).
        :param hidden_dim: Dimension des représentations cachées LSTM.
        :param epochs: Nombre d'époques d'entraînement.
        :param lr: Taux d'apprentissage de l'optimiseur Adam.
        :param batch_size: Taille des mini-batchs.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.models: Dict[Tuple[str, Optional[int]], PyTorchLSTMModule] = {}
        self.scalers: Dict[str, float] = {}

    def _normalize(self, series: np.ndarray, log_scale: float) -> np.ndarray:
        """Applique la transformation log1p normalisée."""
        return np.log1p(np.clip(series, 0, None)) / max(log_scale, 1e-6)

    def _denormalize(self, values: np.ndarray, log_scale: float) -> np.ndarray:
        """Inverse la transformation log1p normalisée."""
        return np.expm1(np.clip(values, 0, None) * log_scale)

    def extract_sequences_for_series(
        self,
        series: np.ndarray,
        log_scale: float,
        is_train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construit les fenêtres glissantes d'entrée et les cibles pour le modèle LSTM.

        :param series: Série temporelle brute.
        :param log_scale: Échelle de normalisation log1p.
        :param is_train: Mode apprentissage ou inférence.
        :return: Tuple (tenseurs X_arr, y_arr).
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

        X_arr = np.array(X_list, dtype=np.float32)[:, :, np.newaxis] if len(X_list) > 0 \
            else np.empty((0, self.sequence_length, 1), dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32) if len(y_list) > 0 else np.empty((0,), dtype=np.float32)

        return X_arr, y_arr

    def fit(self, df_train_pivoted: pd.DataFrame) -> None:
        """
        Entraîne un modèle LSTM distinct par couple (tranche, station Macro-RAN).
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
                model = PyTorchLSTMModule(input_dim=1, hidden_dim=self.hidden_dim)
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
        Génère les prédictions de trafic par inférence récurrente.
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