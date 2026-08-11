"""Optuna tuning for per-slice LSTM forecasting."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nsf.benchmark.deterministic import BenchmarkDataConfig, _prepare_panel, _series_map, _validate_dense
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.evaluation.deterministic import mase, rmse
from nsf.splitting.panel_folds import make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class LSTMTuningConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    n_trials: int
    objective: str
    validation_fraction: float
    train_origin_stride: int
    max_windows_per_slice: int
    max_epochs: int
    patience: int
    device: str
    output_dir: str


def load_lstm_tuning_config(path: str | Path) -> LSTMTuningConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tuning = raw.get("tuning", {})
    return LSTMTuningConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        n_trials=int(tuning.get("n_trials", 20)),
        objective=str(tuning.get("objective", "MASE")),
        validation_fraction=float(tuning.get("validation_fraction", 0.2)),
        train_origin_stride=int(tuning.get("train_origin_stride", 144)),
        max_windows_per_slice=int(tuning.get("max_windows_per_slice", 3000)),
        max_epochs=int(tuning.get("max_epochs", 20)),
        patience=int(tuning.get("patience", 4)),
        device=str(tuning.get("device", "auto")),
        output_dir=str(raw.get("output", {}).get("output_dir", "forecasting/experiments/runs/lstm_tuning")),
    )


def _select_device(device: str):
    import torch

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    if device == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _slice_series(series_map: dict[str, pd.DataFrame], slice_name: str) -> list[pd.DataFrame]:
    return [sub for sub in series_map.values() if str(sub["slice"].iloc[0]) == slice_name]


def _fit_log_scaler(series_list: list[pd.DataFrame], train_end_idx: int) -> tuple[float, float]:
    values = np.concatenate([np.log1p(np.maximum(0.0, sub["y"].iloc[: train_end_idx + 1].to_numpy(dtype=float))) for sub in series_list])
    mean = float(values.mean())
    std = float(values.std())
    if std == 0.0 or not np.isfinite(std):
        std = 1.0
    return mean, std


def _scale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((np.log1p(np.maximum(0.0, values.astype(float))) - mean) / std).astype(np.float32)


def _inverse_scale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.maximum(0.0, np.expm1(values.astype(float) * std + mean))


def _make_windows(
    series_list: list[pd.DataFrame],
    input_size: int,
    horizon: int,
    last_origin_idx: int,
    stride: int,
    mean: float,
    std: float,
    max_windows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    y_raw_rows = []
    first_origin_idx = input_size - 1
    if last_origin_idx < first_origin_idx:
        raise ValueError("Not enough history to build LSTM windows")
    origins = list(range(first_origin_idx, last_origin_idx + 1, stride))
    candidates: list[tuple[pd.DataFrame, int]] = []
    for sub in series_list:
        candidates.extend((sub, origin_idx) for origin_idx in origins)
    if len(candidates) > max_windows:
        rng = random.Random(seed)
        candidates = sorted(rng.sample(candidates, max_windows), key=lambda item: item[1])
    for sub, origin_idx in candidates:
        values = sub["y"].to_numpy(dtype=float)
        x_raw = values[origin_idx - input_size + 1 : origin_idx + 1]
        y_raw = values[origin_idx + 1 : origin_idx + 1 + horizon]
        if len(x_raw) != input_size or len(y_raw) != horizon:
            continue
        x_rows.append(_scale(x_raw, mean, std))
        y_rows.append(_scale(y_raw, mean, std))
        y_raw_rows.append(y_raw.astype(np.float32))
    if not x_rows:
        raise ValueError("No LSTM windows generated")
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32), np.asarray(y_raw_rows, dtype=np.float32)


def _chronological_split(x: np.ndarray, y: np.ndarray, y_raw: np.ndarray, validation_fraction: float):
    val_size = max(1, int(round(len(x) * validation_fraction)))
    train_size = len(x) - val_size
    if train_size <= 0:
        raise ValueError("Validation split leaves no training windows")
    return x[:train_size], y[:train_size], y_raw[:train_size], x[train_size:], y[train_size:], y_raw[train_size:]


def _seasonal_scale(series_list: list[pd.DataFrame], train_end_idx: int, period: int) -> float:
    diffs = []
    for sub in series_list:
        values = sub["y"].iloc[: train_end_idx + 1].to_numpy(dtype=float)
        if len(values) > period:
            diffs.extend(np.abs(values[period:] - values[:-period]).tolist())
    return float(np.mean(diffs)) if diffs else 1.0


def _suggest_params(trial, max_epochs: int) -> dict[str, Any]:
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": trial.suggest_int("epochs", max(2, max_epochs // 2), max_epochs),
    }


def _train_eval_lstm(params: dict[str, Any], x_train, y_train, x_val, y_val, y_val_raw, mean: float, std: float, scale: float, objective: str, device) -> tuple[float, float]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class HorizonLSTM(nn.Module):
        def __init__(self, hidden_size: int, num_layers: int, dropout: float, horizon: int):
            super().__init__()
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True, dropout=lstm_dropout)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, horizon)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.dropout(out[:, -1, :]))

    model = HorizonLSTM(
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
        horizon=y_train.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(x_train).unsqueeze(-1), torch.from_numpy(y_train))
    loader = DataLoader(train_ds, batch_size=int(params["batch_size"]), shuffle=True)
    x_val_t = torch.from_numpy(x_val).unsqueeze(-1).to(device)
    best = float("inf")
    best_pred = None
    patience_left = 3
    start = time.perf_counter()
    for _epoch in range(int(params["epochs"])):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(x_val_t).cpu().numpy()
        pred_raw = _inverse_scale(pred, mean, std)
        true_raw = y_val_raw
        score = rmse(true_raw.ravel(), pred_raw.ravel()) if objective.upper() == "RMSE" else mase(true_raw.ravel(), pred_raw.ravel(), scale)
        if score < best:
            best = score
            best_pred = pred_raw
            patience_left = 3
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    elapsed = time.perf_counter() - start
    return float(best), elapsed


def tune_lstm_by_slice(config: LSTMTuningConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
    import optuna

    device = _select_device(config.device)
    panel = _prepare_panel(read_panel(config.data.panel_csv))
    timestamps = pd.DatetimeIndex(sorted(panel["ds"].unique()))
    series_map = _series_map(panel)
    _validate_dense(series_map, timestamps)
    folds = make_panel_folds(
        timestamps=timestamps,
        input_size=config.backtest.input_size,
        horizon=config.backtest.horizon,
        n_folds=config.backtest.n_folds,
        fold_stride=config.backtest.fold_stride,
        expanding=config.backtest.expanding,
    )
    fold = folds[0]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_rows = []
    best_params_by_slice = {}
    for slice_name in sorted(panel["slice"].unique()):
        series_list = _slice_series(series_map, str(slice_name))
        mean, std = _fit_log_scaler(series_list, fold.train_end_idx)
        last_origin_idx = fold.target_start_idx - config.backtest.horizon - 1
        x, y, y_raw = _make_windows(
            series_list=series_list,
            input_size=config.backtest.input_size,
            horizon=config.backtest.horizon,
            last_origin_idx=last_origin_idx,
            stride=config.train_origin_stride,
            mean=mean,
            std=std,
            max_windows=config.max_windows_per_slice,
            seed=config.seed,
        )
        x_train, y_train, _y_train_raw, x_val, y_val, y_val_raw = _chronological_split(x, y, y_raw, config.validation_fraction)
        scale = _seasonal_scale(series_list, fold.train_end_idx, period=144)

        def objective(trial):
            params = _suggest_params(trial, config.max_epochs)
            score, seconds = _train_eval_lstm(params, x_train, y_train, x_val, y_val, y_val_raw, mean, std, scale, config.objective, device)
            row = {
                "slice": str(slice_name),
                "trial": trial.number,
                "objective": config.objective.upper(),
                "score": score,
                "seconds": seconds,
                "train_windows": len(x_train),
                "val_windows": len(x_val),
                "device": str(device),
            }
            row.update(params)
            trial_rows.append(row)
            return score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed))
        study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)
        best_params_by_slice[str(slice_name)] = dict(study.best_params)

    paths = {
        "trials": output_dir / "lstm_tuning_trials.csv",
        "best_params": output_dir / "lstm_best_params_by_slice.yaml",
        "run_meta": output_dir / "run_meta.json",
        "resolved_config": output_dir / "resolved_config.yaml",
    }
    pd.DataFrame(trial_rows).to_csv(ensure_parent(paths["trials"]), index=False)
    paths["best_params"].write_text(yaml.safe_dump({"best_params_by_slice": best_params_by_slice}, sort_keys=False), encoding="utf-8")
    paths["run_meta"].write_text(
        json.dumps(
            {
                "slices": len(best_params_by_slice),
                "n_trials": config.n_trials,
                "objective": config.objective.upper(),
                "tuned_fold": fold.fold,
                "device": str(device),
                "input_size": config.backtest.input_size,
                "horizon": config.backtest.horizon,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["resolved_config"].write_text(
        yaml.safe_dump(
            {
                "seed": config.seed,
                "data": config.data.__dict__,
                "backtest": config.backtest.__dict__,
                "tuning": {
                    "n_trials": config.n_trials,
                    "objective": config.objective,
                    "validation_fraction": config.validation_fraction,
                    "train_origin_stride": config.train_origin_stride,
                    "max_windows_per_slice": config.max_windows_per_slice,
                    "max_epochs": config.max_epochs,
                    "patience": config.patience,
                    "device": config.device,
                },
                "output": {"output_dir": config.output_dir},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return paths
