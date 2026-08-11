"""Per-slice LSTM benchmark using tuned hyperparameters."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nsf.benchmark.deterministic import BenchmarkDataConfig, _metric_rows, _summary_metrics, _benchmark_summary, _benchmark_summary_by_slice, _prepare_panel, _series_map, _validate_dense
from nsf.benchmark.lstm_tuning import _fit_log_scaler, _inverse_scale, _make_windows, _scale, _seasonal_scale, _select_device, _slice_series
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.splitting.panel_folds import folds_to_frame, leakage_audit, make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class LSTMBenchmarkConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    params_path: str
    train_origin_stride: int
    max_windows_per_slice: int
    device: str
    output_dir: str


def load_lstm_benchmark_config(path: str | Path) -> LSTMBenchmarkConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    training = raw.get("training", {})
    return LSTMBenchmarkConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        params_path=str(training["params_path"]),
        train_origin_stride=int(training.get("train_origin_stride", 288)),
        max_windows_per_slice=int(training.get("max_windows_per_slice", 2500)),
        device=str(training.get("device", "auto")),
        output_dir=str(raw.get("output", {}).get("output_dir", "forecasting/experiments/runs/lstm_benchmark")),
    )


def _load_best_params(path: str | Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    params = raw.get("best_params_by_slice", raw)
    return {str(slice_name): dict(values or {}) for slice_name, values in params.items()}


def _train_predict_lstm(params: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, mean: float, std: float, device) -> tuple[np.ndarray, float, float, int]:
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
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train).unsqueeze(-1), torch.from_numpy(y_train)),
        batch_size=int(params["batch_size"]),
        shuffle=True,
    )
    train_start = time.perf_counter()
    for _epoch in range(int(params["epochs"])):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - train_start
    infer_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x_eval).unsqueeze(-1).to(device)).cpu().numpy()
    inference_seconds = time.perf_counter() - infer_start
    pred = _inverse_scale(pred_scaled, mean, std)
    parameter_count = sum(param.numel() for param in model.parameters())
    return pred, train_seconds, inference_seconds, parameter_count


def _eval_sequences(series_list: list[pd.DataFrame], fold, input_size: int, mean: float, std: float) -> tuple[np.ndarray, pd.DataFrame]:
    x_rows = []
    meta_rows = []
    origin_idx = fold.train_end_idx
    for sub in series_list:
        values = sub["y"].to_numpy(dtype=float)
        x_raw = values[origin_idx - input_size + 1 : origin_idx + 1]
        if len(x_raw) != input_size:
            raise ValueError("Incomplete LSTM eval input window")
        x_rows.append(_scale(x_raw, mean, std))
        meta_rows.append(
            {
                "unique_id": str(sub["unique_id"].iloc[0]),
                "slice": str(sub["slice"].iloc[0]),
            }
        )
    return np.asarray(x_rows, dtype=np.float32), pd.DataFrame(meta_rows)


def run_lstm_benchmark(config: LSTMBenchmarkConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
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
    best_params = _load_best_params(config.params_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    timing_rows = []
    metadata_rows = []
    scales = {}

    for fold in folds:
        for slice_name in sorted(panel["slice"].unique()):
            series_list = _slice_series(series_map, str(slice_name))
            if str(slice_name) not in best_params:
                raise KeyError(f"Missing LSTM params for slice {slice_name}")
            params = best_params[str(slice_name)]
            mean, std = _fit_log_scaler(series_list, fold.train_end_idx)
            scale = _seasonal_scale(series_list, fold.train_end_idx, period=144)
            scales[(fold.fold, str(slice_name))] = scale
            last_origin_idx = fold.target_start_idx - config.backtest.horizon - 1
            x_train, y_train, _y_raw = _make_windows(
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
            x_eval, meta = _eval_sequences(series_list, fold, config.backtest.input_size, mean, std)
            pred, train_seconds, inference_seconds, parameter_count = _train_predict_lstm(params, x_train, y_train, x_eval, mean, std, device)
            timing_rows.append(
                {
                    "model": "lstm",
                    "fold": fold.fold,
                    "slice": str(slice_name),
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "train_windows": len(x_train),
                    "eval_series": len(x_eval),
                    "device": str(device),
                }
            )
            metadata_rows.append(
                {
                    "model": "lstm",
                    "fold": fold.fold,
                    "slice": str(slice_name),
                    "parameter_count": parameter_count,
                    "implementation": "torch.nn.LSTM",
                    "training_scope": "per_slice",
                    "trained_models": 1,
                }
            )
            for row_idx, meta_row in meta.iterrows():
                sub = series_map[str(meta_row["unique_id"])]
                target = sub.iloc[fold.target_start_idx : fold.target_end_idx + 1]
                for step_idx in range(config.backtest.horizon):
                    prediction_rows.append(
                        {
                            "fold": fold.fold,
                            "model": "lstm",
                            "training_scope": "per_slice",
                            "trained_slice": str(slice_name),
                            "unique_id": str(meta_row["unique_id"]),
                            "slice": str(slice_name),
                            "origin_timestamp": fold.train_end,
                            "timestamp": target["ds"].iloc[step_idx],
                            "horizon": step_idx + 1,
                            "y_true": float(target["y"].iloc[step_idx]),
                            "y_pred": float(pred[row_idx, step_idx]),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    metrics_by_fold = _metric_rows(predictions, scales)
    metrics = _summary_metrics(metrics_by_fold)
    summary = _benchmark_summary(metrics)
    summary_by_slice = _benchmark_summary_by_slice(metrics)
    timing = pd.DataFrame(timing_rows)
    metadata = pd.DataFrame(metadata_rows)
    paths = {
        "resolved_config": output_dir / "resolved_config.yaml",
        "run_meta": output_dir / "run_meta.json",
        "folds": output_dir / "folds.csv",
        "leakage_audit": output_dir / "leakage_audit.csv",
        "predictions": output_dir / "predictions.csv",
        "metrics_by_fold": output_dir / "metrics_by_fold.csv",
        "metrics": output_dir / "metrics.csv",
        "benchmark_summary": output_dir / "benchmark_summary.csv",
        "benchmark_summary_by_slice": output_dir / "benchmark_summary_by_slice.csv",
        "timing": output_dir / "timing.csv",
        "model_metadata": output_dir / "model_metadata.csv",
    }
    paths["resolved_config"].write_text(
        yaml.safe_dump(
            {
                "seed": config.seed,
                "data": config.data.__dict__,
                "backtest": config.backtest.__dict__,
                "training": {
                    "params_path": config.params_path,
                    "train_origin_stride": config.train_origin_stride,
                    "max_windows_per_slice": config.max_windows_per_slice,
                    "device": config.device,
                },
                "output": {"output_dir": config.output_dir},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    paths["run_meta"].write_text(
        json.dumps(
            {
                "panel_rows": int(len(panel)),
                "series": int(panel["unique_id"].nunique()),
                "folds": len(folds),
                "horizon": config.backtest.horizon,
                "predictions": int(len(predictions)),
                "trained_models": int(len(metadata)),
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    folds_to_frame(folds).to_csv(ensure_parent(paths["folds"]), index=False)
    leakage_audit(folds).to_csv(ensure_parent(paths["leakage_audit"]), index=False)
    predictions.to_csv(ensure_parent(paths["predictions"]), index=False)
    metrics_by_fold.to_csv(ensure_parent(paths["metrics_by_fold"]), index=False)
    metrics.to_csv(ensure_parent(paths["metrics"]), index=False)
    summary.to_csv(ensure_parent(paths["benchmark_summary"]), index=False)
    summary_by_slice.to_csv(ensure_parent(paths["benchmark_summary_by_slice"]), index=False)
    timing.to_csv(ensure_parent(paths["timing"]), index=False)
    metadata.to_csv(ensure_parent(paths["model_metadata"]), index=False)
    return paths
