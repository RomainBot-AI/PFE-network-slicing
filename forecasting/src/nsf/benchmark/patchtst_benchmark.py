"""Per-slice PatchTST benchmark using NeuralForecast."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nsf.benchmark.deterministic import (
    BenchmarkDataConfig,
    _benchmark_summary,
    _benchmark_summary_by_slice,
    _metric_rows,
    _prepare_panel,
    _series_map,
    _summary_metrics,
    _validate_dense,
)
from nsf.benchmark.lstm_tuning import _seasonal_scale, _slice_series
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.splitting.panel_folds import folds_to_frame, leakage_audit, make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


@dataclass(frozen=True)
class PatchTSTModelConfig:
    max_steps: int = 300
    learning_rate: float = 1e-4
    batch_size: int = 32
    windows_batch_size: int = 1024
    inference_windows_batch_size: int = 1024
    step_size: int = 288
    scaler_type: str = "standard"
    encoder_layers: int = 3
    n_heads: int = 8
    hidden_size: int = 128
    linear_hidden_size: int = 256
    dropout: float = 0.2
    fc_dropout: float = 0.2
    head_dropout: float = 0.0
    attn_dropout: float = 0.0
    patch_len: int = 16
    stride: int = 8
    revin: bool = True
    revin_affine: bool = False
    revin_subtract_last: bool = True
    activation: str = "gelu"
    res_attention: bool = True
    batch_normalization: bool = False
    learn_pos_embed: bool = True
    val_check_steps: int = 50
    early_stop_patience_steps: int = -1


@dataclass(frozen=True)
class PatchTSTTrainingConfig:
    device: str = "auto"
    slices: tuple[str, ...] = field(default_factory=tuple)
    params_path: str | None = None


@dataclass(frozen=True)
class PatchTSTBenchmarkConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    model: PatchTSTModelConfig
    training: PatchTSTTrainingConfig
    output_dir: str


def load_patchtst_benchmark_config(path: str | Path) -> PatchTSTBenchmarkConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model_raw = raw.get("model", {})
    training_raw = raw.get("training", {})
    return PatchTSTBenchmarkConfig(
        seed=int(raw.get("seed", 42)),
        data=BenchmarkDataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        model=PatchTSTModelConfig(
            max_steps=int(model_raw.get("max_steps", 300)),
            learning_rate=float(model_raw.get("learning_rate", 1e-4)),
            batch_size=int(model_raw.get("batch_size", 32)),
            windows_batch_size=int(model_raw.get("windows_batch_size", 1024)),
            inference_windows_batch_size=int(model_raw.get("inference_windows_batch_size", 1024)),
            step_size=int(model_raw.get("step_size", 288)),
            scaler_type=str(model_raw.get("scaler_type", "standard")),
            encoder_layers=int(model_raw.get("encoder_layers", 3)),
            n_heads=int(model_raw.get("n_heads", 8)),
            hidden_size=int(model_raw.get("hidden_size", 128)),
            linear_hidden_size=int(model_raw.get("linear_hidden_size", 256)),
            dropout=float(model_raw.get("dropout", 0.2)),
            fc_dropout=float(model_raw.get("fc_dropout", 0.2)),
            head_dropout=float(model_raw.get("head_dropout", 0.0)),
            attn_dropout=float(model_raw.get("attn_dropout", 0.0)),
            patch_len=int(model_raw.get("patch_len", 16)),
            stride=int(model_raw.get("stride", 8)),
            revin=bool(model_raw.get("revin", True)),
            revin_affine=bool(model_raw.get("revin_affine", False)),
            revin_subtract_last=bool(model_raw.get("revin_subtract_last", True)),
            activation=str(model_raw.get("activation", "gelu")),
            res_attention=bool(model_raw.get("res_attention", True)),
            batch_normalization=bool(model_raw.get("batch_normalization", False)),
            learn_pos_embed=bool(model_raw.get("learn_pos_embed", True)),
            val_check_steps=int(model_raw.get("val_check_steps", 50)),
            early_stop_patience_steps=int(model_raw.get("early_stop_patience_steps", -1)),
        ),
        training=PatchTSTTrainingConfig(
            device=str(training_raw.get("device", "auto")),
            slices=tuple(str(value) for value in training_raw.get("slices", [])),
            params_path=training_raw.get("params_path"),
        ),
        output_dir=str(raw.get("output", {}).get("output_dir", "forecasting/experiments/runs/patchtst_benchmark")),
    )


def _trainer_kwargs(device: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    if device == "cuda":
        kwargs.update({"accelerator": "gpu", "devices": 1})
    elif device == "cpu":
        kwargs.update({"accelerator": "cpu"})
    else:
        kwargs.update({"accelerator": "auto"})
    return kwargs


def _parameter_count(model: Any) -> int | None:
    try:
        return int(sum(param.numel() for param in model.parameters()))
    except Exception:
        return None


def _load_best_params(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return dict(raw.get("best_params", raw))


def _apply_params(model: PatchTSTModelConfig, params: dict[str, Any]) -> PatchTSTModelConfig:
    if not params:
        return model
    return PatchTSTModelConfig(
        max_steps=int(params.get("max_steps", model.max_steps)),
        learning_rate=float(params.get("learning_rate", model.learning_rate)),
        batch_size=int(params.get("batch_size", model.batch_size)),
        windows_batch_size=int(params.get("windows_batch_size", model.windows_batch_size)),
        inference_windows_batch_size=int(params.get("inference_windows_batch_size", model.inference_windows_batch_size)),
        step_size=int(params.get("step_size", model.step_size)),
        scaler_type=str(params.get("scaler_type", model.scaler_type)),
        encoder_layers=int(params.get("encoder_layers", model.encoder_layers)),
        n_heads=int(params.get("n_heads", model.n_heads)),
        hidden_size=int(params.get("hidden_size", model.hidden_size)),
        linear_hidden_size=int(params.get("linear_hidden_size", model.linear_hidden_size)),
        dropout=float(params.get("dropout", model.dropout)),
        fc_dropout=float(params.get("fc_dropout", model.fc_dropout)),
        head_dropout=float(params.get("head_dropout", model.head_dropout)),
        attn_dropout=float(params.get("attn_dropout", model.attn_dropout)),
        patch_len=int(params.get("patch_len", model.patch_len)),
        stride=int(params.get("stride", model.stride)),
        revin=bool(params.get("revin", model.revin)),
        revin_affine=bool(params.get("revin_affine", model.revin_affine)),
        revin_subtract_last=bool(params.get("revin_subtract_last", model.revin_subtract_last)),
        activation=str(params.get("activation", model.activation)),
        res_attention=bool(params.get("res_attention", model.res_attention)),
        batch_normalization=bool(params.get("batch_normalization", model.batch_normalization)),
        learn_pos_embed=bool(params.get("learn_pos_embed", model.learn_pos_embed)),
        val_check_steps=int(params.get("val_check_steps", model.val_check_steps)),
        early_stop_patience_steps=int(params.get("early_stop_patience_steps", model.early_stop_patience_steps)),
    )


def _make_model(config: PatchTSTBenchmarkConfig, alias: str):
    from neuralforecast.models import PatchTST

    model = config.model
    return PatchTST(
        h=config.backtest.horizon,
        input_size=config.backtest.input_size,
        encoder_layers=model.encoder_layers,
        n_heads=model.n_heads,
        hidden_size=model.hidden_size,
        linear_hidden_size=model.linear_hidden_size,
        dropout=model.dropout,
        fc_dropout=model.fc_dropout,
        head_dropout=model.head_dropout,
        attn_dropout=model.attn_dropout,
        patch_len=model.patch_len,
        stride=model.stride,
        revin=model.revin,
        revin_affine=model.revin_affine,
        revin_subtract_last=model.revin_subtract_last,
        activation=model.activation,
        res_attention=model.res_attention,
        batch_normalization=model.batch_normalization,
        learn_pos_embed=model.learn_pos_embed,
        max_steps=model.max_steps,
        learning_rate=model.learning_rate,
        batch_size=model.batch_size,
        windows_batch_size=model.windows_batch_size,
        inference_windows_batch_size=model.inference_windows_batch_size,
        step_size=model.step_size,
        scaler_type=model.scaler_type,
        val_check_steps=model.val_check_steps,
        early_stop_patience_steps=model.early_stop_patience_steps,
        random_seed=config.seed,
        alias=alias,
        **_trainer_kwargs(config.training.device),
    )


def _training_frame(series_list: list[pd.DataFrame], fold) -> pd.DataFrame:
    rows = []
    for sub in series_list:
        train = sub.iloc[: fold.train_end_idx + 1, :][["unique_id", "ds", "y"]].copy()
        train["y"] = np.log1p(np.maximum(0.0, train["y"].to_numpy(dtype=float)))
        rows.append(train)
    return pd.concat(rows, ignore_index=True)


def _fit_predict_patchtst(train_df: pd.DataFrame, config: PatchTSTBenchmarkConfig, alias: str) -> tuple[pd.DataFrame, float, float, int | None]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    try:
        import torch

        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass
    from neuralforecast import NeuralForecast

    model = _make_model(config, alias)
    nf = NeuralForecast(models=[model], freq=config.data.frequency)
    train_start = time.perf_counter()
    nf.fit(df=train_df)
    train_seconds = time.perf_counter() - train_start
    infer_start = time.perf_counter()
    forecast = nf.predict()
    inference_seconds = time.perf_counter() - infer_start
    return forecast, train_seconds, inference_seconds, _parameter_count(nf.models[0])


def run_patchtst_benchmark(config: PatchTSTBenchmarkConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
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
    slice_names = sorted(panel["slice"].unique())
    if config.training.slices:
        requested = set(config.training.slices)
        slice_names = [slice_name for slice_name in slice_names if slice_name in requested]
        missing = sorted(requested - set(slice_names))
        if missing:
            raise ValueError(f"Unknown slices requested: {missing}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = _apply_params(config.model, _load_best_params(config.training.params_path))
    config = PatchTSTBenchmarkConfig(
        seed=config.seed,
        data=config.data,
        backtest=config.backtest,
        model=model_config,
        training=config.training,
        output_dir=config.output_dir,
    )
    prediction_rows = []
    timing_rows = []
    metadata_rows = []
    scales = {}

    for fold in folds:
        for slice_name in slice_names:
            print(
                f"[PatchTST] fold={fold.fold + 1}/{len(folds)} slice={slice_name} started",
                flush=True,
            )
            series_list = _slice_series(series_map, str(slice_name))
            train_df = _training_frame(series_list, fold)
            alias = "patchtst"
            scale = _seasonal_scale(series_list, fold.train_end_idx, period=144)
            scales[(fold.fold, str(slice_name))] = scale
            forecast, train_seconds, inference_seconds, parameter_count = _fit_predict_patchtst(train_df, config, alias=alias)
            print(
                f"[PatchTST] fold={fold.fold + 1}/{len(folds)} slice={slice_name} "
                f"done train={train_seconds:.2f}s infer={inference_seconds:.2f}s",
                flush=True,
            )
            forecast = forecast.rename(columns={alias: "y_pred"})
            forecast["y_pred"] = np.maximum(0.0, np.expm1(forecast["y_pred"].to_numpy(dtype=float)))
            forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
            timing_rows.append(
                {
                    "model": "patchtst",
                    "fold": fold.fold,
                    "slice": str(slice_name),
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "train_rows": len(train_df),
                    "eval_series": forecast["unique_id"].nunique(),
                    "device": config.training.device,
                }
            )
            metadata_rows.append(
                {
                    "model": "patchtst",
                    "fold": fold.fold,
                    "slice": str(slice_name),
                    "parameter_count": parameter_count,
                    "implementation": "neuralforecast.models.PatchTST",
                    "training_scope": "per_slice",
                    "trained_models": 1,
                }
            )
            for unique_id, pred_sub in forecast.groupby("unique_id", sort=False):
                original = series_map[str(unique_id)]
                target = original.iloc[fold.target_start_idx : fold.target_end_idx + 1]
                pred_values = pred_sub.sort_values("ds")["y_pred"].to_numpy(dtype=float)
                if len(pred_values) != config.backtest.horizon:
                    raise ValueError(f"Expected {config.backtest.horizon} predictions for {unique_id}, got {len(pred_values)}")
                for step_idx in range(config.backtest.horizon):
                    prediction_rows.append(
                        {
                            "fold": fold.fold,
                            "model": "patchtst",
                            "training_scope": "per_slice",
                            "trained_slice": str(slice_name),
                            "unique_id": str(unique_id),
                            "slice": str(slice_name),
                            "origin_timestamp": fold.train_end,
                            "timestamp": target["ds"].iloc[step_idx],
                            "horizon": step_idx + 1,
                            "y_true": float(target["y"].iloc[step_idx]),
                            "y_pred": float(pred_values[step_idx]),
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
                "model": config.model.__dict__,
                "training": {
                    "device": config.training.device,
                    "slices": list(config.training.slices),
                    "params_path": config.training.params_path,
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
                "training_scope": "per_slice",
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
