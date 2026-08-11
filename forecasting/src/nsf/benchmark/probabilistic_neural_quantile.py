"""NeuralForecast quantile benchmarks for N-HiTS and PatchTST."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from nsf.benchmark.deterministic import (
    BenchmarkDataConfig,
    BenchmarkOutputConfig,
    _benchmark_summary,
    _benchmark_summary_by_slice,
    _metric_rows,
    _prepare_panel,
    _series_map,
    _summary_metrics,
    _validate_dense,
)
from nsf.benchmark.lstm_tuning import _seasonal_scale, _slice_series
from nsf.benchmark.nhits_benchmark import (
    NHITSBenchmarkConfig,
    NHITSModelConfig,
    NHITSTrainingConfig,
    _apply_tuned_params as _apply_nhits_params,
    _load_best_params as _load_nhits_params,
    load_nhits_benchmark_config,
)
from nsf.benchmark.patchtst_benchmark import (
    PatchTSTBenchmarkConfig,
    PatchTSTModelConfig,
    PatchTSTTrainingConfig,
    _apply_params as _apply_patchtst_params,
    _load_best_params as _load_patchtst_params,
    load_patchtst_benchmark_config,
)
from nsf.benchmark.probabilistic_lightgbm import (
    _enforce_monotonic_quantiles,
    _probabilistic_metric_rows,
    _probabilistic_summary,
    _probabilistic_summary_by_slice,
    _quantile_column,
    _summary_probabilistic_metrics,
    _validate_quantiles,
)
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.splitting.panel_folds import folds_to_frame, leakage_audit, make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed

Family = Literal["nhits", "patchtst"]


@dataclass(frozen=True)
class ProbabilisticNeuralModelConfig:
    family: Family
    name: str
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    params: NHITSModelConfig | PatchTSTModelConfig | None = None


@dataclass(frozen=True)
class ProbabilisticNeuralTrainingConfig:
    device: str = "auto"
    slices: tuple[str, ...] = field(default_factory=tuple)
    params_path: str | None = None
    seasonal_scale_period: int = 144


@dataclass(frozen=True)
class ProbabilisticNeuralConfig:
    seed: int
    data: BenchmarkDataConfig
    backtest: BacktestConfig
    model: ProbabilisticNeuralModelConfig
    training: ProbabilisticNeuralTrainingConfig
    output: BenchmarkOutputConfig


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


def _load_base_config(path: str | Path, family: Family):
    if family == "nhits":
        return load_nhits_benchmark_config(path)
    if family == "patchtst":
        return load_patchtst_benchmark_config(path)
    raise ValueError(f"Unsupported probabilistic neural family: {family}")


def load_probabilistic_neural_config(path: str | Path) -> ProbabilisticNeuralConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model_raw = raw.get("model", {})
    family = str(model_raw.get("family", "")).lower()
    if family not in {"nhits", "patchtst"}:
        raise ValueError("model.family must be 'nhits' or 'patchtst'")
    base = _load_base_config(path, family)  # type: ignore[arg-type]
    training_raw = raw.get("training", {})
    output_raw = raw.get("output", {})
    return ProbabilisticNeuralConfig(
        seed=base.seed,
        data=base.data,
        backtest=base.backtest,
        model=ProbabilisticNeuralModelConfig(
            family=family,  # type: ignore[arg-type]
            name=str(model_raw.get("name", f"{family}_quantile")),
            quantiles=tuple(float(value) for value in model_raw.get("quantiles", [0.1, 0.5, 0.9])),
            params=base.model,
        ),
        training=ProbabilisticNeuralTrainingConfig(
            device=str(training_raw.get("device", "auto")),
            slices=tuple(str(value) for value in training_raw.get("slices", [])),
            params_path=training_raw.get("params_path"),
            seasonal_scale_period=int(training_raw.get("seasonal_scale_period", 144)),
        ),
        output=BenchmarkOutputConfig(run_dir=str(output_raw.get("run_dir", output_raw.get("output_dir", f"forecasting/experiments/runs/probabilistic_{family}")))),
    )


def _make_nhits(config: ProbabilisticNeuralConfig, model_config: NHITSModelConfig, alias: str):
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NHITS

    return NHITS(
        h=config.backtest.horizon,
        input_size=config.backtest.input_size,
        max_steps=model_config.max_steps,
        learning_rate=model_config.learning_rate,
        batch_size=model_config.batch_size,
        windows_batch_size=model_config.windows_batch_size,
        step_size=model_config.step_size,
        scaler_type=model_config.scaler_type,
        stack_types=list(model_config.stack_types),
        n_blocks=list(model_config.n_blocks),
        mlp_units=[list(row) for row in model_config.mlp_units],
        n_pool_kernel_size=list(model_config.n_pool_kernel_size),
        n_freq_downsample=list(model_config.n_freq_downsample),
        dropout_prob_theta=model_config.dropout_prob_theta,
        loss=MQLoss(quantiles=list(config.model.quantiles)),
        val_check_steps=model_config.val_check_steps,
        early_stop_patience_steps=model_config.early_stop_patience_steps,
        random_seed=config.seed,
        alias=alias,
        **_trainer_kwargs(config.training.device),
    )


def _make_patchtst(config: ProbabilisticNeuralConfig, model_config: PatchTSTModelConfig, alias: str):
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import PatchTST

    return PatchTST(
        h=config.backtest.horizon,
        input_size=config.backtest.input_size,
        encoder_layers=model_config.encoder_layers,
        n_heads=model_config.n_heads,
        hidden_size=model_config.hidden_size,
        linear_hidden_size=model_config.linear_hidden_size,
        dropout=model_config.dropout,
        fc_dropout=model_config.fc_dropout,
        head_dropout=model_config.head_dropout,
        attn_dropout=model_config.attn_dropout,
        patch_len=model_config.patch_len,
        stride=model_config.stride,
        revin=model_config.revin,
        revin_affine=model_config.revin_affine,
        revin_subtract_last=model_config.revin_subtract_last,
        activation=model_config.activation,
        res_attention=model_config.res_attention,
        batch_normalization=model_config.batch_normalization,
        learn_pos_embed=model_config.learn_pos_embed,
        loss=MQLoss(quantiles=list(config.model.quantiles)),
        max_steps=model_config.max_steps,
        learning_rate=model_config.learning_rate,
        batch_size=model_config.batch_size,
        windows_batch_size=model_config.windows_batch_size,
        inference_windows_batch_size=model_config.inference_windows_batch_size,
        step_size=model_config.step_size,
        scaler_type=model_config.scaler_type,
        val_check_steps=model_config.val_check_steps,
        early_stop_patience_steps=model_config.early_stop_patience_steps,
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


def _fit_predict(
    train_df: pd.DataFrame,
    config: ProbabilisticNeuralConfig,
    model_config: NHITSModelConfig | PatchTSTModelConfig,
    alias: str,
) -> tuple[pd.DataFrame, float, float, int | None]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    try:
        import torch

        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass
    from neuralforecast import NeuralForecast

    if config.model.family == "nhits":
        model = _make_nhits(config, model_config, alias)  # type: ignore[arg-type]
    else:
        model = _make_patchtst(config, model_config, alias)  # type: ignore[arg-type]
    nf = NeuralForecast(models=[model], freq=config.data.frequency)
    train_start = time.perf_counter()
    nf.fit(df=train_df)
    train_seconds = time.perf_counter() - train_start
    infer_start = time.perf_counter()
    forecast = nf.predict()
    inference_seconds = time.perf_counter() - infer_start
    return forecast, train_seconds, inference_seconds, _parameter_count(nf.models[0])


def _central_level(quantiles: tuple[float, ...]) -> int:
    lower = min(quantiles)
    upper = max(quantiles)
    if not np.isclose(lower, 1.0 - upper):
        raise ValueError("MQLoss helper expects symmetric lower/upper quantiles")
    return int(round((upper - lower) * 100))


def _forecast_to_quantiles(forecast: pd.DataFrame, alias: str, quantiles: tuple[float, ...]) -> pd.DataFrame:
    level = _central_level(quantiles)
    lower_candidates = (f"{alias}-lo-{float(level):.1f}", f"{alias}-lo-{level}")
    upper_candidates = (f"{alias}-hi-{float(level):.1f}", f"{alias}-hi-{level}")
    median_col = f"{alias}-median"
    lower_col = next((col for col in lower_candidates if col in forecast.columns), None)
    upper_col = next((col for col in upper_candidates if col in forecast.columns), None)
    missing = []
    if lower_col is None:
        missing.append("/".join(lower_candidates))
    if median_col not in forecast.columns:
        missing.append(median_col)
    if upper_col is None:
        missing.append("/".join(upper_candidates))
    if missing:
        raise ValueError(f"Forecast is missing expected quantile columns: {missing}. Available: {forecast.columns.tolist()}")

    assert lower_col is not None
    assert upper_col is not None
    out = forecast[["unique_id", "ds", lower_col, median_col, upper_col]].copy()
    out[_quantile_column(min(quantiles))] = np.maximum(0.0, np.expm1(out[lower_col].to_numpy(dtype=float)))
    out[_quantile_column(0.5)] = np.maximum(0.0, np.expm1(out[median_col].to_numpy(dtype=float)))
    out[_quantile_column(max(quantiles))] = np.maximum(0.0, np.expm1(out[upper_col].to_numpy(dtype=float)))
    return out[["unique_id", "ds", *[_quantile_column(q) for q in quantiles]]]


def _model_for_slice(config: ProbabilisticNeuralConfig, slice_name: str):
    assert config.model.params is not None
    if config.model.family == "nhits":
        best_params = _load_nhits_params(config.training.params_path)
        return _apply_nhits_params(config.model.params, best_params.get(slice_name, {}))  # type: ignore[arg-type]
    params = _load_patchtst_params(config.training.params_path)
    return _apply_patchtst_params(config.model.params, params)  # type: ignore[arg-type]


def run_probabilistic_neural_benchmark(config: ProbabilisticNeuralConfig) -> dict[str, Path]:
    set_global_seed(config.seed)
    quantiles = _validate_quantiles(config.model.quantiles)
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

    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    timing_rows = []
    metadata_rows = []
    scales = {}
    alias = config.model.name

    for fold in folds:
        for slice_name in slice_names:
            print(
                f"[{config.model.name}] fold={fold.fold + 1}/{len(folds)} slice={slice_name} started",
                flush=True,
            )
            series_list = _slice_series(series_map, str(slice_name))
            train_df = _training_frame(series_list, fold)
            scales[(fold.fold, str(slice_name))] = _seasonal_scale(
                series_list,
                fold.train_end_idx,
                period=config.training.seasonal_scale_period,
            )
            model_config = _model_for_slice(config, str(slice_name))
            forecast, train_seconds, inference_seconds, parameter_count = _fit_predict(train_df, config, model_config, alias=alias)
            print(
                f"[{config.model.name}] fold={fold.fold + 1}/{len(folds)} slice={slice_name} "
                f"done train={train_seconds:.2f}s infer={inference_seconds:.2f}s",
                flush=True,
            )
            forecast = _forecast_to_quantiles(forecast, alias=alias, quantiles=quantiles)
            forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
            timing_rows.append(
                {
                    "model": config.model.name,
                    "fold": fold.fold,
                    "slice": str(slice_name),
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "train_rows": len(train_df),
                    "eval_series": forecast["unique_id"].nunique(),
                    "device": config.training.device,
                    "params_source": config.training.params_path or "",
                }
            )
            metadata_rows.append(
                {
                    "model": config.model.name,
                    "fold": fold.fold,
                    "slice": str(slice_name),
                    "parameter_count": parameter_count,
                    "implementation": f"neuralforecast.models.{config.model.family.upper()} + MQLoss",
                    "training_scope": "per_slice",
                    "trained_models": 1,
                    "quantiles": ",".join(str(q) for q in quantiles),
                    "params_source": config.training.params_path or "",
                }
            )
            for unique_id, pred_sub in forecast.groupby("unique_id", sort=False):
                original = series_map[str(unique_id)]
                target = original.iloc[fold.target_start_idx : fold.target_end_idx + 1]
                pred_sub = _enforce_monotonic_quantiles(pred_sub.sort_values("ds"), quantiles)
                if len(pred_sub) != config.backtest.horizon:
                    raise ValueError(f"Expected {config.backtest.horizon} predictions for {unique_id}, got {len(pred_sub)}")
                for step_idx in range(config.backtest.horizon):
                    row = {
                        "fold": fold.fold,
                        "model": config.model.name,
                        "training_scope": "per_slice",
                        "trained_slice": str(slice_name),
                        "unique_id": str(unique_id),
                        "slice": str(slice_name),
                        "origin_timestamp": fold.train_end,
                        "timestamp": target["ds"].iloc[step_idx],
                        "horizon": step_idx + 1,
                        "y_true": float(target["y"].iloc[step_idx]),
                    }
                    for quantile in quantiles:
                        row[_quantile_column(quantile)] = float(pred_sub[_quantile_column(quantile)].iloc[step_idx])
                    prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    median_predictions = predictions.rename(columns={_quantile_column(0.5): "y_pred"})
    deterministic_metrics_by_fold = _metric_rows(median_predictions, scales)
    deterministic_metrics = _summary_metrics(deterministic_metrics_by_fold)
    deterministic_summary = _benchmark_summary(deterministic_metrics)
    deterministic_summary_by_slice = _benchmark_summary_by_slice(deterministic_metrics)
    probabilistic_metrics_by_fold = _probabilistic_metric_rows(predictions, quantiles)
    probabilistic_metrics = _summary_probabilistic_metrics(probabilistic_metrics_by_fold)
    probabilistic_summary = _probabilistic_summary(probabilistic_metrics)
    probabilistic_summary_by_slice = _probabilistic_summary_by_slice(probabilistic_metrics)
    timing = pd.DataFrame(timing_rows)
    metadata = pd.DataFrame(metadata_rows)

    paths = {
        "resolved_config": run_dir / "resolved_config.yaml",
        "run_meta": run_dir / "run_meta.json",
        "folds": run_dir / "folds.csv",
        "leakage_audit": run_dir / "leakage_audit.csv",
        "predictions": run_dir / "predictions_probabilistic.csv",
        "deterministic_metrics_by_fold": run_dir / "metrics_by_fold.csv",
        "deterministic_metrics": run_dir / "metrics.csv",
        "deterministic_summary": run_dir / "benchmark_summary.csv",
        "deterministic_summary_by_slice": run_dir / "benchmark_summary_by_slice.csv",
        "probabilistic_metrics_by_fold": run_dir / "metrics_probabilistic_by_fold.csv",
        "probabilistic_metrics": run_dir / "metrics_probabilistic.csv",
        "probabilistic_summary": run_dir / "benchmark_probabilistic_summary.csv",
        "probabilistic_summary_by_slice": run_dir / "benchmark_probabilistic_summary_by_slice.csv",
        "timing": run_dir / "timing.csv",
        "model_metadata": run_dir / "model_metadata.csv",
    }
    paths["resolved_config"].write_text(
        yaml.safe_dump(
            {
                "seed": config.seed,
                "data": config.data.__dict__,
                "backtest": config.backtest.__dict__,
                "model": {
                    "family": config.model.family,
                    "name": config.model.name,
                    "quantiles": list(config.model.quantiles),
                    "params": config.model.params.__dict__ if config.model.params is not None else {},
                },
                "training": {
                    "device": config.training.device,
                    "slices": list(config.training.slices),
                    "params_path": config.training.params_path,
                    "seasonal_scale_period": config.training.seasonal_scale_period,
                },
                "output": config.output.__dict__,
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
                "quantiles": list(quantiles),
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
    deterministic_metrics_by_fold.to_csv(ensure_parent(paths["deterministic_metrics_by_fold"]), index=False)
    deterministic_metrics.to_csv(ensure_parent(paths["deterministic_metrics"]), index=False)
    deterministic_summary.to_csv(ensure_parent(paths["deterministic_summary"]), index=False)
    deterministic_summary_by_slice.to_csv(ensure_parent(paths["deterministic_summary_by_slice"]), index=False)
    probabilistic_metrics_by_fold.to_csv(ensure_parent(paths["probabilistic_metrics_by_fold"]), index=False)
    probabilistic_metrics.to_csv(ensure_parent(paths["probabilistic_metrics"]), index=False)
    probabilistic_summary.to_csv(ensure_parent(paths["probabilistic_summary"]), index=False)
    probabilistic_summary_by_slice.to_csv(ensure_parent(paths["probabilistic_summary_by_slice"]), index=False)
    timing.to_csv(ensure_parent(paths["timing"]), index=False)
    metadata.to_csv(ensure_parent(paths["model_metadata"]), index=False)
    return paths


def load_probabilistic_nhits_config(path: str | Path) -> ProbabilisticNeuralConfig:
    config = load_probabilistic_neural_config(path)
    if config.model.family != "nhits":
        raise ValueError("Expected model.family: nhits")
    return config


def load_probabilistic_patchtst_config(path: str | Path) -> ProbabilisticNeuralConfig:
    config = load_probabilistic_neural_config(path)
    if config.model.family != "patchtst":
        raise ValueError("Expected model.family: patchtst")
    return config
