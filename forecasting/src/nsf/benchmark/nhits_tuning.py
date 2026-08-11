"""Optuna tuning for per-slice N-HiTS forecasting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nsf.benchmark.deterministic import BenchmarkDataConfig, _prepare_panel, _series_map, _validate_dense
from nsf.benchmark.nhits_benchmark import NHITSBenchmarkConfig, NHITSModelConfig, NHITSTrainingConfig, _fit_predict_nhits, _training_frame
from nsf.benchmark.lstm_tuning import _seasonal_scale, _slice_series
from nsf.config import BacktestConfig
from nsf.data.loading import read_panel
from nsf.evaluation.deterministic import mae, mase, rmse, wape
from nsf.splitting.panel_folds import make_panel_folds
from nsf.utils.io import ensure_parent
from nsf.utils.seed import set_global_seed


class _InnerFold:
    def __init__(self, train_end_idx: int):
        self.train_start_idx = 0
        self.train_end_idx = train_end_idx


def load_nhits_tuning_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {
        "seed": int(raw.get("seed", 42)),
        "data": BenchmarkDataConfig(**raw["data"]),
        "backtest": BacktestConfig(**raw["backtest"]),
        "tuning": dict(raw.get("tuning", {})),
        "base_model": dict(raw.get("base_model", {})),
        "training": dict(raw.get("training", {})),
        "output_dir": str(raw.get("output", {}).get("output_dir", "forecasting/experiments/runs/nhits_tuning")),
    }


def _suggest_params(trial, tuning: dict[str, Any]) -> dict[str, Any]:
    max_steps_range = tuning.get("max_steps_range", [100, 500])
    lr_range = tuning.get("learning_rate_range", [1e-4, 5e-3])
    return {
        "max_steps": trial.suggest_int("max_steps", int(max_steps_range[0]), int(max_steps_range[1])),
        "learning_rate": trial.suggest_float("learning_rate", float(lr_range[0]), float(lr_range[1]), log=True),
        "batch_size": trial.suggest_categorical("batch_size", tuning.get("batch_size_choices", [16, 32, 64])),
        "windows_batch_size": trial.suggest_categorical("windows_batch_size", tuning.get("windows_batch_size_choices", [256, 512, 1024])),
        "step_size": trial.suggest_categorical("step_size", tuning.get("step_size_choices", [144, 288, 576])),
        "mlp_width": trial.suggest_categorical("mlp_width", tuning.get("mlp_width_choices", [128, 256, 512])),
        "n_blocks": trial.suggest_categorical("n_blocks", tuning.get("n_blocks_choices", [1, 2])),
        "dropout_prob_theta": trial.suggest_float("dropout_prob_theta", 0.0, float(tuning.get("max_dropout", 0.2))),
    }


def _model_from_trial(base: dict[str, Any], params: dict[str, Any]) -> NHITSModelConfig:
    width = int(params["mlp_width"])
    n_blocks = int(params["n_blocks"])
    return NHITSModelConfig(
        max_steps=int(params["max_steps"]),
        learning_rate=float(params["learning_rate"]),
        batch_size=int(params["batch_size"]),
        windows_batch_size=int(params["windows_batch_size"]),
        step_size=int(params["step_size"]),
        scaler_type=str(base.get("scaler_type", "standard")),
        stack_types=tuple(base.get("stack_types", ["identity", "identity", "identity"])),
        n_blocks=(n_blocks, n_blocks, n_blocks),
        mlp_units=((width, width), (width, width), (width, width)),
        n_pool_kernel_size=tuple(int(value) for value in base.get("n_pool_kernel_size", [2, 2, 1])),
        n_freq_downsample=tuple(int(value) for value in base.get("n_freq_downsample", [4, 2, 1])),
        dropout_prob_theta=float(params["dropout_prob_theta"]),
        val_check_steps=int(base.get("val_check_steps", 50)),
        early_stop_patience_steps=int(base.get("early_stop_patience_steps", -1)),
    )


def _score_predictions(forecast: pd.DataFrame, series_list: list[pd.DataFrame], target_start_idx: int, horizon: int, objective: str, scale: float) -> float:
    true_rows = []
    pred_rows = []
    for sub in series_list:
        unique_id = str(sub["unique_id"].iloc[0])
        target = sub.iloc[target_start_idx : target_start_idx + horizon]
        pred = forecast[forecast["unique_id"].astype(str) == unique_id].sort_values("ds")
        if len(pred) != horizon:
            raise ValueError(f"Expected {horizon} validation predictions for {unique_id}, got {len(pred)}")
        true_rows.extend(target["y"].to_numpy(dtype=float).tolist())
        pred_rows.extend(np.maximum(0.0, np.expm1(pred["nhits"].to_numpy(dtype=float))).tolist())
    y_true = np.asarray(true_rows, dtype=float)
    y_pred = np.asarray(pred_rows, dtype=float)
    metric = objective.upper()
    if metric == "RMSE":
        return rmse(y_true, y_pred)
    if metric == "MAE":
        return mae(y_true, y_pred)
    if metric == "WAPE":
        return wape(y_true, y_pred)
    return mase(y_true, y_pred, scale)


def _inner_train_end_indices(fold, input_size: int, horizon: int, n_folds: int, fold_stride: int) -> list[int]:
    latest = fold.train_end_idx - horizon
    earliest = input_size - 1
    indices = []
    for offset in reversed(range(n_folds)):
        train_end_idx = latest - offset * fold_stride
        if train_end_idx < earliest:
            continue
        indices.append(train_end_idx)
    if not indices:
        raise ValueError("Not enough history for N-HiTS internal validation folds")
    return indices


def tune_nhits_by_slice(config: dict[str, Any]) -> dict[str, Path]:
    set_global_seed(config["seed"])
    import optuna

    panel = _prepare_panel(read_panel(config["data"].panel_csv))
    timestamps = pd.DatetimeIndex(sorted(panel["ds"].unique()))
    series_map = _series_map(panel)
    _validate_dense(series_map, timestamps)
    folds = make_panel_folds(
        timestamps=timestamps,
        input_size=config["backtest"].input_size,
        horizon=config["backtest"].horizon,
        n_folds=config["backtest"].n_folds,
        fold_stride=config["backtest"].fold_stride,
        expanding=config["backtest"].expanding,
    )
    tuning = config["tuning"]
    objective_name = str(tuning.get("objective", "MASE")).upper()
    n_trials = int(tuning.get("n_trials", 20))
    internal_validation_folds = int(tuning.get("internal_validation_folds", 1))
    internal_fold_stride = int(tuning.get("internal_fold_stride", config["backtest"].fold_stride))
    tuned_fold = folds[0]
    inner_train_end_indices = _inner_train_end_indices(
        tuned_fold,
        input_size=config["backtest"].input_size,
        horizon=config["backtest"].horizon,
        n_folds=internal_validation_folds,
        fold_stride=internal_fold_stride,
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_names = sorted(panel["slice"].unique())
    requested = tuple(str(value) for value in config["training"].get("slices", []))
    if requested:
        slice_names = [slice_name for slice_name in slice_names if slice_name in set(requested)]
    trial_rows = []
    best_params_by_slice = {}
    base_training = NHITSTrainingConfig(device=str(config["training"].get("device", "auto")), slices=())

    for slice_name in slice_names:
        series_list = _slice_series(series_map, str(slice_name))

        def objective(trial):
            params = _suggest_params(trial, tuning)
            model_config = _model_from_trial(config["base_model"], params)
            scores = []
            total_train_seconds = 0.0
            total_inference_seconds = 0.0
            last_parameter_count = None
            for inner_fold_idx, inner_train_end_idx in enumerate(inner_train_end_indices):
                scale = _seasonal_scale(series_list, inner_train_end_idx, period=144)
                train_df = _training_frame(series_list, _InnerFold(inner_train_end_idx))
                bench_config = NHITSBenchmarkConfig(
                    seed=config["seed"],
                    data=config["data"],
                    backtest=config["backtest"],
                    model=model_config,
                    training=base_training,
                    output_dir=config["output_dir"],
                )
                forecast, train_seconds, inference_seconds, parameter_count = _fit_predict_nhits(train_df, bench_config, alias="nhits")
                score = _score_predictions(forecast, series_list, inner_train_end_idx + 1, config["backtest"].horizon, objective_name, scale)
                scores.append(score)
                total_train_seconds += train_seconds
                total_inference_seconds += inference_seconds
                last_parameter_count = parameter_count
                row = {
                    "slice": str(slice_name),
                    "trial": trial.number,
                    "inner_fold": inner_fold_idx,
                    "inner_train_end_idx": inner_train_end_idx,
                    "objective": objective_name,
                    "score": score,
                    "mean_score_so_far": float(np.mean(scores)),
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "train_rows": len(train_df),
                    "parameter_count": parameter_count,
                    "device": base_training.device,
                }
                row.update(params)
                trial_rows.append(row)
            mean_score = float(np.mean(scores))
            trial_rows.append(
                {
                    "slice": str(slice_name),
                    "trial": trial.number,
                    "inner_fold": "mean",
                    "inner_train_end_idx": "",
                    "objective": objective_name,
                    "score": mean_score,
                    "mean_score_so_far": mean_score,
                    "train_seconds": total_train_seconds,
                    "inference_seconds": total_inference_seconds,
                    "train_rows": "",
                    "parameter_count": last_parameter_count,
                    "device": base_training.device,
                    **params,
                }
            )
            return mean_score

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config["seed"]))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params_by_slice[str(slice_name)] = dict(study.best_params)

    paths = {
        "trials": output_dir / "nhits_tuning_trials.csv",
        "best_params": output_dir / "nhits_best_params_by_slice.yaml",
        "run_meta": output_dir / "run_meta.json",
        "resolved_config": output_dir / "resolved_config.yaml",
    }
    pd.DataFrame(trial_rows).to_csv(ensure_parent(paths["trials"]), index=False)
    paths["best_params"].write_text(yaml.safe_dump({"best_params_by_slice": best_params_by_slice}, sort_keys=False), encoding="utf-8")
    paths["run_meta"].write_text(
        json.dumps(
            {
                "slices": len(best_params_by_slice),
                "n_trials": n_trials,
                "objective": objective_name,
                "tuned_fold": tuned_fold.fold,
                "internal_validation_folds": len(inner_train_end_indices),
                "internal_fold_stride": internal_fold_stride,
                "inner_train_end_indices": inner_train_end_indices,
                "validation_horizon": config["backtest"].horizon,
                "input_size": config["backtest"].input_size,
                "horizon": config["backtest"].horizon,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["resolved_config"].write_text(
        yaml.safe_dump(
            {
                "seed": config["seed"],
                "data": config["data"].__dict__,
                "backtest": config["backtest"].__dict__,
                "tuning": tuning,
                "base_model": config["base_model"],
                "training": config["training"],
                "output": {"output_dir": config["output_dir"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return paths
