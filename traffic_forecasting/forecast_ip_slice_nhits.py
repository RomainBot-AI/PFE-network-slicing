#!/usr/bin/env python3
"""Forecast traffic from a slice/ip_id panel with N-HiTS and aggregate by slice."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from traffic_forecasting.common import (
        SLICES,
        bias,
        ensure_parent,
        mae,
        rmse,
        smape,
        under_over_error,
    )
except ModuleNotFoundError:
    from common import (
        SLICES,
        bias,
        ensure_parent,
        mae,
        rmse,
        smape,
        under_over_error,
    )


@dataclass(frozen=True)
class PanelNHITSConfig:
    input_csv: str
    predictions_csv: str
    metrics_csv: str
    test_size: int
    horizons: tuple[int, ...]
    freq: str
    input_size: int
    step_size: int
    max_steps: int
    train_tail: int | None
    val_size: int
    batch_size: int
    windows_batch_size: int
    seed: int
    device: str
    training_scope: str
    model_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="traffic_forecasting/data/ip_slice_traffic_10min_long.csv",
        help="Long panel CSV produced by build_ip_slice_series.py.",
    )
    parser.add_argument("--predictions-csv", default="traffic_forecasting/outputs/predictions_ip_slice_nhits.csv")
    parser.add_argument("--metrics-csv", default="traffic_forecasting/reports/metrics_ip_slice_nhits.csv")
    parser.add_argument("--test-size", type=int, default=144)
    parser.add_argument("--horizons", default="1,6,12")
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--input-size", type=int, default=1008)
    parser.add_argument("--step-size", type=int, default=1, help="Stride between training windows, in time steps.")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--train-tail", type=int, default=0, help="Use the last N timestamps. Use 0 for full history.")
    parser.add_argument("--val-size", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--windows-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--training-scope",
        choices=("global", "per-slice"),
        default="global",
        help="Train one global model on every series or one independent model per slice.",
    )
    parser.add_argument(
        "--model-name",
        default="nhits_panel",
        help="Model name written to prediction and metric CSV files.",
    )
    return parser.parse_args()


def parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("Expected at least one integer")
    return parsed


def read_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"unique_id": str, "slice": str})
    required = {"unique_id", "ds", "y", "slice"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True).dt.tz_convert(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["unique_id"] = df["unique_id"].astype(str)
    df["slice"] = df["slice"].astype(str)
    df = df.dropna(subset=["unique_id", "ds", "y", "slice"])
    df = df[df["slice"].isin(SLICES)].copy()
    if df.empty:
        raise ValueError(f"No valid panel rows found in {path}")
    return df.sort_values(["unique_id", "ds"])


def split_by_timestamp(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    timestamps = pd.DatetimeIndex(sorted(df["ds"].unique()))
    if len(timestamps) <= test_size:
        raise ValueError(f"Timestamp count {len(timestamps)} must be greater than test_size {test_size}")
    test_timestamps = timestamps[-test_size:]
    train_timestamps = timestamps[:-test_size]
    train = df[df["ds"].isin(train_timestamps)].copy()
    test = df[df["ds"].isin(test_timestamps)].copy()
    return train, test, test_timestamps


def apply_train_tail(train: pd.DataFrame, train_tail: int | None) -> pd.DataFrame:
    if not train_tail or train_tail <= 0:
        return train
    timestamps = pd.DatetimeIndex(sorted(train["ds"].unique()))
    keep = timestamps[-train_tail:]
    return train[train["ds"].isin(keep)].copy()


def to_model_df(train: pd.DataFrame) -> pd.DataFrame:
    model_df = train[["unique_id", "ds", "y"]].copy()
    model_df["y"] = np.log1p(np.maximum(0.0, model_df["y"].to_numpy(dtype=float)))
    return model_df


def count_training_windows(train: pd.DataFrame, input_size: int, horizon: int, step_size: int) -> pd.DataFrame:
    counts = train.groupby(["unique_id", "slice"], as_index=False)["ds"].nunique().rename(columns={"ds": "points"})
    usable = counts["points"] - input_size - horizon + 1
    counts["windows"] = np.where(usable > 0, ((usable - 1) // step_size) + 1, 0).astype(int)
    return counts


def aggregate_truth(test: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
    values = test[test["ds"] == timestamp].groupby("slice")["y"].sum()
    return values.reindex(SLICES, fill_value=0.0)


def make_nhits_model(config: PanelNHITSConfig, max_horizon: int, accelerator: str, alias: str):
    from neuralforecast.models import NHITS

    return NHITS(
        h=max_horizon,
        input_size=config.input_size,
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        windows_batch_size=config.windows_batch_size,
        scaler_type="robust",
        step_size=config.step_size,
        random_seed=config.seed,
        alias=alias,
        n_blocks=[1, 1, 1],
        mlp_units=[[128, 128], [128, 128], [128, 128]],
        accelerator=accelerator,
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def fit_predict_scope(
    train: pd.DataFrame,
    config: PanelNHITSConfig,
    max_horizon: int,
    accelerator: str,
    alias: str,
    label: str,
    freq: str,
) -> pd.DataFrame:
    from neuralforecast import NeuralForecast

    metadata_cols = [col for col in train.columns if col not in {"ds", "y"}]
    metadata = train[metadata_cols].drop_duplicates("unique_id")
    model_df = to_model_df(train)
    val_size = min(config.val_size, max(0, model_df["ds"].nunique() - config.input_size - max_horizon))
    window_counts = count_training_windows(train, config.input_size, max_horizon, config.step_size)
    print(
        "Window audit: "
        f"scope={label} "
        f"train_points={model_df['ds'].nunique()} "
        f"input_size={config.input_size} "
        f"horizon={max_horizon} "
        f"step_size={config.step_size} "
        f"series={window_counts['unique_id'].nunique()} "
        f"train_windows={int(window_counts['windows'].sum())}"
    )

    if int(window_counts["windows"].sum()) <= 0:
        raise ValueError(f"No training windows available for scope={label}")

    model = make_nhits_model(config, max_horizon, accelerator, alias)
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=model_df, val_size=val_size, verbose=False)
    forecast = nf.predict(verbose=False).reset_index()
    forecast = forecast.merge(metadata, on="unique_id", how="left")
    forecast["y_pred"] = np.maximum(0.0, np.expm1(forecast[alias].to_numpy(dtype=float)))
    forecast = forecast.sort_values(["unique_id", "ds"])
    forecast["step"] = forecast.groupby("unique_id").cumcount() + 1
    return forecast


def run(config: PanelNHITSConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This script requires neuralforecast and torch. Install with: "
            ".venv/bin/pip install neuralforecast"
        ) from exc

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    use_cuda = config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available())
    accelerator = "gpu" if use_cuda else "cpu"
    if use_cuda:
        torch.set_float32_matmul_precision("high")

    panel = read_panel(config.input_csv)
    max_horizon = max(config.horizons)
    train, test, test_timestamps = split_by_timestamp(panel, config.test_size)
    if len(test_timestamps) < max_horizon:
        raise ValueError("Test set is shorter than max horizon")
    train = apply_train_tail(train, config.train_tail)

    print(f"Split audit: test_start={test_timestamps[0]} training_scope={config.training_scope}")

    if config.training_scope == "global":
        forecasts = [
            fit_predict_scope(
                train=train,
                config=config,
                max_horizon=max_horizon,
                accelerator=accelerator,
                alias="nhits_ip_slice",
                label="global",
                freq=config.freq,
            )
        ]
        model_name = config.model_name
    else:
        forecasts = []
        for slice_name in SLICES:
            slice_train = train[train["slice"] == slice_name].copy()
            forecasts.append(
                fit_predict_scope(
                    train=slice_train,
                    config=config,
                    max_horizon=max_horizon,
                    accelerator=accelerator,
                    alias="nhits_ip_slice",
                    label=slice_name,
                    freq=config.freq,
                )
            )
        model_name = f"{config.model_name}_per_slice"

    forecast = pd.concat(forecasts, ignore_index=True)

    origin_timestamp = train["ds"].max()
    prediction_rows = []
    metric_rows = []

    for horizon in config.horizons:
        target_timestamp = test_timestamps[horizon - 1]
        predicted = (
            forecast[forecast["step"] == horizon]
            .groupby("slice")["y_pred"]
            .sum()
            .reindex(SLICES, fill_value=0.0)
        )
        truth = aggregate_truth(test, target_timestamp)

        for slice_name in SLICES:
            true_value = float(truth.loc[slice_name])
            pred_value = float(predicted.loc[slice_name])
            y_true = np.asarray([true_value], dtype=float)
            y_pred = np.asarray([pred_value], dtype=float)
            under, over = under_over_error(y_true, y_pred)

            prediction_rows.append(
                {
                    "timestamp": target_timestamp,
                    "origin_timestamp": origin_timestamp,
                    "model": model_name,
                    "slice": slice_name,
                    "horizon": horizon,
                    "evaluation": f"direct_panel_aggregate_{config.training_scope}",
                    "y_true": true_value,
                    "y_pred": pred_value,
                }
            )
            metric_rows.append(
                {
                    "model": model_name,
                    "slice": slice_name,
                    "horizon": horizon,
                    "evaluation": f"direct_panel_aggregate_{config.training_scope}",
                    "MAE": mae(y_true, y_pred),
                    "RMSE": rmse(y_true, y_pred),
                    "sMAPE": smape(y_true, y_pred),
                    "bias": bias(y_true, y_pred),
                    "under_prediction_error": under,
                    "over_prediction_error": over,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    predictions_path = ensure_parent(config.predictions_csv)
    metrics_path = ensure_parent(config.metrics_csv)
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    return predictions, metrics


def main() -> None:
    args = parse_args()
    config = PanelNHITSConfig(
        input_csv=args.input_csv,
        predictions_csv=args.predictions_csv,
        metrics_csv=args.metrics_csv,
        test_size=args.test_size,
        horizons=parse_csv_ints(args.horizons),
        freq=args.freq,
        input_size=args.input_size,
        step_size=args.step_size,
        max_steps=args.max_steps,
        train_tail=None if args.train_tail == 0 else args.train_tail,
        val_size=args.val_size,
        batch_size=args.batch_size,
        windows_batch_size=args.windows_batch_size,
        seed=args.seed,
        device=args.device,
        training_scope=args.training_scope,
        model_name=args.model_name,
    )
    predictions, metrics = run(config)
    print(f"Wrote {config.predictions_csv} rows={len(predictions)}")
    print(f"Wrote {config.metrics_csv} rows={len(metrics)}")
    print(metrics.sort_values(["slice", "horizon"]).to_string(index=False))


if __name__ == "__main__":
    main()
