#!/usr/bin/env python3
"""Run reproducible forecasts for the 4 network slices."""

from __future__ import annotations

import argparse
import os
import contextlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from traffic_forecasting.common import (
        SLICES,
        bias,
        chronological_split,
        ensure_parent,
        mae,
        read_slice_series,
        rmse,
        smape,
        under_over_error,
    )
except ModuleNotFoundError:
    from common import (
        SLICES,
        bias,
        chronological_split,
        ensure_parent,
        mae,
        read_slice_series,
        rmse,
        smape,
        under_over_error,
    )


@dataclass(frozen=True)
class ForecastConfig:
    input_csv: str
    predictions_csv: str
    metrics_csv: str
    test_size: int
    horizons: tuple[int, ...]
    models: tuple[str, ...]
    seasonal_period: int
    moving_average_window: int
    freq: str
    nhits_input_size: int
    nhits_max_steps: int
    nhits_train_tail: int | None
    nhits_val_size: int
    seed: int
    evaluation_mode: str
    prophet_train_tail: int | None
    lstm_train_tail: int | None
    lstm_window: int
    lstm_hidden_size: int
    lstm_epochs: int
    lstm_batch_size: int
    lstm_learning_rate: float
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="traffic_forecasting/data/slice_traffic_10min.csv",
        help="Wide time-series CSV produced by build_slice_series.py.",
    )
    parser.add_argument("--predictions-csv", default="traffic_forecasting/outputs/predictions.csv")
    parser.add_argument("--metrics-csv", default="traffic_forecasting/reports/metrics.csv")
    parser.add_argument("--test-size", type=int, default=144, help="Number of final 10-minute points used as test set.")
    parser.add_argument("--horizons", default="1,6,12", help="Comma-separated horizons, in time steps.")
    parser.add_argument(
        "--models",
        default="naive,seasonal_naive,moving_average",
        help="Comma-separated model names. Available: naive, seasonal_naive, moving_average, nhits, prophet, lstm.",
    )
    parser.add_argument("--seasonal-period", type=int, default=144, help="One day for 10-minute data.")
    parser.add_argument("--moving-average-window", type=int, default=12)
    parser.add_argument("--freq", default="10min", help="Frequency passed to NeuralForecast models.")
    parser.add_argument("--nhits-input-size", type=int, default=72)
    parser.add_argument("--nhits-max-steps", type=int, default=200)
    parser.add_argument("--nhits-train-tail", type=int, default=5000, help="Use the last N train rows for N-HiTS. Use 0 for all train rows.")
    parser.add_argument("--nhits-val-size", type=int, default=144)
    parser.add_argument("--prophet-train-tail", type=int, default=5000, help="Use the last N train rows for Prophet. Use 0 for all train rows.")
    parser.add_argument("--lstm-train-tail", type=int, default=2000, help="Use the last N train rows for LSTM. Use 0 for all train rows.")
    parser.add_argument("--lstm-window", type=int, default=36)
    parser.add_argument("--lstm-hidden-size", type=int, default=16)
    parser.add_argument("--lstm-epochs", type=int, default=5)
    parser.add_argument("--lstm-batch-size", type=int, default=64)
    parser.add_argument("--lstm-learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-mode",
        choices=("direct", "rolling"),
        default="direct",
        help="direct compares all models from the same train/test origin; rolling is available for fast baselines only.",
    )
    return parser.parse_args()


def parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("Expected at least one integer")
    return parsed


def parse_csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("Expected at least one model")
    return parsed


def forecast_naive(history: pd.Series, steps: int) -> np.ndarray:
    return np.repeat(float(history.iloc[-1]), steps)


def forecast_seasonal_naive(history: pd.Series, steps: int, seasonal_period: int) -> np.ndarray:
    if len(history) < seasonal_period:
        return forecast_naive(history, steps)
    season = history.iloc[-seasonal_period:].to_numpy(dtype=float)
    reps = int(np.ceil(steps / len(season)))
    return np.tile(season, reps)[:steps]


def forecast_moving_average(history: pd.Series, steps: int, window: int) -> np.ndarray:
    window = max(1, min(window, len(history)))
    value = float(history.iloc[-window:].mean())
    return np.repeat(value, steps)


def make_forecast(model: str, history: pd.Series, steps: int, config: ForecastConfig) -> np.ndarray:
    if model == "naive":
        return forecast_naive(history, steps)
    if model == "seasonal_naive":
        return forecast_seasonal_naive(history, steps, config.seasonal_period)
    if model == "moving_average":
        return forecast_moving_average(history, steps, config.moving_average_window)
    raise ValueError(f"Unknown or unsupported model for this pipeline stage: {model}")


def evaluate_direct_fast_models(series: pd.DataFrame, config: ForecastConfig, models: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    train, test = chronological_split(series, config.test_size)
    max_horizon = max(config.horizons)
    if len(test) < max_horizon:
        raise ValueError("Test set is shorter than max horizon")

    prediction_rows = []
    metric_rows = []

    for model in models:
        for horizon in config.horizons:
            for slice_name in SLICES:
                history = train[slice_name]
                forecast_values = make_forecast(model, history, max_horizon, config)
                pred_value = float(forecast_values[horizon - 1])
                true_value = float(test[slice_name].iloc[horizon - 1])
                y_true = np.asarray([true_value], dtype=float)
                y_pred = np.asarray([pred_value], dtype=float)
                under, over = under_over_error(y_true, y_pred)

                prediction_rows.append(
                    {
                        "timestamp": test.index[horizon - 1],
                        "origin_timestamp": train.index[-1],
                        "model": model,
                        "slice": slice_name,
                        "horizon": horizon,
                        "evaluation": "direct",
                        "y_true": true_value,
                        "y_pred": pred_value,
                    }
                )
                metric_rows.append(
                    {
                        "model": model,
                        "slice": slice_name,
                        "horizon": horizon,
                        "evaluation": "direct",
                        "MAE": mae(y_true, y_pred),
                        "RMSE": rmse(y_true, y_pred),
                        "sMAPE": smape(y_true, y_pred),
                        "bias": bias(y_true, y_pred),
                        "under_prediction_error": under,
                        "over_prediction_error": over,
                    }
                )

    return prediction_rows, metric_rows


def evaluate_rolling_fast_models(series: pd.DataFrame, config: ForecastConfig, models: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    max_horizon = max(config.horizons)
    train, test = chronological_split(series, config.test_size)
    if len(test) < max_horizon:
        raise ValueError("Test set is shorter than max horizon")

    prediction_rows = []
    metric_rows = []
    train_len = len(train)

    for model in models:
        for horizon in config.horizons:
            for slice_name in SLICES:
                y_true_values = []
                y_pred_values = []

                for origin_offset in range(0, len(test) - horizon + 1):
                    history_end = train_len + origin_offset
                    target_pos = history_end + horizon - 1
                    history = series[slice_name].iloc[:history_end]
                    target_timestamp = series.index[target_pos]
                    true_value = float(series[slice_name].iloc[target_pos])
                    pred_value = float(make_forecast(model, history, horizon, config)[-1])

                    y_true_values.append(true_value)
                    y_pred_values.append(pred_value)

                    prediction_rows.append(
                        {
                            "timestamp": target_timestamp,
                            "origin_timestamp": series.index[history_end - 1],
                            "model": model,
                            "slice": slice_name,
                            "horizon": horizon,
                            "evaluation": "rolling",
                            "y_true": true_value,
                            "y_pred": pred_value,
                        }
                    )

                y_true = np.asarray(y_true_values, dtype=float)
                y_pred = np.asarray(y_pred_values, dtype=float)
                under, over = under_over_error(y_true, y_pred)

                metric_rows.append(
                    {
                        "model": model,
                        "slice": slice_name,
                        "horizon": horizon,
                        "evaluation": "rolling",
                        "MAE": mae(y_true, y_pred),
                        "RMSE": rmse(y_true, y_pred),
                        "sMAPE": smape(y_true, y_pred),
                        "bias": bias(y_true, y_pred),
                        "under_prediction_error": under,
                        "over_prediction_error": over,
                    }
                )

    return prediction_rows, metric_rows


def to_neuralforecast_long(df: pd.DataFrame) -> pd.DataFrame:
    long_df = (
        df.reset_index()
        .melt(id_vars="timestamp", var_name="unique_id", value_name="y")
        .rename(columns={"timestamp": "ds"})
    )
    long_df["ds"] = pd.to_datetime(long_df["ds"], utc=True).dt.tz_convert(None)
    long_df["y"] = np.log1p(long_df["y"].astype(float))
    return long_df[["unique_id", "ds", "y"]]


def evaluate_nhits(series: pd.DataFrame, config: ForecastConfig) -> tuple[list[dict], list[dict]]:
    if config.evaluation_mode != "direct":
        raise ValueError("N-HiTS currently supports only --evaluation-mode direct")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Model 'nhits' requires neuralforecast. Install it with: "
            ".venv/bin/pip install neuralforecast"
        ) from exc

    use_cuda = config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available())
    accelerator = "gpu" if use_cuda else "cpu"

    train, test = chronological_split(series, config.test_size)
    max_horizon = max(config.horizons)
    train_for_model = train
    if config.nhits_train_tail and config.nhits_train_tail > 0:
        train_for_model = train.tail(config.nhits_train_tail)

    long_train = to_neuralforecast_long(train_for_model)
    model = NHITS(
        h=max_horizon,
        input_size=config.nhits_input_size,
        max_steps=config.nhits_max_steps,
        batch_size=32,
        windows_batch_size=256,
        scaler_type="robust",
        random_seed=config.seed,
        alias="nhits",
        n_blocks=[1, 1, 1],
        mlp_units=[[128, 128], [128, 128], [128, 128]],
        accelerator=accelerator,
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    nf = NeuralForecast(models=[model], freq=config.freq)
    val_size = min(config.nhits_val_size, max(0, len(train_for_model) - config.nhits_input_size - max_horizon))
    nf.fit(df=long_train, val_size=val_size, verbose=False)
    forecast = nf.predict(verbose=False).reset_index()

    prediction_rows = []
    metric_rows = []

    for horizon in config.horizons:
        for slice_name in SLICES:
            sub = forecast[forecast["unique_id"] == slice_name].sort_values("ds")
            if len(sub) < horizon:
                raise RuntimeError(f"N-HiTS returned only {len(sub)} rows for {slice_name}; need {horizon}")

            pred_log = float(sub["nhits"].iloc[horizon - 1])
            pred_value = float(np.expm1(pred_log))
            pred_value = max(0.0, pred_value)
            true_value = float(test[slice_name].iloc[horizon - 1])
            target_timestamp = test.index[horizon - 1]
            origin_timestamp = train.index[-1]

            prediction_rows.append(
                {
                    "timestamp": target_timestamp,
                    "origin_timestamp": origin_timestamp,
                    "model": "nhits",
                    "slice": slice_name,
                    "horizon": horizon,
                    "evaluation": "direct",
                    "y_true": true_value,
                    "y_pred": pred_value,
                }
            )

            y_true = np.asarray([true_value], dtype=float)
            y_pred = np.asarray([pred_value], dtype=float)
            under, over = under_over_error(y_true, y_pred)
            metric_rows.append(
                {
                    "model": "nhits",
                    "slice": slice_name,
                    "horizon": horizon,
                    "evaluation": "direct",
                    "MAE": mae(y_true, y_pred),
                    "RMSE": rmse(y_true, y_pred),
                    "sMAPE": smape(y_true, y_pred),
                    "bias": bias(y_true, y_pred),
                    "under_prediction_error": under,
                    "over_prediction_error": over,
                }
            )

    return prediction_rows, metric_rows


def evaluate_prophet(series: pd.DataFrame, config: ForecastConfig) -> tuple[list[dict], list[dict]]:
    if config.evaluation_mode != "direct":
        raise ValueError("Prophet currently supports only --evaluation-mode direct")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    try:
        from prophet import Prophet
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Model 'prophet' requires prophet. Install it with: "
            ".venv/bin/pip install prophet"
        ) from exc

    train, test = chronological_split(series, config.test_size)
    max_horizon = max(config.horizons)
    prediction_rows = []
    metric_rows = []

    for slice_name in SLICES:
        train_series = train[slice_name]
        if config.prophet_train_tail and config.prophet_train_tail > 0:
            train_series = train_series.tail(config.prophet_train_tail)

        train_df = pd.DataFrame(
            {
                "ds": pd.to_datetime(train_series.index, utc=True).tz_convert(None),
                "y": np.log1p(train_series.to_numpy(dtype=float)),
            }
        )
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.05,
        )
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                model.fit(train_df)

        future = pd.DataFrame(
            {
                "ds": pd.to_datetime(test.index[:max_horizon], utc=True).tz_convert(None),
            }
        )
        forecast = model.predict(future)
        yhat = np.maximum(0.0, np.expm1(forecast["yhat"].to_numpy(dtype=float)))

        for horizon in config.horizons:
            pred_value = float(yhat[horizon - 1])
            true_value = float(test[slice_name].iloc[horizon - 1])
            y_true = np.asarray([true_value], dtype=float)
            y_pred = np.asarray([pred_value], dtype=float)
            under, over = under_over_error(y_true, y_pred)

            prediction_rows.append(
                {
                    "timestamp": test.index[horizon - 1],
                    "origin_timestamp": train.index[-1],
                    "model": "prophet",
                    "slice": slice_name,
                    "horizon": horizon,
                    "evaluation": "direct",
                    "y_true": true_value,
                    "y_pred": pred_value,
                }
            )
            metric_rows.append(
                {
                    "model": "prophet",
                    "slice": slice_name,
                    "horizon": horizon,
                    "evaluation": "direct",
                    "MAE": mae(y_true, y_pred),
                    "RMSE": rmse(y_true, y_pred),
                    "sMAPE": smape(y_true, y_pred),
                    "bias": bias(y_true, y_pred),
                    "under_prediction_error": under,
                    "over_prediction_error": over,
                }
            )

    return prediction_rows, metric_rows


def make_lstm_supervised(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for idx in range(window, len(values)):
        x_rows.append(values[idx - window:idx])
        y_rows.append(values[idx])
    if not x_rows:
        raise ValueError(f"Not enough values ({len(values)}) for LSTM window {window}")
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def evaluate_lstm(series: pd.DataFrame, config: ForecastConfig) -> tuple[list[dict], list[dict]]:
    if config.evaluation_mode != "direct":
        raise ValueError("LSTM currently supports only --evaluation-mode direct")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Model 'lstm' requires torch.") from exc

    torch.manual_seed(config.seed)
    use_cuda = config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    if not use_cuda:
        torch.set_num_threads(1)

    class SliceLSTM(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :]).squeeze(-1)

    train, test = chronological_split(series, config.test_size)
    max_horizon = max(config.horizons)
    prediction_rows = []
    metric_rows = []

    for slice_name in SLICES:
        train_series = train[slice_name]
        if config.lstm_train_tail and config.lstm_train_tail > 0:
            train_series = train_series.tail(config.lstm_train_tail)

        y_log = np.log1p(train_series.to_numpy(dtype=float))
        mu = float(y_log.mean())
        sigma = float(y_log.std())
        if sigma == 0:
            sigma = 1.0
        y_norm = ((y_log - mu) / sigma).astype(np.float32)

        x_train, y_train = make_lstm_supervised(y_norm, config.lstm_window)
        x_tensor = torch.from_numpy(x_train).unsqueeze(-1)
        y_tensor = torch.from_numpy(y_train)
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=config.lstm_batch_size,
            shuffle=True,
        )

        model = SliceLSTM(config.lstm_hidden_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lstm_learning_rate)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(config.lstm_epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        history = list(y_norm[-config.lstm_window:])
        preds_norm = []
        with torch.no_grad():
            for _ in range(max_horizon):
                x = torch.tensor(history[-config.lstm_window:], dtype=torch.float32, device=device).view(1, config.lstm_window, 1)
                pred_norm = float(model(x).item())
                preds_norm.append(pred_norm)
                history.append(pred_norm)

        preds_log = np.asarray(preds_norm, dtype=float) * sigma + mu
        preds = np.maximum(0.0, np.expm1(preds_log))

        for horizon in config.horizons:
            pred_value = float(preds[horizon - 1])
            true_value = float(test[slice_name].iloc[horizon - 1])
            y_true = np.asarray([true_value], dtype=float)
            y_pred = np.asarray([pred_value], dtype=float)
            under, over = under_over_error(y_true, y_pred)

            prediction_rows.append(
                {
                    "timestamp": test.index[horizon - 1],
                    "origin_timestamp": train.index[-1],
                    "model": "lstm",
                    "slice": slice_name,
                    "horizon": horizon,
                    "evaluation": "direct",
                    "y_true": true_value,
                    "y_pred": pred_value,
                }
            )
            metric_rows.append(
                {
                    "model": "lstm",
                    "slice": slice_name,
                    "horizon": horizon,
                    "evaluation": "direct",
                    "MAE": mae(y_true, y_pred),
                    "RMSE": rmse(y_true, y_pred),
                    "sMAPE": smape(y_true, y_pred),
                    "bias": bias(y_true, y_pred),
                    "under_prediction_error": under,
                    "over_prediction_error": over,
                }
            )

    return prediction_rows, metric_rows


def evaluate_forecasts(series: pd.DataFrame, config: ForecastConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    trained_models = {"nhits", "prophet", "lstm"}
    fast_models = tuple(model for model in config.models if model not in trained_models)
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []

    if fast_models:
        if config.evaluation_mode == "direct":
            preds, metrics = evaluate_direct_fast_models(series, config, fast_models)
        else:
            preds, metrics = evaluate_rolling_fast_models(series, config, fast_models)
        prediction_rows.extend(preds)
        metric_rows.extend(metrics)

    if "nhits" in config.models:
        preds, metrics = evaluate_nhits(series, config)
        prediction_rows.extend(preds)
        metric_rows.extend(metrics)

    if "prophet" in config.models:
        preds, metrics = evaluate_prophet(series, config)
        prediction_rows.extend(preds)
        metric_rows.extend(metrics)

    if "lstm" in config.models:
        preds, metrics = evaluate_lstm(series, config)
        prediction_rows.extend(preds)
        metric_rows.extend(metrics)

    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def run(config: ForecastConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = read_slice_series(config.input_csv)
    predictions, metrics = evaluate_forecasts(series, config)
    predictions_path = ensure_parent(config.predictions_csv)
    metrics_path = ensure_parent(config.metrics_csv)
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    return predictions, metrics


def main() -> None:
    args = parse_args()
    config = ForecastConfig(
        input_csv=args.input_csv,
        predictions_csv=args.predictions_csv,
        metrics_csv=args.metrics_csv,
        test_size=args.test_size,
        horizons=parse_csv_ints(args.horizons),
        models=parse_csv_strings(args.models),
        seasonal_period=args.seasonal_period,
        moving_average_window=args.moving_average_window,
        freq=args.freq,
        nhits_input_size=args.nhits_input_size,
        nhits_max_steps=args.nhits_max_steps,
        nhits_train_tail=None if args.nhits_train_tail == 0 else args.nhits_train_tail,
        nhits_val_size=args.nhits_val_size,
        seed=args.seed,
        evaluation_mode=args.evaluation_mode,
        prophet_train_tail=None if args.prophet_train_tail == 0 else args.prophet_train_tail,
        lstm_train_tail=None if args.lstm_train_tail == 0 else args.lstm_train_tail,
        lstm_window=args.lstm_window,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_epochs=args.lstm_epochs,
        lstm_batch_size=args.lstm_batch_size,
        lstm_learning_rate=args.lstm_learning_rate,
        device=args.device,
    )
    predictions, metrics = run(config)
    print(f"Wrote {config.predictions_csv} rows={len(predictions)}")
    print(f"Wrote {config.metrics_csv} rows={len(metrics)}")
    print(metrics.sort_values(["slice", "horizon", "RMSE"]).to_string(index=False))


if __name__ == "__main__":
    main()
