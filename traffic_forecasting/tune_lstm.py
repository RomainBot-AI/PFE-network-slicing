#!/usr/bin/env python3
"""Tune PyTorch LSTM parameters for the direct forecasting benchmark."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace

import pandas as pd

try:
    from traffic_forecasting.common import ensure_parent, read_slice_series
    from traffic_forecasting.forecast_4_slices import ForecastConfig, evaluate_lstm, parse_csv_ints
except ModuleNotFoundError:
    from common import ensure_parent, read_slice_series
    from forecast_4_slices import ForecastConfig, evaluate_lstm, parse_csv_ints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="traffic_forecasting/data/slice_traffic_10min.csv")
    parser.add_argument("--output-csv", default="traffic_forecasting/reports/lstm_tuning.csv")
    parser.add_argument("--test-size", type=int, default=144)
    parser.add_argument("--horizons", default="1,6,12")
    parser.add_argument("--train-tails", default="1000,2000,5000")
    parser.add_argument("--windows", default="36,72,144")
    parser.add_argument("--hidden-sizes", default="16,32,64")
    parser.add_argument("--epochs", default="5,10,20")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def base_config(args: argparse.Namespace) -> ForecastConfig:
    return ForecastConfig(
        input_csv=args.input_csv,
        predictions_csv="",
        metrics_csv="",
        test_size=args.test_size,
        horizons=parse_csv_ints(args.horizons),
        models=("lstm",),
        seasonal_period=144,
        moving_average_window=12,
        freq=args.freq,
        nhits_input_size=72,
        nhits_max_steps=200,
        nhits_train_tail=5000,
        nhits_val_size=144,
        seed=args.seed,
        evaluation_mode="direct",
        prophet_train_tail=5000,
        lstm_train_tail=5000,
        lstm_window=144,
        lstm_hidden_size=64,
        lstm_epochs=20,
        lstm_batch_size=args.batch_size,
        lstm_learning_rate=args.learning_rate,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    series = read_slice_series(args.input_csv)
    template = base_config(args)
    rows = []

    configs = itertools.product(
        parse_csv_ints(args.train_tails),
        parse_csv_ints(args.windows),
        parse_csv_ints(args.hidden_sizes),
        parse_csv_ints(args.epochs),
    )

    for train_tail, window, hidden_size, epochs in configs:
        if train_tail <= window + max(template.horizons):
            continue
        print(
            "LSTM tuning "
            f"train_tail={train_tail} window={window} "
            f"hidden_size={hidden_size} epochs={epochs}"
        )
        config = replace(
            template,
            lstm_train_tail=train_tail,
            lstm_window=window,
            lstm_hidden_size=hidden_size,
            lstm_epochs=epochs,
        )
        try:
            _, metrics = evaluate_lstm(series, config)
            for row in metrics:
                row = dict(row)
                row["lstm_train_tail"] = train_tail
                row["lstm_window"] = window
                row["lstm_hidden_size"] = hidden_size
                row["lstm_epochs"] = epochs
                rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "model": "lstm",
                    "slice": "__error__",
                    "horizon": -1,
                    "evaluation": "direct",
                    "MAE": float("nan"),
                    "RMSE": float("nan"),
                    "sMAPE": float("nan"),
                    "bias": float("nan"),
                    "under_prediction_error": float("nan"),
                    "over_prediction_error": float("nan"),
                    "lstm_train_tail": train_tail,
                    "lstm_window": window,
                    "lstm_hidden_size": hidden_size,
                    "lstm_epochs": epochs,
                    "error": repr(exc),
                }
            )

        output_path = ensure_parent(args.output_csv)
        pd.DataFrame(rows).to_csv(output_path, index=False)

    results = pd.DataFrame(rows)
    output_path = ensure_parent(args.output_csv)
    results.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")

    valid = results[results["slice"] != "__error__"].copy()
    if not valid.empty:
        summary = (
            valid.groupby(["lstm_train_tail", "lstm_window", "lstm_hidden_size", "lstm_epochs"], as_index=False)
            .agg(mean_rmse=("RMSE", "mean"), median_rmse=("RMSE", "median"))
            .sort_values("mean_rmse")
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
