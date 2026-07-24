#!/usr/bin/env python3
"""Run a small reproducible N-HiTS tuning grid against the direct benchmark."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd

try:
    from traffic_forecasting.common import ensure_parent, read_slice_series
    from traffic_forecasting.forecast_4_slices import (
        ForecastConfig,
        evaluate_direct_fast_models,
        evaluate_nhits,
        parse_csv_ints,
    )
except ModuleNotFoundError:
    from common import ensure_parent, read_slice_series
    from forecast_4_slices import (
        ForecastConfig,
        evaluate_direct_fast_models,
        evaluate_nhits,
        parse_csv_ints,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="traffic_forecasting/data/slice_traffic_10min.csv")
    parser.add_argument("--output-csv", default="traffic_forecasting/reports/nhits_tuning.csv")
    parser.add_argument("--baseline-output-csv", default="traffic_forecasting/reports/nhits_tuning_baselines.csv")
    parser.add_argument("--test-size", type=int, default=144)
    parser.add_argument("--horizons", default="1,6,12")
    parser.add_argument("--input-sizes", default="72,144,288")
    parser.add_argument("--train-tails", default="2000,5000,10000")
    parser.add_argument("--max-steps", default="100,200")
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
        models=("nhits",),
        seasonal_period=144,
        moving_average_window=12,
        freq=args.freq,
        nhits_input_size=144,
        nhits_max_steps=100,
        nhits_train_tail=2000,
        nhits_val_size=144,
        seed=args.seed,
        evaluation_mode="direct",
        prophet_train_tail=5000,
        lstm_train_tail=2000,
        lstm_window=36,
        lstm_hidden_size=16,
        lstm_epochs=5,
        lstm_batch_size=64,
        lstm_learning_rate=0.001,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    series = read_slice_series(args.input_csv)
    template = base_config(args)

    _, baseline_metrics = evaluate_direct_fast_models(
        series,
        replace(template, models=("naive", "seasonal_naive", "moving_average")),
        ("naive", "seasonal_naive", "moving_average"),
    )
    baseline_df = pd.DataFrame(baseline_metrics)
    baseline_path = ensure_parent(args.baseline_output_csv)
    baseline_df.to_csv(baseline_path, index=False)

    rows = []
    configs = itertools.product(
        parse_csv_ints(args.input_sizes),
        parse_csv_ints(args.train_tails),
        parse_csv_ints(args.max_steps),
    )

    for input_size, train_tail, max_steps in configs:
        print(f"N-HiTS tuning input_size={input_size} train_tail={train_tail} max_steps={max_steps}")
        config = replace(
            template,
            nhits_input_size=input_size,
            nhits_train_tail=train_tail,
            nhits_max_steps=max_steps,
        )
        try:
            _, metrics = evaluate_nhits(series, config)
            for row in metrics:
                row = dict(row)
                row["nhits_input_size"] = input_size
                row["nhits_train_tail"] = train_tail
                row["nhits_max_steps"] = max_steps
                rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "model": "nhits",
                    "slice": "__error__",
                    "horizon": -1,
                    "evaluation": "direct",
                    "MAE": float("nan"),
                    "RMSE": float("nan"),
                    "sMAPE": float("nan"),
                    "bias": float("nan"),
                    "under_prediction_error": float("nan"),
                    "over_prediction_error": float("nan"),
                    "nhits_input_size": input_size,
                    "nhits_train_tail": train_tail,
                    "nhits_max_steps": max_steps,
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
            valid.groupby(["nhits_input_size", "nhits_train_tail", "nhits_max_steps"], as_index=False)
            .agg(mean_rmse=("RMSE", "mean"), median_rank_metric=("RMSE", "median"))
            .sort_values("mean_rmse")
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
