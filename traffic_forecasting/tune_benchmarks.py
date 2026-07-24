#!/usr/bin/env python3
"""Tune baseline and Prophet parameters for the direct forecasting benchmark."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace

import pandas as pd

try:
    from traffic_forecasting.common import read_slice_series, ensure_parent
    from traffic_forecasting.forecast_4_slices import (
        ForecastConfig,
        evaluate_direct_fast_models,
        evaluate_prophet,
        parse_csv_ints,
    )
except ModuleNotFoundError:
    from common import read_slice_series, ensure_parent
    from forecast_4_slices import (
        ForecastConfig,
        evaluate_direct_fast_models,
        evaluate_prophet,
        parse_csv_ints,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="traffic_forecasting/data/slice_traffic_10min.csv")
    parser.add_argument("--output-csv", default="traffic_forecasting/reports/tuning_baselines_prophet.csv")
    parser.add_argument("--test-size", type=int, default=144)
    parser.add_argument("--horizons", default="1,6,12")
    parser.add_argument("--moving-average-windows", default="3,6,12,24,72,144")
    parser.add_argument("--seasonal-periods", default="72,144,288,1008")
    parser.add_argument("--prophet-train-tails", default="1000,2000,5000,10000")
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
        models=(),
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
        lstm_train_tail=2000,
        lstm_window=36,
        lstm_hidden_size=16,
        lstm_epochs=5,
        lstm_batch_size=64,
        lstm_learning_rate=0.001,
        device=args.device,
    )


def append_metadata(rows: list[dict], **metadata: int | str | None) -> list[dict]:
    output = []
    for row in rows:
        enriched = dict(row)
        enriched.update(metadata)
        output.append(enriched)
    return output


def main() -> None:
    args = parse_args()
    series = read_slice_series(args.input_csv)
    template = base_config(args)
    rows = []

    print("Tuning naive")
    _, metrics = evaluate_direct_fast_models(series, template, ("naive",))
    rows.extend(append_metadata(metrics, tuned_model="naive"))

    for window in parse_csv_ints(args.moving_average_windows):
        print(f"Tuning moving_average window={window}")
        config = replace(template, moving_average_window=window)
        _, metrics = evaluate_direct_fast_models(series, config, ("moving_average",))
        rows.extend(append_metadata(metrics, tuned_model="moving_average", moving_average_window=window))

    for seasonal_period in parse_csv_ints(args.seasonal_periods):
        print(f"Tuning seasonal_naive seasonal_period={seasonal_period}")
        config = replace(template, seasonal_period=seasonal_period)
        _, metrics = evaluate_direct_fast_models(series, config, ("seasonal_naive",))
        rows.extend(append_metadata(metrics, tuned_model="seasonal_naive", seasonal_period=seasonal_period))

    for train_tail in parse_csv_ints(args.prophet_train_tails):
        print(f"Tuning prophet train_tail={train_tail}")
        config = replace(template, prophet_train_tail=train_tail)
        _, metrics = evaluate_prophet(series, config)
        rows.extend(append_metadata(metrics, tuned_model="prophet", prophet_train_tail=train_tail))

        output_path = ensure_parent(args.output_csv)
        pd.DataFrame(rows).to_csv(output_path, index=False)

    results = pd.DataFrame(rows)
    output_path = ensure_parent(args.output_csv)
    results.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")

    summary = (
        results.groupby(["model"], as_index=False)
        .agg(mean_rmse=("RMSE", "mean"), median_rmse=("RMSE", "median"))
        .sort_values("mean_rmse")
    )
    print(summary.to_string(index=False))

    best = results.sort_values("RMSE").groupby(["slice", "horizon"], as_index=False).first()
    print()
    print("Best by slice/horizon:")
    print(best[["slice", "horizon", "model", "RMSE", "MAE", "sMAPE"]].sort_values(["slice", "horizon"]).to_string(index=False))


if __name__ == "__main__":
    main()
