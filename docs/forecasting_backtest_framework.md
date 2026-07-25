# Forecasting backtest framework

The forecasting benchmark now has a configurable panel backtest entry point:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_panel_backtest \
  --config configs/experiment/subnet_slice_baseline_backtest.yaml
```

The framework is designed for the subnet/slice panel:

```text
traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
```

## Outputs

Each run writes a self-contained folder under `experiments/runs/<run_id>/`:

```text
resolved_config.yaml
run_meta.json
folds.csv
leakage_audit.csv
predictions.csv
metrics_by_fold.csv
metrics.csv
```

`experiments/runs/` is ignored by Git because these are generated artifacts.

## Leakage checks

The first explicit checks are:

- no random split;
- one common timestamp grid for all subnet/slice series;
- each fold uses only historical context before the target block;
- train and target windows have zero overlap;
- scaling is not applied globally in the current baseline engine.

Future models that require scaling must fit scalers inside each fold using only
the fold training block, then apply those fold-specific parameters to validation
or test targets.

## Fold-aware preprocessing

The preprocessing entry point creates a tabular modeling dataset with calendar
features, historical lags, fold-specific target scaling, and explicit audits:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.prepare_panel_dataset \
  --config configs/experiment/subnet_slice_preprocess.yaml
```

Main outputs:

```text
data/processed/subnet_slice_preprocess/
  resolved_config.yaml
  features.csv
  scalers.csv
  folds.csv
  leakage_audit.csv
  feature_audit.csv
  run_meta.json
```

Current full preprocessing configuration:

- input size: 2016
- horizon: 36
- folds: 5
- series: 179
- rows: 32,220
- max lag: 1008

The generated `features.csv` is a direct supervised dataset for classical ML
models such as LightGBM. Deep models can either consume this table or use the
same fold definitions to build tensors.

## Deterministic benchmark

The first deterministic benchmark compares:

- `persistence`
- `seasonal_naive_daily`
- `seasonal_naive_weekly`
- `lightgbm`

Command:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_deterministic_benchmark \
  --config configs/experiment/deterministic_benchmark.yaml
```

Report:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.make_benchmark_report
```

Outputs:

```text
experiments/runs/deterministic_benchmark/
  resolved_config.yaml
  run_meta.json
  folds.csv
  leakage_audit.csv
  predictions.csv
  metrics_by_fold.csv
  metrics.csv
  benchmark_summary.csv
  timing.csv
  model_metadata.csv
  benchmark_report.html
```

Current full run:

- folds: 5
- horizon: 36
- predictions: 128,880
- LightGBM training scope: per-slice
- LightGBM trained models: 720
- leakage audit: all folds have zero train/target overlap

LightGBM tuning is available with:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.tune_lightgbm \
  --config configs/experiment/lightgbm_tuning.yaml
```

The tuned benchmark config is:

```text
configs/experiment/deterministic_benchmark_lightgbm_tuned.yaml
```

The frozen LightGBM summary is:

```text
docs/lightgbm_benchmark_summary.md
```

The current end-to-end benchmark status is tracked in:

```text
docs/forecasting_benchmark_status.md
```

This file is the preferred entry point before continuing with Prophet and
PatchTST.

## Current model support

The first implementation supports fast deterministic baselines:

- `persistence`
- `seasonal_naive_daily`
- `seasonal_naive_weekly`
- `moving_average`

This validates the experiment harness before adding heavier models such as
LightGBM, LSTM, N-HiTS, or PatchTST.
