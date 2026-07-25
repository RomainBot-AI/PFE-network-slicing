# LSTM tuning plan

This is the first deep-learning baseline after LightGBM.

## Role

LSTM is the historical deep baseline. It is included to compare classical
sequence learning against:

- persistence and seasonal naive baselines;
- LightGBM with engineered lags;
- later modern deep models such as N-HiTS and PatchTST.

## Protocol

The LSTM tuner is per-slice:

```text
one Optuna study per slice
input sequence length: configurable
forecast horizon: configurable
objective: MASE by default
```

The full default config uses:

```text
input_size: 2016
horizon: 36
n_trials: 20 per slice
max_epochs: 20
max_windows_per_slice: 2500
device: auto
```

The tuner uses only historical windows before the selected fold target period.
The validation block is chronological and taken from the final part of generated
training windows. Test folds are not used for hyperparameter selection.

## Command to run

```bash
PYTHONPATH=src .venv/bin/python -m scripts.tune_lstm \
  --config configs/experiment/lstm_tuning.yaml
```

Expected outputs:

```text
experiments/runs/lstm_tuning/
  lstm_tuning_trials.csv
  lstm_best_params_by_slice.yaml
  run_meta.json
  resolved_config.yaml
```

## Smoke test already validated

Smoke config:

```text
configs/experiment/lstm_tuning_smoke.yaml
```

Smoke output:

```text
experiments/runs/lstm_tuning_smoke/
```

The smoke test uses tiny sequences and one trial per slice only. It validates the
pipeline, not model quality.

## Benchmark After Tuning

After the full tuning command has produced:

```text
experiments/runs/lstm_tuning/lstm_best_params_by_slice.yaml
```

run the full LSTM benchmark with:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_lstm_benchmark \
  --config configs/experiment/lstm_benchmark.yaml
```

Expected outputs:

```text
experiments/runs/lstm_benchmark/
  resolved_config.yaml
  run_meta.json
  folds.csv
  leakage_audit.csv
  predictions.csv
  metrics_by_fold.csv
  metrics.csv
  benchmark_summary.csv
  benchmark_summary_by_slice.csv
  timing.csv
  model_metadata.csv
```

This benchmark trains one LSTM per slice and fold:

```text
4 slices x 5 folds = 20 LSTM models
```

## Completed Full Benchmark

Output:

```text
experiments/runs/lstm_benchmark/
```

Summary:

```text
global: RMSE 30.54M, WAPE 1.054, MASE 0.573
trained models: 20
device: CUDA
predictions: 32,220
```

Comparison file:

```text
experiments/runs/lstm_benchmark/lstm_vs_lightgbm_tuned_by_slice.csv
```
