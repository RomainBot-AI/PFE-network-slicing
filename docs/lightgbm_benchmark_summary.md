# LightGBM benchmark summary

This document freezes the current LightGBM state before moving to deep learning
models.

## Role in the benchmark

LightGBM is the main machine-learning baseline:

- it uses hand-crafted temporal features;
- it trains fast;
- it gives a strong performance/cost reference before LSTM, N-HiTS, and PatchTST;
- it is trained per slice, not globally.

## Protocol

Common forecasting protocol:

```text
granularity: subnet/slice
input_size: 2016 steps = 14 days
horizon: 36 steps = 6 hours
folds: 5 rolling-origin folds
training scope: per_slice
```

For LightGBM:

```text
models trained: 720 = 4 slices x 5 folds x 36 horizons
features: 23
```

Each LightGBM model is trained on only one slice. The slice is therefore not a
global sharing variable; it defines the training subset.

## Commands

Base benchmark:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_deterministic_benchmark \
  --config configs/experiment/deterministic_benchmark.yaml
```

LightGBM tuning:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.tune_lightgbm \
  --config configs/experiment/lightgbm_tuning.yaml
```

Tuned benchmark:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_deterministic_benchmark \
  --config configs/experiment/deterministic_benchmark_lightgbm_tuned.yaml
```

Tuned report:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.make_benchmark_report \
  --run-dir experiments/runs/deterministic_benchmark_lightgbm_tuned
```

## Outputs to keep

```text
experiments/runs/deterministic_benchmark/
experiments/runs/lightgbm_tuning/
experiments/runs/deterministic_benchmark_lightgbm_tuned/
```

Important files:

```text
benchmark_summary.csv
benchmark_summary_by_slice.csv
lightgbm_base_vs_tuned_by_slice.csv
timing.csv
model_metadata.csv
benchmark_report.html
```

## Base vs tuned LightGBM

Tuning uses bounded random search:

```text
20 trials per slice
objective: MASE
tuned fold: first rolling-origin fold only
validation: final 20% of generated training examples
test folds: never used for tuning
```

Comparison by slice:

| slice | RMSE change | WAPE change | MASE change | interpretation |
| --- | ---: | ---: | ---: | --- |
| URLLC | -0.01% | -9.87% | -0.11% | small but consistent improvement |
| URLLC_eMBB_MIX | -0.53% | -32.42% | -1.32% | tuning clearly helps normalized errors |
| eMBB | +2.11% | -43.03% | -0.29% | WAPE/MASE improve but RMSE worsens |
| mMTC | -3.74% | -5.71% | -5.11% | tuning improves all main metrics |

## Current conclusion

LightGBM is accepted as the ML benchmark.

Main conclusion:

- tuned LightGBM improves WAPE and MASE across all slices;
- tuned LightGBM improves RMSE on URLLC, URLLC_eMBB_MIX, and mMTC;
- eMBB remains better served by persistence under RMSE;
- persistence remains the strongest global RMSE baseline because eMBB dominates absolute traffic volume.

This is enough for LightGBM's role in the thesis. Further optimization is not
recommended before evaluating the deep-learning models.
