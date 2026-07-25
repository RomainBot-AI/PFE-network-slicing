# N-HiTS benchmark plan

N-HiTS is the first modern deep forecasting architecture in the benchmark.

## Role

N-HiTS is kept per slice, like LSTM, to respect the traffic-regime separation
created by the slicing stage. It predicts the full 36-step horizon jointly.

## Protocol

```text
granularity: subnet/slice
training scope: per_slice
input_size: 2016 steps = 14 days
horizon: 36 steps = 6 hours
folds: 5 rolling-origin folds
frequency: 10 minutes
```

The runner trains:

```text
20 models = 4 slices x 5 folds
32,220 predictions = 179 series x 36 horizons x 5 folds
```

## Leakage Controls

- Folds are generated before model training and shared with the other models.
- Each model receives only rows up to `train_end`.
- Forecast targets start strictly after the training origin.
- The log transform is deterministic and does not use future statistics.
- NeuralForecast scaling is fitted inside each fold on the training frame only.

## Commands

Smoke check:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_nhits_benchmark --config configs/experiment/nhits_benchmark_smoke.yaml
```

Full benchmark:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_nhits_benchmark --config configs/experiment/nhits_benchmark.yaml
```

Optuna tuning:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.tune_nhits --config configs/experiment/nhits_tuning.yaml
```

Robust Optuna tuning:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.tune_nhits --config configs/experiment/nhits_tuning_robust.yaml
```

Tuned benchmark:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_nhits_benchmark --config configs/experiment/nhits_benchmark_tuned.yaml
```

Robust tuned benchmark:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_nhits_benchmark --config configs/experiment/nhits_benchmark_robust_tuned.yaml
```

Report:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.make_benchmark_report --run-dir experiments/runs/nhits_benchmark
```

Tuned report:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.make_benchmark_report --run-dir experiments/runs/nhits_benchmark_tuned
```

## Initial Configuration

The first full benchmark uses a moderate budget:

```text
max_steps: 300
step_size: 288
batch_size: 32
windows_batch_size: 1024
mlp_units: [256, 256] x 3 stacks
```

This is intentionally not the final tuned model. It verifies that N-HiTS is
stable and comparable under the same temporal protocol before adding Optuna.

## Tuning

The Optuna tuner uses the first rolling-origin fold only and creates an internal
validation target inside that fold's training history:

```text
inner train: up to fold.train_end_idx - horizon
inner validation: next horizon points
test folds: not used during hyperparameter selection
```

Search budget:

```text
n_trials: 12 per slice
objective: MASE
search space: max_steps, learning_rate, batch_size, windows_batch_size,
              step_size, mlp_width, n_blocks, dropout_prob_theta
```

Tuning output:

```text
experiments/runs/nhits_tuning/
  nhits_tuning_trials.csv
  nhits_best_params_by_slice.yaml
  run_meta.json
  resolved_config.yaml
```

## Tuned Results

Completed tuned benchmark:

```text
experiments/runs/nhits_benchmark_tuned/
```

Global N-HiTS tuned:

```text
RMSE: 33.87M
WAPE: 1.811
MASE: 0.650
```

By slice:

| slice | RMSE | WAPE | MASE |
| --- | ---: | ---: | ---: |
| URLLC | 1.524M | 2.654 | 0.097 |
| URLLC_eMBB_MIX | 36.39M | 2.011 | 1.037 |
| eMBB | 97.55M | 1.536 | 0.689 |
| mMTC | 5.77K | 1.044 | 0.777 |

Compared with LSTM 5000 windows:

| slice | RMSE change | WAPE change | MASE change |
| --- | ---: | ---: | ---: |
| URLLC | +0.56% | +161.50% | +1.17% |
| URLLC_eMBB_MIX | +4.06% | +88.72% | +5.11% |
| eMBB | +22.57% | +47.68% | +31.90% |
| mMTC | +25.02% | +23.05% | +19.65% |

Training cost:

```text
trained models: 20
total train time: 131.88 s
total inference time: 2.09 s
```

Interpretation: tuned N-HiTS is valid as an experimental modern-deep model, but
it does not beat the retained LSTM 5000-window baseline on the rolling-origin
test folds. The tuning objective improved the inner validation fold, but the
selected parameters do not generalize well enough across the full temporal
backtest.

## Robust Tuning

The first tuning run selected hyperparameters from a single internal validation
block. Because its test performance was weak, a second robust tuning protocol is
available:

```text
config: configs/experiment/nhits_tuning_robust.yaml
output: experiments/runs/nhits_tuning_robust/
n_trials: 10 per slice
internal validation folds: 3
objective: mean MASE over the 3 internal folds
```

The search space is deliberately more regularized:

```text
mlp_width: 64, 128, 256
n_blocks: 1
dropout_prob_theta: 0.0 to 0.4
scaler_type: robust
step_size: 144 or 288
```

This run is slower than the first tuning run because each trial trains three
N-HiTS models. It is the recommended final check before rejecting N-HiTS.

## Robust Tuned Results

Completed robust tuned benchmark:

```text
experiments/runs/nhits_benchmark_robust_tuned/
```

Global N-HiTS robust tuned:

```text
RMSE: 35.61M
WAPE: 2.713
MASE: 0.679
```

By slice:

| slice | RMSE | WAPE | MASE |
| --- | ---: | ---: | ---: |
| URLLC | 1.581M | 6.823 | 0.101 |
| URLLC_eMBB_MIX | 34.86M | 0.880 | 0.977 |
| eMBB | 105.99M | 1.884 | 0.716 |
| mMTC | 5.91K | 1.264 | 0.922 |

Compared with the first N-HiTS tuning:

| slice | RMSE change | WAPE change | MASE change |
| --- | ---: | ---: | ---: |
| URLLC | +3.74% | +157.08% | +4.37% |
| URLLC_eMBB_MIX | -4.20% | -56.23% | -5.79% |
| eMBB | +8.65% | +22.61% | +3.98% |
| mMTC | +2.27% | +21.13% | +18.70% |

Compared with LSTM 5000 windows:

| slice | RMSE change | WAPE change | MASE change |
| --- | ---: | ---: | ---: |
| URLLC | +4.32% | +572.26% | +5.59% |
| URLLC_eMBB_MIX | -0.30% | -17.39% | -0.98% |
| eMBB | +33.17% | +81.07% | +37.15% |
| mMTC | +27.86% | +49.05% | +42.03% |

Interpretation: robust tuning improves `URLLC_eMBB_MIX`, but it worsens the
dominant `eMBB` slice and degrades the global benchmark. N-HiTS should therefore
remain in the final comparison as the tested modern deep model, but it should
not be selected as the best forecasting model for this dataset.
