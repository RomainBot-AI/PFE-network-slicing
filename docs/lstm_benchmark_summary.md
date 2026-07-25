# LSTM benchmark summary

This document freezes the LSTM baseline before moving to N-HiTS.

## Role

LSTM is the historical deep-learning baseline. It tests whether a recurrent
sequence model improves over:

- persistence and seasonal naive baselines;
- tuned LightGBM with hand-crafted lags.

It is not expected to dominate automatically. Its value is to provide a deep
baseline before modern architectures such as N-HiTS and PatchTST.

## Protocol

```text
granularity: subnet/slice
training scope: per_slice
input_size: 2016 steps = 14 days
horizon: 36 steps = 6 hours
folds: 5 rolling-origin folds
device: CUDA
```

The LSTM predicts all 36 horizons jointly.

```text
trained models: 20 = 4 slices x 5 folds
predictions: 32,220 = 179 series x 36 horizons x 5 folds
```

## Tuning

Tuning command:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.tune_lstm \
  --config configs/experiment/lstm_tuning.yaml
```

Tuning setup:

```text
one Optuna study per slice
n_trials: 20 per slice
objective: MASE
validation: chronological final 20% of generated training windows
tuned fold: first rolling-origin fold
test folds: not used for hyperparameter selection
```

Tuning output:

```text
experiments/runs/lstm_tuning/
  lstm_tuning_trials.csv
  lstm_best_params_by_slice.yaml
  run_meta.json
  resolved_config.yaml
```

## Benchmark

Benchmark command:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_lstm_benchmark \
  --config configs/experiment/lstm_benchmark.yaml
```

Benchmark output:

```text
experiments/runs/lstm_benchmark/
```

Key files:

```text
benchmark_summary.csv
benchmark_summary_by_slice.csv
timing.csv
model_metadata.csv
benchmark_report.html
lstm_vs_lightgbm_tuned_by_slice.csv
lstm_training_cost_summary.csv
```

## Results

Global LSTM:

```text
RMSE: 30.54M
WAPE: 1.054
MASE: 0.573
```

By slice:

| slice | RMSE | WAPE | MASE |
| --- | ---: | ---: | ---: |
| URLLC | 1.516M | 1.004 | 0.096 |
| URLLC_eMBB_MIX | 35.12M | 1.246 | 0.998 |
| eMBB | 85.53M | 1.124 | 0.557 |
| mMTC | 4.55K | 0.841 | 0.643 |

Compared with tuned LightGBM:

- URLLC: RMSE almost identical; LSTM much better in MASE.
- URLLC_eMBB_MIX: tuned LightGBM is better.
- eMBB: tuned LightGBM is better in RMSE/WAPE; LSTM is better in MASE.
- mMTC: LSTM is better across RMSE/WAPE/MASE.

## Training Cost

The run may look fast despite the 7.2M-row panel because the LSTM benchmark does
not train on every raw panel row.

The configuration caps the sequence dataset:

```text
max_windows_per_slice: 2500
train_origin_stride: 288
```

For the benchmark:

```text
20 trained models
2,500 training windows per model
50,000 total training windows
```

Measured cost:

```text
total train time: 177.39 s
total inference time: 0.19 s
mean train time per model: 8.87 s
```

By slice:

| slice | total train time | mean train time/model | parameters |
| --- | ---: | ---: | ---: |
| URLLC | 77.57 s | 15.51 s | 71,716 |
| URLLC_eMBB_MIX | 48.87 s | 9.77 s | 71,716 |
| eMBB | 33.84 s | 6.77 s | 52,772 |
| mMTC | 17.10 s | 3.42 s | 14,116 |

This is coherent because:

- CUDA was used;
- only 20 models were trained;
- each model sees at most 2,500 windows;
- the LSTM architectures are small;
- the benchmark predicts all 36 horizons jointly, unlike LightGBM's one model per horizon.

## Sensitivity Run: 5000 Windows

To test whether the LSTM was under-trained with 2,500 windows per model, run the
same benchmark with more windows and identical tuned hyperparameters:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_lstm_benchmark \
  --config configs/experiment/lstm_benchmark_5000w.yaml
```

This run uses:

```text
max_windows_per_slice: 5000
trained models: 20
expected training windows: 100,000
```

Compare it against:

```text
experiments/runs/lstm_benchmark/
```

The goal is a sensitivity check, not a new tuning run.

Completed output:

```text
experiments/runs/lstm_benchmark_5000w/
```

Results:

```text
global RMSE: 29.02M vs 30.54M with 2500 windows
global WAPE: 0.992 vs 1.054
global MASE: 0.563 vs 0.573
total train time: 354.06 s vs 177.39 s
```

By slice, 5000 windows vs 2500 windows:

| slice | RMSE change | WAPE change | MASE change | interpretation |
| --- | ---: | ---: | ---: | --- |
| URLLC | +0.00% | +1.13% | +0.00% | no useful gain |
| URLLC_eMBB_MIX | -0.41% | -14.46% | -1.15% | useful improvement |
| eMBB | -6.94% | -7.45% | -6.16% | clear improvement |
| mMTC | +1.55% | +0.82% | +0.90% | slight degradation |

Conclusion: 5000 windows is better globally because it improves eMBB and
URLLC_eMBB_MIX, but the gain is not uniform. For the benchmark table, keep both
2500 and 5000 sensitivity results visible, and prefer 5000 if selecting the best
LSTM variant.

## Sensitivity Run: 10000 Windows

A final sensitivity run tested whether the 5000-window gain continues:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_lstm_benchmark \
  --config configs/experiment/lstm_benchmark_10000w.yaml
```

Completed output:

```text
max_windows_per_slice: 10000
trained models: 20
actual training windows: 118,319
total train time: 402.20 s
total inference time: 0.23 s
```

The run does not reach 200,000 windows because several slice/fold datasets do
not contain 10,000 available chronological training windows after applying the
2016-step history, 36-step horizon, and per-slice split constraints.

Global results:

| run | RMSE | WAPE | MASE | train time |
| --- | ---: | ---: | ---: | ---: |
| 2500 windows | 30.54M | 1.054 | 0.573 | 177.39 s |
| 5000 windows | 29.02M | 0.992 | 0.563 | 354.06 s |
| 10000 windows | 29.17M | 0.980 | 0.562 | 402.20 s |

By slice, 10000 windows vs 5000 windows:

| slice | RMSE change | WAPE change | MASE change | interpretation |
| --- | ---: | ---: | ---: | --- |
| URLLC | -0.00% | -0.19% | +0.01% | unchanged |
| URLLC_eMBB_MIX | -0.27% | -14.58% | -1.17% | useful WAPE/MASE gain |
| eMBB | +0.87% | +9.85% | +0.64% | degradation on the dominant slice |
| mMTC | -0.95% | +0.79% | +0.45% | mixed, practically marginal |

Conclusion: 10000 windows is useful as a sensitivity result, but it should not
replace 5000 windows as the main LSTM variant. The extra cost is moderate
because the available windows saturate before 10,000 on several folds, but the
dominant eMBB slice gets worse. Keep 5000 windows as the best cost/performance
choice unless the report prioritizes WAPE/MASE over RMSE.

## Conclusion

LSTM is accepted as the deep baseline.

It is competitive and improves normalized scale error on several slices, but it
does not dominate tuned LightGBM everywhere. This supports the report narrative:
deep learning is useful, but not automatically superior to strong tabular and
persistence baselines.
