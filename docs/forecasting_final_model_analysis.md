# Forecasting final model analysis

This document freezes the final forecasting benchmark interpretation after all
main model families were trained and tuned.

## Protocol

All retained comparisons use the same forecasting protocol:

```text
dataset: traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
granularity: subnet/slice
series: 179
slices: 4
frequency: 10 minutes
input_size: 2016 steps = 14 days
horizon: 36 steps = 6 hours
folds: 5 rolling-origin folds
fold_stride: 144 steps = 1 day
training scope: per_slice for learned panel models
```

There is no random split. The same temporal folds are shared across all models.
Generated benchmark artifacts are stored under `experiments/runs/` and should
not be committed.

## Retained runs

| family | retained run | role |
| --- | --- | --- |
| Baselines + LightGBM | `experiments/runs/deterministic_benchmark_lightgbm_tuned/` | deterministic and ML baseline |
| LSTM | `experiments/runs/lstm_benchmark_5000w/` | recurrent deep baseline |
| Prophet | `experiments/runs/prophet_benchmark_tuned/` | statistical/decomposition baseline |
| N-HiTS | `experiments/runs/nhits_benchmark_tuned/` | modern deep comparison |
| PatchTST | `experiments/runs/patchtst_benchmark_tuned/` | transformer-family comparison |

Sensitivity runs retained for interpretation:

```text
experiments/runs/nhits_benchmark_robust_tuned/
experiments/runs/patchtst_benchmark_fast/
experiments/runs/patchtst_benchmark_balanced/
experiments/runs/patchtst_benchmark_heavy1h/
```

## Global ranking

| rank by RMSE | model/run | RMSE | WAPE | MASE | interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Prophet tuned | 27.31M | 0.904 | 0.545 | best global RMSE and MASE |
| 2 | PatchTST tuned | 28.20M | 1.018 | 0.574 | best transformer-family result |
| 3 | LSTM 5000w | 29.02M | 0.992 | 0.563 | best recurrent deep baseline |
| 4 | LightGBM tuned | 29.31M | 0.876 | 0.600 | best global WAPE |
| 5 | PatchTST fast | 29.80M | 1.018 | 0.576 | useful PatchTST sensitivity |
| 6 | N-HiTS tuned | 33.87M | 1.811 | 0.650 | not retained as winner |
| 7 | N-HiTS robust tuned | 35.61M | 2.713 | 0.679 | confirms N-HiTS rejection |
| 8 | PatchTST balanced | 37.55M | 1.125 | 0.636 | larger PatchTST worsens |
| 9 | PatchTST heavy1h | 40.04M | 1.727 | 0.710 | heavy PatchTST worsens further |

Important nuance: persistence remains a strong RMSE reference because `eMBB`
dominates absolute traffic volume. Model selection should therefore consider
RMSE, WAPE, MASE, and per-slice behavior rather than only global RMSE.

## Per-slice winners

| slice | best RMSE | best WAPE | best MASE |
| --- | --- | --- | --- |
| URLLC | PatchTST tuned, by a negligible margin | LightGBM tuned | Prophet tuned |
| URLLC_eMBB_MIX | Prophet tuned | Prophet tuned | LightGBM tuned |
| eMBB | Prophet tuned | LightGBM tuned | Prophet tuned |
| mMTC | Prophet tuned | Prophet tuned | Prophet tuned |

Prophet is therefore the most regular model, but it does not strictly dominate
every metric on every slice. LightGBM remains important because it has the best
WAPE globally and on high-volume `eMBB`. PatchTST has a very small RMSE edge on
`URLLC`, but the difference versus Prophet is practically negligible.

## Critical-slice reading

If `eMBB` is excluded because the objective prioritizes critical slices, the
ranking on `URLLC`, `URLLC_eMBB_MIX`, and `mMTC` remains favorable to Prophet:

| model/run | RMSE | WAPE | MASE |
| --- | ---: | ---: | ---: |
| Prophet tuned | 12.09M | 0.873 | 0.567 |
| PatchTST fast | 12.11M | 0.935 | 0.585 |
| PatchTST tuned | 12.11M | 0.962 | 0.596 |
| LightGBM tuned | 12.12M | 0.890 | 0.490 |
| PatchTST heavy1h | 12.15M | 1.364 | 0.677 |
| LSTM 5000w | 12.16M | 0.976 | 0.577 |

This confirms that the heavy PatchTST run is not rejected only because `eMBB`
dominates global RMSE. It is also weaker on critical slices, especially in WAPE
and MASE.

## Model decisions

Prophet tuned is the preferred final forecasting model for the benchmark:

- best global RMSE;
- best global MASE;
- best or near-best behavior on most slices;
- low computational cost despite local per-series training;
- interpretable statistical/decomposition structure.

LightGBM tuned remains the best operational ML baseline:

- best global WAPE;
- fast training and inference;
- strong cost/performance ratio;
- useful if relative error and operational simplicity are prioritized over
  global RMSE.

LSTM 5000w is the retained recurrent deep baseline:

- competitive with Prophet and LightGBM;
- better than the initial 2500-window variant;
- 10000 windows does not justify replacing 5000 windows because `eMBB`
  degrades.

PatchTST tuned is the retained transformer-family result:

- second-best global RMSE among retained learned/statistical models;
- better than PatchTST fast on RMSE;
- worse than Prophet, LSTM, and LightGBM on WAPE;
- heavier PatchTST configurations degrade, so extra capacity does not help on
  this panel.

N-HiTS is not retained as a winner:

- tuned N-HiTS is worse than Prophet, PatchTST tuned, LSTM, and LightGBM on the
  main global metrics;
- robust tuning confirms that increasing tuning robustness does not rescue the
  model on this dataset.

## Final conclusion

The final thesis narrative should not be “deep learning always wins.” The
benchmark shows that a well-tuned statistical model, Prophet, is the strongest
overall choice under this subnet/slice protocol. LightGBM remains the strongest
relative-error and cost/performance baseline, while LSTM and PatchTST provide
credible deep-learning comparisons without dominating the full benchmark.

The final model recommendation is:

```text
Primary forecasting model: Prophet tuned
Operational ML baseline: LightGBM tuned
Deep-learning baseline: LSTM 5000w
Transformer comparison: PatchTST tuned
Rejected modern deep comparison: N-HiTS tuned / robust tuned
```

## Input-history sensitivity

A final sensitivity check varied only the available input history while keeping
the same 6-hour horizon and the same 5 rolling-origin folds:

```text
1d history: 144 steps
7d history: 1008 steps
14d history: 2016 steps
```

The sensitivity runs cover the retained model families:

```text
Prophet tuned
LightGBM tuned
LSTM 5000w
N-HiTS tuned
PatchTST tuned
```

Global results:

| model/run | 1d RMSE | 7d RMSE | 14d RMSE | best history by RMSE |
| --- | ---: | ---: | ---: | --- |
| Prophet tuned | 31.16M | 30.79M | 27.31M | 14d |
| LightGBM tuned | 33.90M | 33.73M | 29.31M | 14d |
| LSTM 5000w | 27.65M | 31.35M | 29.02M | 1d |
| N-HiTS tuned | 25.98M | 27.60M | 33.87M | 1d |
| PatchTST tuned | 27.80M | 29.19M | 28.20M | 1d |

Interpretation:

- Prophet and LightGBM benefit from the full 14-day history.
- LSTM, N-HiTS, and PatchTST perform better globally with a 1-day input window
  under the current tuned hyperparameters.
- This suggests that the dense subnet/slice panel contains strong short-term
  structure, while longer histories may introduce noise or make the deep models
  harder to optimize.
- The 1-day Prophet run required a specific stable configuration
  (`daily_seasonality=false`, `weekly_seasonality=false`, no log transform);
  otherwise it produced numerical blow-ups.
- The LightGBM 1-day MASE should not be used as a primary comparison because a
  daily seasonal scale cannot be estimated inside an exactly 144-step training
  window. RMSE and WAPE remain interpretable.

This sensitivity changes the modeling interpretation: the main 14-day protocol
is still the fixed fair comparison, but if the operational objective is purely
short-term 6-hour forecasting, the best deep/modern models should also be
reported with 1-day history. In that setting, N-HiTS 1d becomes the best global
RMSE run among the tested history sensitivities, while Prophet remains the best
model under the original 14-day protocol.

Final comparison artifacts:

```text
reports/model_comparison_global.csv
reports/model_comparison_by_slice.csv
reports/model_comparison_bias.csv
reports/history_sensitivity_global.csv
reports/history_sensitivity_by_slice.csv
docs/forecasting_benchmark_status.md
```
