# Probabilistic benchmark status

This document records the current probabilistic forecasting state after running
all retained probabilistic candidates.

## Shared protocol

Probabilistic runs keep the deterministic benchmark structure:

```text
dataset: traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
granularity: subnet/slice
frequency: 10 minutes
horizon: 36 steps = 6 hours
folds: 5 rolling-origin folds
fold_stride: 144 steps = 1 day
quantiles: 0.1, 0.5, 0.9
interval: central 80%
```

Two input-history scenarios are used:

- `14d`: reference scientific comparison with `input_size=2016`.
- `1d`: short-history operational comparison with `input_size=144`.

## Completed runs

| model | scenario | run directory | status |
| --- | --- | --- | --- |
| LightGBM quantile | 14d | `experiments/runs/probabilistic_lightgbm_14d/` | completed |
| LightGBM quantile | 1d | `experiments/runs/probabilistic_lightgbm_1d/` | completed |
| DeepAR | 14d | `experiments/runs/probabilistic_deepar_14d/` | completed |
| DeepAR | 1d | `experiments/runs/probabilistic_deepar_1d/` | completed |
| Prophet native intervals | 14d | `experiments/runs/probabilistic_prophet_14d/` | completed |
| Prophet native intervals | 1d | `experiments/runs/probabilistic_prophet_1d/` | completed |
| PatchTST quantile | 14d | `experiments/runs/probabilistic_patchtst_14d/` | completed |
| PatchTST quantile | 1d | `experiments/runs/probabilistic_patchtst_1d/` | completed |
| N-HiTS quantile | 14d | `experiments/runs/probabilistic_nhits_14d/` | completed |
| N-HiTS quantile | 1d | `experiments/runs/probabilistic_nhits_1d/` | completed |

## Current global results

| model | scenario | RMSE median | WAPE median | coverage | interval width | interval score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM quantile | 14d | 32.16M | 0.893 | 0.835 | 6.36M | 33.37M |
| Prophet interval | 1d | 31.16M | 118.801 | 0.925 | 23.33M | 39.97M |
| LightGBM quantile | 1d | 33.96M | 0.984 | 0.885 | 6.17M | 46.65M |
| DeepAR | 14d | 26.11M | 0.958 | 0.855 | 43.21M | 65.36M |
| PatchTST quantile | 1d | 28.57M | 0.939 | 0.839 | 40.32M | 67.18M |
| N-HiTS quantile | 1d | 27.77M | 0.948 | 0.823 | 102.03M | 127.59M |
| N-HiTS quantile | 14d | 29.07M | 1.177 | 0.731 | 150.76M | 181.79M |
| DeepAR | 1d | 27.67M | 0.964 | 0.867 | 177.71M | 199.92M |
| Prophet interval | 14d | 27.31M | 0.904 | 0.859 | 280.39M | 303.15M |
| PatchTST quantile | 14d | 28.93M | 0.997 | 0.661 | 307.12M | 340.65M |

## Interpretation

DeepAR 14d gives the best global median RMSE among the completed probabilistic
runs. N-HiTS 1d and DeepAR 1d are also strong median-forecast deep baselines.

LightGBM quantile 14d gives the best global interval score. Its intervals are
much more compact than the neural and Prophet alternatives while keeping
coverage above the nominal 80% central interval. It is therefore the strongest
operational uncertainty baseline when interval sharpness matters.

Prophet 14d keeps its deterministic point-forecast strength but produces very
wide native intervals, especially on high-volume `eMBB`; this makes its global
interval score weak. Prophet 1d has high coverage and a better interval score
than Prophet 14d, but its WAPE is unstable and should not be used as a primary
selection metric.

PatchTST and N-HiTS quantile runs are useful as probabilistic deep-learning
comparisons, but they do not beat LightGBM 14d on interval score or DeepAR 14d
on global median RMSE.

## Per-slice reading

- `URLLC`: Prophet 14d, DeepAR 14d, and LightGBM 14d have almost identical
  interval scores; LightGBM 1d has the sharpest interval but slightly worse
  interval score.
- `URLLC_eMBB_MIX`: DeepAR 14d has the best interval score by a small margin,
  with LightGBM 14d and Prophet 14d very close.
- `eMBB`: LightGBM 14d is clearly best on interval score because the other
  models create much wider intervals.
- `mMTC`: Prophet 1d has the best interval score; N-HiTS 1d and LightGBM 14d are
  close alternatives.

## Consolidated artifacts

```text
reports/probabilistic_model_comparison_global.csv
reports/probabilistic_model_comparison_by_slice.csv
```

## Final recommendation

For probabilistic forecasting, retain:

- **LightGBM quantile 14d** as the primary operational uncertainty model because
  it has the best interval score and compact intervals.
- **DeepAR 14d** as the strongest probabilistic deep-learning point-forecast
  baseline because it has the best median RMSE.
- **Prophet 14d** as the statistical interval baseline, but discuss its interval
  width limitation.
- **PatchTST 1d** and **N-HiTS 1d** as short-history probabilistic deep-learning
  sensitivity runs.
