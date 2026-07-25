# Final forecasting conclusion

This document freezes the report-level interpretation after deterministic and
probabilistic forecasting benchmarks.

## Experimental frame

The final forecasting study uses the subnet/slice panel built from the CESNET
relationship mapping:

```text
dataset: traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
granularity: id_institution_subnet + slice
series: 179
frequency: 10 minutes
horizon: 36 steps = 6 hours
folds: 5 rolling-origin folds
```

The reference deterministic setup uses 14 days of input history
(`input_size=2016`). A 1-day history scenario is kept as an operational
short-history sensitivity.

## Deterministic conclusion

The deterministic benchmark does not support a simple "deep learning always
wins" conclusion.

On the fixed 14-day protocol, **Prophet** is the strongest global deterministic
model:

- best global RMSE: 27.31M;
- best global MASE: 0.545;
- best or near-best behavior on most slices.

**LightGBM** remains the best operational machine-learning baseline:

- best global WAPE among retained deterministic learned models: 0.876;
- strong cost/performance ratio;
- especially useful when normalized error is more important than absolute RMSE.

**PatchTST** is the retained transformer comparison:

- second-best global deterministic RMSE: 28.20M;
- useful as evidence that a transformer-family model was tested under the same
  rolling-origin protocol;
- not better than Prophet globally and weaker than LightGBM on WAPE.

**LSTM** remains the recurrent deep-learning baseline. **N-HiTS** is retained as
a tested modern deep model, but not as a deterministic winner under the 14-day
reference protocol.

## Probabilistic conclusion

The probabilistic benchmark changes the model selection criterion. The best
median forecast is not necessarily the best uncertainty forecast.

**LightGBM quantile 14d** is the preferred operational probabilistic model:

- best global interval score: 33.37M;
- compact interval width: 6.36M;
- empirical coverage above the nominal central 80% interval: 0.835;
- strong WAPE: 0.893.

This makes LightGBM quantile 14d the most defensible model for proactive
resource allocation, because it provides a sharp and calibrated-enough upper
quantile (`q90`) that can be used as a conservative demand estimate.

**DeepAR 14d** is the strongest probabilistic deep-learning point-forecast
baseline:

- best global median RMSE among probabilistic runs: 26.11M;
- good MASE: 0.540;
- coverage: 0.855.

However, DeepAR intervals are much wider than LightGBM intervals. Its interval
score is therefore worse despite better median RMSE.

**Prophet intervals** are important as a statistical baseline, but the native
intervals are too wide in the 14-day scenario, especially for high-volume
`eMBB`. Prophet 1d improves interval score but has unstable WAPE and should not
be the primary probabilistic choice.

**PatchTST quantile** and **N-HiTS quantile** are useful deep-learning
sensitivity runs. Their 1-day variants are stronger than their 14-day variants,
which confirms the deterministic sensitivity finding that some deep models
benefit from shorter input histories. They do not beat LightGBM quantile 14d on
interval score or DeepAR 14d on median RMSE.

## Model recommendations

For the thesis narrative:

| role | recommended model | reason |
| --- | --- | --- |
| Deterministic final model | Prophet 14d | best global RMSE and MASE under the reference protocol |
| Operational deterministic baseline | LightGBM 14d | best WAPE and simple deployment profile |
| Probabilistic operational model | LightGBM quantile 14d | best interval score and compact uncertainty intervals |
| Probabilistic deep baseline | DeepAR 14d | best median RMSE among probabilistic models |
| Transformer comparison | PatchTST 14d deterministic, PatchTST 1d probabilistic sensitivity | credible transformer-family comparison without global dominance |
| Short-history sensitivity | N-HiTS 1d and PatchTST 1d | shows that deep models may prefer shorter histories |

For simulation integration:

```text
recommended demand signal: LightGBM quantile 14d q90
fallback point signal: DeepAR 14d q50 or Prophet 14d deterministic forecast
```

Using `q90` from LightGBM quantile is the most operationally coherent choice:
it directly encodes uncertainty and reduces the risk of under-allocating
resources before a traffic increase.

## Report artifacts

Final tables:

```text
reports/final_forecasting_tables.md
reports/model_comparison_global.csv
reports/model_comparison_by_slice.csv
reports/probabilistic_model_comparison_global.csv
reports/probabilistic_model_comparison_by_slice.csv
```

Final figures:

```text
reports/figures/deterministic_global_rmse.png
reports/figures/probabilistic_interval_score.png
reports/figures/probabilistic_coverage_width.png
```
