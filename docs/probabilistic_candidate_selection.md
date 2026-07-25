# Probabilistic candidate selection

This document selects a small set of deterministic models for a probabilistic forecasting extension. SARIMA is intentionally not included in the implemented benchmark; Prophet is the retained statistical baseline.

## Option C: probabilistic benchmark protocol

The probabilistic benchmark should be run in two scenarios:

- **Reference scenario (`14d`)**: same setup as the deterministic benchmark, used for scientific comparability.
- **Short-history scenario (`1d`)**: operational short-term setting motivated by the deterministic sensitivity analysis.

This avoids forcing every probabilistic model into the 14-day window when the deterministic study shows that short-history deep models can be stronger.

### Best deterministic history by slice

| slice          | best_rmse_history | best_rmse_model | best_rmse_display | best_wape_history | best_wape_model | best_wape_display | best_mase_history | best_mase_model | best_mase_display |
| -------------- | ----------------- | --------------- | ----------------- | ----------------- | --------------- | ----------------- | ----------------- | --------------- | ----------------- |
| URLLC          | 7d                | N-HiTS tuned    | 1.52M             | 7d                | LightGBM tuned  | 0.999             | 7d                | LSTM 5000w      | 0.096             |
| URLLC_eMBB_MIX | 14d               | Prophet tuned   | 34.76M            | 7d                | PatchTST tuned  | 0.729             | 14d               | LightGBM tuned  | 0.441             |
| eMBB           | 1d                | Prophet tuned   | 67.32M            | 14d               | LightGBM tuned  | 0.835             | 1d                | N-HiTS tuned    | 0.445             |
| mMTC           | 1d                | Prophet tuned   | 4.28k             | 1d                | PatchTST tuned  | 0.826             | 1d                | PatchTST tuned  | 0.628             |

### Models favored by short histories

| slice          | model_label    | best_history_rmse | best_rmse_display | best_history_wape | best_wape_display | best_history_mase | best_mase_display |
| -------------- | -------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| URLLC          | N-HiTS tuned   | 7d                | 1.52M             | 1d                | 1.02              | 1d                | 0.096             |
| URLLC          | LSTM 5000w     | 1d                | 1.52M             | 7d                | 1.006             | 7d                | 0.096             |
| URLLC_eMBB_MIX | PatchTST tuned | 7d                | 34.78M            | 7d                | 0.729             | 7d                | 0.966             |
| URLLC_eMBB_MIX | N-HiTS tuned   | 1d                | 34.81M            | 7d                | 0.77              | 1d                | 0.969             |
| URLLC_eMBB_MIX | LSTM 5000w     | 7d                | 34.90M            | 1d                | 0.915             | 7d                | 0.976             |
| eMBB           | Prophet tuned  | 1d                | 67.32M            | 14d               | 0.998             | 14d               | 0.48              |
| eMBB           | N-HiTS tuned   | 1d                | 67.58M            | 1d                | 1.092             | 1d                | 0.445             |
| eMBB           | LSTM 5000w     | 1d                | 74.17M            | 1d                | 0.839             | 1d                | 0.494             |
| eMBB           | PatchTST tuned | 1d                | 74.88M            | 1d                | 1.15              | 1d                | 0.489             |
| mMTC           | Prophet tuned  | 1d                | 4.28k             | 14d               | 0.835             | 14d               | 0.638             |
| mMTC           | PatchTST tuned | 1d                | 4.40k             | 1d                | 0.826             | 1d                | 0.628             |
| mMTC           | N-HiTS tuned   | 1d                | 4.43k             | 1d                | 0.852             | 1d                | 0.642             |
| mMTC           | LSTM 5000w     | 1d                | 4.61k             | 14d               | 0.848             | 14d               | 0.649             |

## Selection Rule

- Primary unit of decision: slice-level performance, not the aggregate global score.
- A model is considered competitive on a slice if its RMSE is within 3% of the best slice RMSE.
- WAPE and MASE are used as secondary checks with 10% and 10% margins.
- Fold stability is measured with the coefficient of variation of fold-level RMSE.
- The final probabilistic set should cover complementary families, not only the best deterministic WAPE.
- The 14-day reference scenario selects robust deterministic candidates; the 1-day scenario re-opens short-history candidates such as N-HiTS.

## Per-slice deterministic reading

| slice          | best_rmse_model | best_rmse_display | best_wape_model | best_wape_display | within_3pct_rmse                                                        |
| -------------- | --------------- | ----------------- | --------------- | ----------------- | ----------------------------------------------------------------------- |
| URLLC          | PatchTST tuned  | 1.52M             | LightGBM tuned  | 1.003             | PatchTST tuned, Prophet tuned, LightGBM tuned, LSTM 5000w, N-HiTS tuned |
| URLLC_eMBB_MIX | Prophet tuned   | 34.76M            | Prophet tuned   | 0.764             | Prophet tuned, PatchTST tuned, LightGBM tuned, LSTM 5000w               |
| eMBB           | Prophet tuned   | 72.94M            | LightGBM tuned  | 0.835             | Prophet tuned                                                           |
| mMTC           | Prophet tuned   | 4.53k             | Prophet tuned   | 0.835             | Prophet tuned, LSTM 5000w                                               |

## Recommended probabilistic candidates

| model          | family           | slice_rmse_margin_count | slice_wape_margin_count | slice_mase_margin_count | mean_rmse_gap_display | mean_rmse_fold_cv_display | probabilistic_path                                                                              |
| -------------- | ---------------- | ----------------------- | ----------------------- | ----------------------- | --------------------- | ------------------------- | ----------------------------------------------------------------------------------------------- |
| Prophet tuned  | statistical      | 4                       | 3                       | 3                       | 0.0%                  | 0.515                     | Native prediction intervals; calibrate coverage per slice because deterministic runs show bias. |
| LightGBM tuned | machine_learning | 2                       | 4                       | 1                       | 4.0%                  | 0.527                     | Quantile regression with separate alpha models, or conformalized residual intervals.            |
| PatchTST tuned | deep_transformer | 2                       | 1                       | 2                       | 5.7%                  | 0.531                     | NeuralForecast quantile/distribution losses; natural probabilistic transformer candidate.       |

## Not retained as primary probabilistic candidates

| model        | family         | mean_rmse_gap_display | probabilistic_path                                                                                           |
| ------------ | -------------- | --------------------- | ------------------------------------------------------------------------------------------------------------ |
| LSTM 5000w   | deep_recurrent | 2.9%                  | Quantile loss head or distributional output head; requires extra implementation.                             |
| N-HiTS tuned | deep_mlp       | 16.6%                 | NeuralForecast quantile/distribution losses; stronger as a 1-day sensitivity than under the 14-day protocol. |

## Detailed slice scores

| slice          | model_label    | RMSE_display | RMSE_gap_display | WAPE_display | WAPE_gap_display | MASE_display | MASE_gap_display | RMSE_cv_display | robust_score |
| -------------- | -------------- | ------------ | ---------------- | ------------ | ---------------- | ------------ | ---------------- | --------------- | ------------ |
| URLLC          | PatchTST tuned | 1.52M        | 0.0%             | 1.152        | 14.8%            | 0.096        | 0.0%             | 0.881           | 2            |
| URLLC          | Prophet tuned  | 1.52M        | 0.0%             | 1.020        | 1.6%             | 0.096        | 0.0%             | 0.881           | 3            |
| URLLC          | LightGBM tuned | 1.52M        | 0.0%             | 1.003        | 0.0%             | 0.277        | 188.7%           | 0.881           | 2            |
| URLLC          | LSTM 5000w     | 1.52M        | 0.0%             | 1.015        | 1.2%             | 0.096        | 0.0%             | 0.881           | 3            |
| URLLC          | N-HiTS tuned   | 1.52M        | 0.6%             | 2.654        | 164.6%           | 0.097        | 1.2%             | 0.872           | 2            |
| URLLC_eMBB_MIX | Prophet tuned  | 34.76M       | 0.0%             | 0.764        | 0.0%             | 0.967        | 119.5%           | 0.857           | 2            |
| URLLC_eMBB_MIX | PatchTST tuned | 34.81M       | 0.1%             | 0.778        | 1.8%             | 0.969        | 120.0%           | 0.855           | 2            |
| URLLC_eMBB_MIX | LightGBM tuned | 34.83M       | 0.2%             | 0.784        | 2.6%             | 0.441        | 0.0%             | 0.855           | 3            |
| URLLC_eMBB_MIX | LSTM 5000w     | 34.97M       | 0.6%             | 1.066        | 39.5%            | 0.987        | 123.9%           | 0.853           | 1            |
| URLLC_eMBB_MIX | N-HiTS tuned   | 36.39M       | 4.7%             | 2.011        | 163.2%           | 1.037        | 135.4%           | 0.806           | 0            |
| eMBB           | Prophet tuned  | 72.94M       | 0.0%             | 0.998        | 19.5%            | 0.480        | 0.0%             | 0.297           | 2            |
| eMBB           | PatchTST tuned | 76.48M       | 4.8%             | 1.185        | 41.9%            | 0.506        | 5.5%             | 0.284           | 1            |
| eMBB           | LSTM 5000w     | 79.59M       | 9.1%             | 1.040        | 24.6%            | 0.522        | 8.9%             | 0.346           | 1            |
| eMBB           | LightGBM tuned | 80.87M       | 10.9%            | 0.835        | 0.0%             | 0.932        | 94.4%            | 0.340           | 1            |
| eMBB           | N-HiTS tuned   | 97.55M       | 33.7%            | 1.536        | 84.1%            | 0.689        | 43.6%            | 0.583           | 0            |
| mMTC           | Prophet tuned  | 4.53k        | 0.0%             | 0.835        | 0.0%             | 0.638        | 0.0%             | 0.026           | 3            |
| mMTC           | LSTM 5000w     | 4.62k        | 2.0%             | 0.848        | 1.6%             | 0.649        | 1.8%             | 0.025           | 3            |
| mMTC           | LightGBM tuned | 4.75k        | 4.8%             | 0.883        | 5.8%             | 0.752        | 17.8%            | 0.031           | 1            |
| mMTC           | PatchTST tuned | 5.33k        | 17.7%            | 0.958        | 14.7%            | 0.724        | 13.5%            | 0.104           | 0            |
| mMTC           | N-HiTS tuned   | 5.77k        | 27.5%            | 1.044        | 25.0%            | 0.777        | 21.8%            | 0.350           | 0            |

## Paired fold-horizon comparison against the best RMSE model

This sign test is used as a conservative diagnostic, not as a definitive statistical proof, because adjacent horizons are not fully independent.

| slice          | best_model     | model          | relative_rmse_delta_display | wins_vs_best | losses_vs_best | ties_vs_best | sign_test_pvalue_display | statistically_different_5pct |
| -------------- | -------------- | -------------- | --------------------------- | ------------ | -------------- | ------------ | ------------------------ | ---------------------------- |
| URLLC          | PatchTST tuned | PatchTST tuned | 0.00%                       | 0            | 0              | 180          |                          | False                        |
| URLLC          | PatchTST tuned | Prophet tuned  | 0.00%                       | 106          | 74             | 0            | 0.0206                   | True                         |
| URLLC          | PatchTST tuned | LightGBM tuned | 0.00%                       | 75           | 105            | 0            | 0.0304                   | True                         |
| URLLC          | PatchTST tuned | LSTM 5000w     | 0.00%                       | 93           | 87             | 0            | 0.7095                   | False                        |
| URLLC          | PatchTST tuned | N-HiTS tuned   | 0.57%                       | 50           | 130            | 0            | 0.0000                   | True                         |
| URLLC_eMBB_MIX | Prophet tuned  | Prophet tuned  | 0.00%                       | 0            | 0              | 180          |                          | False                        |
| URLLC_eMBB_MIX | Prophet tuned  | PatchTST tuned | 0.14%                       | 77           | 103            | 0            | 0.0621                   | False                        |
| URLLC_eMBB_MIX | Prophet tuned  | LightGBM tuned | 0.21%                       | 70           | 110            | 0            | 0.0035                   | True                         |
| URLLC_eMBB_MIX | Prophet tuned  | LSTM 5000w     | 0.60%                       | 35           | 145            | 0            | 0.0000                   | True                         |
| URLLC_eMBB_MIX | Prophet tuned  | N-HiTS tuned   | 4.69%                       | 58           | 122            | 0            | 0.0000                   | True                         |
| eMBB           | Prophet tuned  | Prophet tuned  | 0.00%                       | 0            | 0              | 180          |                          | False                        |
| eMBB           | Prophet tuned  | PatchTST tuned | 4.85%                       | 78           | 102            | 0            | 0.0862                   | False                        |
| eMBB           | Prophet tuned  | LSTM 5000w     | 9.12%                       | 78           | 102            | 0            | 0.0862                   | False                        |
| eMBB           | Prophet tuned  | LightGBM tuned | 10.87%                      | 39           | 141            | 0            | 0.0000                   | True                         |
| eMBB           | Prophet tuned  | N-HiTS tuned   | 33.74%                      | 70           | 110            | 0            | 0.0035                   | True                         |
| mMTC           | Prophet tuned  | Prophet tuned  | 0.00%                       | 0            | 0              | 180          |                          | False                        |
| mMTC           | Prophet tuned  | LSTM 5000w     | 1.96%                       | 58           | 122            | 0            | 0.0000                   | True                         |
| mMTC           | Prophet tuned  | LightGBM tuned | 4.83%                       | 40           | 140            | 0            | 0.0000                   | True                         |
| mMTC           | Prophet tuned  | PatchTST tuned | 17.68%                      | 59           | 121            | 0            | 0.0000                   | True                         |
| mMTC           | Prophet tuned  | N-HiTS tuned   | 27.47%                      | 57           | 123            | 0            | 0.0000                   | True                         |

## Final recommendation for the thesis

Retain the following candidates:

- **Prophet tuned, 14d and 1d** as the statistical interval baseline. It is strongest under the 14-day reference and remains important for slice-level RMSE. The interval calibration should be checked because the deterministic benchmark shows systematic bias.
- **LightGBM tuned, 14d** as the operational quantile-regression baseline. It is cheap, robust, and strong on WAPE, especially for high-volume traffic. It should not be selected only because of WAPE, but because it gives a simple and defensible probabilistic ML baseline.
- **PatchTST tuned, 14d and 1d** as the transformer probabilistic candidate. The 14-day version preserves comparability; the 1-day version tests the operational short-history regime.
- **N-HiTS tuned, 1d only** as a secondary short-history probabilistic candidate. It is not retained under the fixed 14-day protocol, but it becomes competitive in the deterministic 1-day sensitivity.
- **DeepAR, 14d and optionally 1d** as a probabilistic deep baseline. It was not in the deterministic benchmark because it is inherently probabilistic, but it is useful as a standard distribution-learning baseline.

Keep **LSTM 5000w** as a deterministic deep baseline, but do not make it a primary probabilistic candidate unless implementation time allows. Its probabilistic version requires a custom quantile or distributional output head and is less directly available than DeepAR, N-HiTS, or PatchTST.
