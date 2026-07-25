# Input-history sensitivity tables

Histories: `1d` = 144 points, `7d` = 1008 points, `14d` = 2016 points. All results use the same 6-hour horizon and 5 rolling-origin folds.

Note: MASE for the `1d` LightGBM/deterministic run is not a primary decision metric because the seasonal scale cannot be estimated normally from a one-day history window. RMSE and WAPE remain interpretable.

## Best model and history by slice

| Slice          | RMSE history | RMSE model    | RMSE   | WAPE history | WAPE model     | WAPE  | MASE history | MASE model     | MASE  |
| -------------- | ------------ | ------------- | ------ | ------------ | -------------- | ----- | ------------ | -------------- | ----- |
| URLLC          | 7d           | N-HiTS tuned  | 1.52M  | 7d           | LightGBM tuned | 0.999 | 7d           | LSTM 5000w     | 0.096 |
| URLLC_eMBB_MIX | 14d          | Prophet tuned | 34.76M | 7d           | PatchTST tuned | 0.729 | 14d          | LightGBM tuned | 0.441 |
| eMBB           | 1d           | Prophet tuned | 67.32M | 14d          | LightGBM tuned | 0.835 | 1d           | N-HiTS tuned   | 0.445 |
| mMTC           | 1d           | Prophet tuned | 4.28k  | 1d           | PatchTST tuned | 0.826 | 1d           | PatchTST tuned | 0.628 |

## Per-slice winners among learned models

| History | Slice          | Best RMSE model | RMSE   | Best WAPE model | WAPE  | Best MASE model | MASE  |
| ------- | -------------- | --------------- | ------ | --------------- | ----- | --------------- | ----- |
| 1d      | URLLC          | N-HiTS tuned    | 1.52M  | LightGBM tuned  | 1.002 | PatchTST tuned  | 0.096 |
| 1d      | URLLC_eMBB_MIX | PatchTST tuned  | 34.78M | PatchTST tuned  | 0.747 | PatchTST tuned  | 0.967 |
| 1d      | eMBB           | Prophet tuned   | 67.32M | LSTM 5000w      | 0.839 | N-HiTS tuned    | 0.445 |
| 1d      | mMTC           | Prophet tuned   | 4.28k  | PatchTST tuned  | 0.826 | PatchTST tuned  | 0.628 |
| 7d      | URLLC          | N-HiTS tuned    | 1.52M  | LightGBM tuned  | 0.999 | LSTM 5000w      | 0.096 |
| 7d      | URLLC_eMBB_MIX | PatchTST tuned  | 34.78M | PatchTST tuned  | 0.729 | PatchTST tuned  | 0.966 |
| 7d      | eMBB           | N-HiTS tuned    | 74.06M | LightGBM tuned  | 0.984 | N-HiTS tuned    | 0.488 |
| 7d      | mMTC           | N-HiTS tuned    | 4.47k  | LSTM 5000w      | 0.852 | N-HiTS tuned    | 0.647 |
| 14d     | URLLC          | PatchTST tuned  | 1.52M  | LightGBM tuned  | 1.003 | Prophet tuned   | 0.096 |
| 14d     | URLLC_eMBB_MIX | Prophet tuned   | 34.76M | Prophet tuned   | 0.764 | LightGBM tuned  | 0.441 |
| 14d     | eMBB           | Prophet tuned   | 72.94M | LightGBM tuned  | 0.835 | Prophet tuned   | 0.480 |
| 14d     | mMTC           | Prophet tuned   | 4.53k  | Prophet tuned   | 0.835 | Prophet tuned   | 0.638 |

## Best history per model and slice

| Slice          | Model          | Best RMSE history | RMSE   | Best WAPE history | WAPE  | Best MASE history | MASE  |
| -------------- | -------------- | ----------------- | ------ | ----------------- | ----- | ----------------- | ----- |
| URLLC          | N-HiTS tuned   | 7d                | 1.52M  | 1d                | 1.020 | 1d                | 0.096 |
| URLLC          | PatchTST tuned | 14d               | 1.52M  | 1d                | 1.005 | 1d                | 0.096 |
| URLLC          | Prophet tuned  | 14d               | 1.52M  | 14d               | 1.020 | 14d               | 0.096 |
| URLLC          | LSTM 5000w     | 1d                | 1.52M  | 7d                | 1.006 | 7d                | 0.096 |
| URLLC          | LightGBM tuned | 14d               | 1.52M  | 7d                | 0.999 | 7d                | 0.253 |
| URLLC_eMBB_MIX | Prophet tuned  | 14d               | 34.76M | 14d               | 0.764 | 14d               | 0.967 |
| URLLC_eMBB_MIX | PatchTST tuned | 7d                | 34.78M | 7d                | 0.729 | 7d                | 0.966 |
| URLLC_eMBB_MIX | N-HiTS tuned   | 1d                | 34.81M | 7d                | 0.770 | 1d                | 0.969 |
| URLLC_eMBB_MIX | LightGBM tuned | 14d               | 34.83M | 14d               | 0.784 | 14d               | 0.441 |
| URLLC_eMBB_MIX | LSTM 5000w     | 7d                | 34.90M | 1d                | 0.915 | 7d                | 0.976 |
| eMBB           | Prophet tuned  | 1d                | 67.32M | 14d               | 0.998 | 14d               | 0.480 |
| eMBB           | N-HiTS tuned   | 1d                | 67.58M | 1d                | 1.092 | 1d                | 0.445 |
| eMBB           | LSTM 5000w     | 1d                | 74.17M | 1d                | 0.839 | 1d                | 0.494 |
| eMBB           | PatchTST tuned | 1d                | 74.88M | 1d                | 1.150 | 1d                | 0.489 |
| eMBB           | LightGBM tuned | 14d               | 80.87M | 14d               | 0.835 | 14d               | 0.932 |
| mMTC           | Prophet tuned  | 1d                | 4.28k  | 14d               | 0.835 | 14d               | 0.638 |
| mMTC           | PatchTST tuned | 1d                | 4.40k  | 1d                | 0.826 | 1d                | 0.628 |
| mMTC           | N-HiTS tuned   | 1d                | 4.43k  | 1d                | 0.852 | 1d                | 0.642 |
| mMTC           | LSTM 5000w     | 1d                | 4.61k  | 14d               | 0.848 | 14d               | 0.649 |
| mMTC           | LightGBM tuned | 14d               | 4.75k  | 14d               | 0.883 | 14d               | 0.752 |

## Global metrics, appendix

| History | Model          | RMSE   | WAPE    | MASE        |
| ------- | -------------- | ------ | ------- | ----------- |
| 1d      | N-HiTS tuned   | 25.98M | 0.936   | 0.538       |
| 1d      | LSTM 5000w     | 27.65M | 0.907   | 0.556       |
| 1d      | PatchTST tuned | 27.80M | 0.932   | 0.545       |
| 1d      | Prophet tuned  | 31.16M | 118.801 | 0.949       |
| 1d      | LightGBM tuned | 33.90M | 0.985   | 5681536.605 |
| 7d      | N-HiTS tuned   | 27.60M | 0.993   | 0.550       |
| 7d      | PatchTST tuned | 29.19M | 0.956   | 0.560       |
| 7d      | Prophet tuned  | 30.79M | 1.023   | 0.589       |
| 7d      | LSTM 5000w     | 31.35M | 1.018   | 0.575       |
| 7d      | LightGBM tuned | 33.73M | 0.964   | 0.765       |
| 14d     | Prophet tuned  | 27.31M | 0.904   | 0.545       |
| 14d     | PatchTST tuned | 28.20M | 1.018   | 0.574       |
| 14d     | LSTM 5000w     | 29.02M | 0.992   | 0.563       |
| 14d     | LightGBM tuned | 29.31M | 0.876   | 0.600       |
| 14d     | N-HiTS tuned   | 33.87M | 1.811   | 0.650       |

## Global best learned model by history, appendix

| History | Criterion | Best model     | Value  |
| ------- | --------- | -------------- | ------ |
| 1d      | MASE      | N-HiTS tuned   | 0.538  |
| 1d      | RMSE      | N-HiTS tuned   | 25.98M |
| 1d      | WAPE      | LSTM 5000w     | 0.907  |
| 7d      | MASE      | N-HiTS tuned   | 0.550  |
| 7d      | RMSE      | N-HiTS tuned   | 27.60M |
| 7d      | WAPE      | PatchTST tuned | 0.956  |
| 14d     | MASE      | Prophet tuned  | 0.545  |
| 14d     | RMSE      | Prophet tuned  | 27.31M |
| 14d     | WAPE      | LightGBM tuned | 0.876  |

## Global best history per learned model, appendix

| Model          | Best RMSE history | RMSE   | Best WAPE history | WAPE  | Best MASE history | MASE  |
| -------------- | ----------------- | ------ | ----------------- | ----- | ----------------- | ----- |
| N-HiTS tuned   | 1d                | 25.98M | 1d                | 0.936 | 1d                | 0.538 |
| Prophet tuned  | 14d               | 27.31M | 14d               | 0.904 | 14d               | 0.545 |
| LSTM 5000w     | 1d                | 27.65M | 1d                | 0.907 | 1d                | 0.556 |
| PatchTST tuned | 1d                | 27.80M | 1d                | 0.932 | 1d                | 0.545 |
| LightGBM tuned | 14d               | 29.31M | 14d               | 0.876 | 14d               | 0.600 |

## Conclusion

Per-slice RMSE winners are led by **Prophet tuned** (3 of 4 slices).
Per-slice WAPE winners are led by **LightGBM tuned, PatchTST tuned** (2 of 4 slices).
Per-slice MASE winners are tied between **LSTM 5000w, LightGBM tuned, N-HiTS tuned, PatchTST tuned** (1 slice each among the leaders).

Per-slice selection:
- `URLLC`: RMSE -> N-HiTS tuned (7d); WAPE -> LightGBM tuned (7d); MASE -> LSTM 5000w (7d).
- `URLLC_eMBB_MIX`: RMSE -> Prophet tuned (14d); WAPE -> PatchTST tuned (7d); MASE -> LightGBM tuned (14d).
- `eMBB`: RMSE -> Prophet tuned (1d); WAPE -> LightGBM tuned (14d); MASE -> N-HiTS tuned (1d).
- `mMTC`: RMSE -> Prophet tuned (1d); WAPE -> PatchTST tuned (1d); MASE -> PatchTST tuned (1d).

The choice is therefore slice-dependent: use the per-slice table above for model selection, not the aggregate global score.
