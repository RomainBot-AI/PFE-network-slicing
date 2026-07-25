# Forecasting benchmark status

State after PatchTST tuning.

## Fixed Protocol

All accepted benchmark runs use the same temporal protocol:

```text
dataset: traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
granularity: subnet/slice
series: 179
slices: 4
input_size: 2016 steps = 14 days
horizon: 36 steps = 6 hours
folds: 5 rolling-origin folds
fold_stride: 144 steps = 1 day
training scope: per_slice
```

No random split is used. The folds are shared across slices and models.

## Completed Models

| family | retained run | status |
| --- | --- | --- |
| Baselines + LightGBM | `experiments/runs/deterministic_benchmark_lightgbm_tuned/` | accepted ML baseline |
| LSTM | `experiments/runs/lstm_benchmark_5000w/` | accepted deep baseline |
| Prophet | `experiments/runs/prophet_benchmark_tuned/` | accepted statistical baseline |
| N-HiTS | `experiments/runs/nhits_benchmark_tuned/` | tested, not retained as best |
| N-HiTS robust | `experiments/runs/nhits_benchmark_robust_tuned/` | sensitivity check, not retained |
| PatchTST tuned | `experiments/runs/patchtst_benchmark_tuned/` | retained transformer-family comparison |
| PatchTST fast | `experiments/runs/patchtst_benchmark_fast/` | transformer-family budget-controlled comparison |
| PatchTST balanced | `experiments/runs/patchtst_benchmark_balanced/` | sensitivity check, not retained |
| PatchTST heavy1h | `experiments/runs/patchtst_benchmark_heavy1h/` | heavy sensitivity check, not retained |

Generated run folders under `experiments/runs/` are artifacts and should not be
committed.

## Current Global Results

| model/run | RMSE | WAPE | MASE | decision |
| --- | ---: | ---: | ---: | --- |
| Prophet tuned | 27.31M | 0.904 | 0.545 | best RMSE/MASE among current learned/statistical models |
| PatchTST tuned | 28.20M | 1.018 | 0.574 | best transformer-family run, not overall winner |
| LSTM 5000w | 29.02M | 0.992 | 0.563 | best deep baseline |
| LightGBM tuned | 29.31M | 0.876 | 0.600 | best WAPE among learned models |
| PatchTST fast | 29.80M | 1.018 | 0.576 | transformer comparison, not best |
| N-HiTS tuned | 33.87M | 1.811 | 0.650 | keep for comparison only |
| N-HiTS robust tuned | 35.61M | 2.713 | 0.679 | confirms rejection |
| PatchTST balanced | 37.55M | 1.125 | 0.636 | larger PatchTST worsens, reject |
| PatchTST heavy1h | 40.04M | 1.727 | 0.710 | heavy architecture worsens, reject |

Important nuance: persistence remains a strong RMSE reference because high-volume
`eMBB` dominates absolute errors. Learned models should therefore be discussed
both globally and per slice.

## Current Per-Slice Reading

LightGBM tuned:

- Strong normalized error and WAPE, especially on `URLLC_eMBB_MIX`.
- Very good cost/performance ratio.
- Still weaker than persistence on global RMSE because of `eMBB`.

LSTM 5000w:

- Best retained deep model.
- Better than the 2500-window variant.
- 10000 windows does not justify replacing 5000 windows because `eMBB` degrades.
- Competitive against LightGBM, but not uniformly better.

Prophet:

- Best current global RMSE and MASE among the learned/statistical models.
- Better than LSTM 5000w globally on RMSE, WAPE, and MASE.
- Does not beat tuned LightGBM on global WAPE.
- Mini-grid tuning selected `changepoint_prior_scale=0.01`,
  `seasonality_prior_scale=10.0`, and additive seasonality.
- Local per-series training is cheap enough for this panel: 895 fitted models,
  44.43 s total train time and 11.89 s total inference time.
- Per-slice result is especially strong on `eMBB` RMSE/MASE, but Prophet has
  systematic negative bias on all slices.

N-HiTS:

- First tuning did not beat LSTM on any slice.
- Robust tuning improved only `URLLC_eMBB_MIX`.
- Robust tuning worsened `eMBB`, `mMTC`, and `URLLC`, so global performance got
  worse.
- Conclusion: keep N-HiTS in the story as a modern deep model tested under the
  same rigorous protocol, but do not retain it as a candidate winner.

PatchTST fast:

- Uses the same 5-fold, 36-horizon, per-slice protocol as the other deep models.
- Budget-controlled configuration: `max_steps=100`, `encoder_layers=2`,
  `hidden_size=64`, `patch_len=32`, `stride=16`.
- Global result is competitive but not better than Prophet, LSTM, or tuned
  LightGBM.
- Training cost is low for the retained fast run: 20 fitted models, 69.27 s
  total train time and 1.89 s total inference time.
- The larger `max_steps=300`, `hidden_size=128` configuration was much more
  expensive and is not needed for the first transformer-family comparison.

PatchTST tuned:

- Small candidate search on the first fold selected the `more_steps`
  configuration: `max_steps=180`, `learning_rate=0.0003`, `hidden_size=64`,
  `encoder_layers=2`, `patch_len=32`, `stride=16`.
- Completed the same 20-model protocol in 115.76 s train time and 1.90 s
  inference time.
- Improves RMSE over PatchTST fast, mainly via `eMBB`, but WAPE remains weaker
  than Prophet, LSTM, and tuned LightGBM.
- Retain this as the main PatchTST result; keep fast and balanced as sensitivity
  runs.

PatchTST balanced:

- Uses a middle configuration: `max_steps=200`, `encoder_layers=3`,
  `hidden_size=96`, `patch_len=32`, `stride=16`.
- Completed the same 20-model protocol in 348.88 s train time and 2.00 s
  inference time.
- It worsened global RMSE, WAPE, and MASE versus PatchTST fast, mainly because
  `eMBB` degraded sharply.
- Conclusion: keep it as a sensitivity check showing that increasing PatchTST
  capacity/budget did not improve this panel benchmark.

PatchTST heavy1h:

- Heavier configuration: `max_steps=300`, `encoder_layers=3`,
  `hidden_size=96`, `patch_len=16`, `stride=8`.
- Completed the same 20-model protocol in 1413.29 s train time and 2.03 s
  inference time.
- It worsened all global metrics versus tuned, fast, and balanced PatchTST.
  The degradation is driven mostly by `eMBB` and `mMTC`.
- Conclusion: increasing PatchTST patch resolution and training budget does not
  help on the current subnet/slice panel.

## Reports Generated

HTML reports:

```text
experiments/runs/deterministic_benchmark_lightgbm_tuned/benchmark_report.html
experiments/runs/lstm_benchmark_5000w/benchmark_report.html
experiments/runs/prophet_benchmark_tuned/benchmark_report.html
experiments/runs/nhits_benchmark_tuned/benchmark_report.html
experiments/runs/nhits_benchmark_robust_tuned/benchmark_report.html
experiments/runs/patchtst_benchmark_tuned/benchmark_report.html
experiments/runs/patchtst_benchmark_fast/benchmark_report.html
experiments/runs/patchtst_benchmark_balanced/benchmark_report.html
experiments/runs/patchtst_benchmark_heavy1h/benchmark_report.html
```

Model-specific docs:

```text
docs/lightgbm_benchmark_summary.md
docs/lstm_benchmark_summary.md
docs/nhits_benchmark_plan.md
```

## Next Work

The main model families have now been tested under the shared protocol.

Before final report tables:

- merge all retained `benchmark_summary_by_slice.csv` files;
- produce one global comparison table;
- produce one per-slice comparison table;
- include training/inference time and parameter counts where available.
