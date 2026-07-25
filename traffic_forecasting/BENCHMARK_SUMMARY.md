# Forecasting benchmark summary

This summary records the current direct benchmark after tuning the available slice-level models.

Protocol:

- input: `traffic_forecasting/data/slice_traffic_10min.csv`
- target: `n_bytes` per slice
- split: chronological
- test size: 144 points
- horizons: 1, 6, 12 steps
- evaluation mode: `direct`
- latest benchmark: full train history for trained models, GPU run
- scope: slice-level baseline only, 4 aggregated time series

Tuned model families:

- `naive`
- `seasonal_naive`
- `moving_average`
- `nhits`
- `prophet`
- `lstm`

## Best Model By Slice And Horizon

| slice | horizon | best model |
| --- | ---: | --- |
| URLLC | 1 | naive |
| URLLC | 6 | nhits |
| URLLC | 12 | nhits |
| URLLC_eMBB_MIX | 1 | moving_average |
| URLLC_eMBB_MIX | 6 | moving_average |
| URLLC_eMBB_MIX | 12 | lstm |
| eMBB | 1 | seasonal_naive |
| eMBB | 6 | seasonal_naive |
| eMBB | 12 | prophet |
| mMTC | 1 | nhits |
| mMTC | 6 | nhits |
| mMTC | 12 | moving_average |

Win count:

- `nhits`: 4
- `moving_average`: 3
- `seasonal_naive`: 2
- `naive`: 1
- `lstm`: 1
- `prophet`: 1

## Interpretation

No single model dominates the tuned benchmark, but the full-history GPU run changes the conclusion about N-HiTS: it becomes the strongest single model by win count.

This benchmark is still useful as a baseline, but it is not the final modeling setup. The sliced dataset contains `1000` distinct `ip_id` values and `3938` observed `slice/ip_id` series, so the forecasting model should learn from the panel and aggregate predictions back to slice demand.

For this project, the most defensible next step is to benchmark `ip_id`-level panel models against these slice-only results. A selection layer by slice and horizon remains useful, but model selection should include the panel runs before PPO integration.

## Panel Extension

New scripts:

- `traffic_forecasting/build_ip_slice_series.py`
- `traffic_forecasting/build_subnet_slice_series.py`
- `traffic_forecasting/forecast_ip_slice_nhits.py`

The panel format is:

```text
unique_id,ds,y,slice,ip_id
```

Preferred panel granularity is now `id_institution_subnet + slice`, using the CESNET relationship mapping:

```text
ForecastingDoc/ids_relationship.csv
```

Subnet/slice outputs:

```text
traffic_forecasting/data/subnet_slice_traffic_10min_long.csv
traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
traffic_forecasting/data/slice_traffic_from_subnet_min2016.csv
traffic_forecasting/reports/subnet_slice_data_audit.md
```

Current dense subnet/slice panel:

```text
179 series
69 subnets
7,215,132 rows
47,435 training windows with input_size=2016, horizon=12, stride=144
```

Fast subnet baseline metrics:

```text
traffic_forecasting/reports/metrics_subnet_slice_baselines.csv
```

Recommended real-series panel benchmark:

```bash
python3 traffic_forecasting/build_ip_slice_series.py \
  --min-total-points 2016 \
  --dense \
  --output-csv traffic_forecasting/data/ip_slice_traffic_min2016_real_full_dense.csv \
  --slice-output-csv traffic_forecasting/data/slice_traffic_min2016_real_full_from_ip.csv

python3 traffic_forecasting/forecast_ip_slice_nhits.py \
  --input-csv traffic_forecasting/data/ip_slice_traffic_min2016_real_full_dense.csv \
  --training-scope per-slice \
  --device cuda \
  --test-size 144 \
  --horizons 1,6,12 \
  --train-tail 0 \
  --input-size 2016 \
  --step-size 144 \
  --max-steps 500 \
  --predictions-csv traffic_forecasting/outputs/predictions_ip_slice_min2016_real_per_slice_window2016_stride144_gpu.csv \
  --metrics-csv traffic_forecasting/reports/metrics_ip_slice_min2016_real_per_slice_window2016_stride144_gpu.csv
```

For fair comparison, rebuild slice-level baselines on the same retained real `ip_id` coverage:

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --input-csv traffic_forecasting/data/slice_traffic_min2016_real_full_from_ip.csv \
  --models naive,seasonal_naive,moving_average,nhits,prophet,lstm \
  --evaluation-mode direct \
  --device cuda \
  --test-size 144 \
  --horizons 1,6,12 \
  --nhits-train-tail 0 \
  --prophet-train-tail 0 \
  --lstm-train-tail 0 \
  --nhits-input-size 1008 \
  --nhits-max-steps 500 \
  --lstm-window 288 \
  --lstm-hidden-size 128 \
  --lstm-epochs 50 \
  --predictions-csv traffic_forecasting/outputs/predictions_min2016_real_slice_baseline_gpu.csv \
  --metrics-csv traffic_forecasting/reports/metrics_min2016_real_slice_baseline_gpu.csv

python3 traffic_forecasting/evaluate_forecasts.py \
  --metrics-csv \
    traffic_forecasting/reports/metrics_min2016_real_slice_baseline_gpu.csv \
    traffic_forecasting/reports/metrics_ip_slice_min2016_real_per_slice_window2016_stride144_gpu.csv \
  --output-csv traffic_forecasting/reports/metrics_min2016_real_slice_vs_panel_ranked.csv
```

## Reproduction

Tune baselines and Prophet:

```bash
python3 traffic_forecasting/tune_benchmarks.py \
  --moving-average-windows 3,6,12,24,72,144,288 \
  --seasonal-periods 72,144,288,1008 \
  --prophet-train-tails 1000,2000,5000,10000
```

Tune N-HiTS:

```bash
python3 traffic_forecasting/tune_nhits.py \
  --input-sizes 36,72,144,288 \
  --train-tails 1000,2000,5000,10000 \
  --max-steps 100,200
```

Tune LSTM:

```bash
python3 traffic_forecasting/tune_lstm.py \
  --train-tails 1000,2000 \
  --windows 36,72 \
  --hidden-sizes 16,32 \
  --epochs 5,10
```

Benchmark all currently integrated models:

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --models naive,seasonal_naive,moving_average,nhits,prophet,lstm \
  --evaluation-mode direct \
  --device cuda \
  --test-size 144 \
  --horizons 1,6,12 \
  --nhits-train-tail 0 \
  --prophet-train-tail 0 \
  --lstm-train-tail 0 \
  --nhits-input-size 1008 \
  --nhits-max-steps 500 \
  --lstm-window 288 \
  --lstm-hidden-size 128 \
  --lstm-epochs 50 \
  --predictions-csv traffic_forecasting/outputs/predictions_full_gpu.csv \
  --metrics-csv traffic_forecasting/reports/metrics_full_gpu.csv
```
