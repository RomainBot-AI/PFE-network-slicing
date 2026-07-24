# Traffic forecasting

Status: this folder is now a legacy-compatible workspace. The current canonical
research pipeline for the thesis is the subnet/slice panel pipeline under
`src/nsf`, `scripts`, and `configs/experiment`.

Use these commands for new work:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.make_eda_report
PYTHONPATH=src .venv/bin/python -m scripts.prepare_panel_dataset
PYTHONPATH=src .venv/bin/python -m scripts.run_panel_backtest
```

The older slice-level and `ip_id` experiments below are kept for traceability and
to avoid breaking previous commands. They should not be used as the primary
benchmark now that subnet/slice has been selected.

This folder contains the reproducible forecasting pipeline for the sliced CESNET traffic dataset.

The current pipeline has two layers:

1. build a compact time series per slice;
2. run baseline forecasts;
3. optionally run N-HiTS;
4. export predictions and metrics.

This first layer is a slice-level benchmark. It is useful as a baseline, but it only contains 4 time series. The next layer uses `ip_id` to build a multi-series panel and learns from many `slice/ip_id` trajectories before aggregating forecasts back to the slice level.

PatchTST can still be added later as the Transformer-family comparison.

## Input

The expected input is the generated sliced dataset:

```text
simulation/mininet/cesnet_points_clustered_4slices.csv
```

Required columns:

- `timestamp`
- `slice`
- `ip_id` for the panel pipeline
- `n_bytes`

## Build slice series

From the repository root:

```bash
python3 traffic_forecasting/build_slice_series.py
```

Default output:

```text
traffic_forecasting/data/slice_traffic_10min.csv
```

The output is a compact wide CSV:

```text
timestamp,URLLC,URLLC_eMBB_MIX,eMBB,mMTC
```

## Build `ip_id` Panel Series

The slice-only dataset has one global series per slice. To train models on many traffic trajectories, build a long panel with one series per observed `slice/ip_id` pair:

```bash
python3 traffic_forecasting/build_ip_slice_series.py
```

Default long output:

```text
traffic_forecasting/data/ip_slice_traffic_10min_long.csv
```

Schema:

```text
unique_id,ds,y,slice,ip_id
```

For debug or first GPU runs, use a dense top-K panel. This keeps the largest real `ip_id` series per slice and fills missing 10-minute points with zero:

```bash
python3 traffic_forecasting/build_ip_slice_series.py \
  --top-k-per-slice 50 \
  --dense \
  --output-csv traffic_forecasting/data/ip_slice_traffic_top50_dense.csv \
  --slice-output-csv traffic_forecasting/data/slice_traffic_top50_from_ip.csv
```

Use `--top-k-per-slice 100` or higher after the first benchmark if GPU time and disk space are acceptable. Avoid `--dense` without top-K: the full dense grid is roughly `40308 * 3938` rows.

To learn from two-week windows across the whole year instead of only the last two weeks, keep every real `slice/ip_id` series with at least 2016 observed points over the full period:

```bash
python3 traffic_forecasting/build_ip_slice_series.py \
  --min-total-points 2016 \
  --dense \
  --output-csv traffic_forecasting/data/ip_slice_traffic_min2016_real_full_dense.csv \
  --slice-output-csv traffic_forecasting/data/slice_traffic_min2016_real_full_from_ip.csv
```

Current output:

```text
659 series
40308 points per series
26562972 rows
```

This is larger on disk, but it lets window-based models train on many overlapping two-week examples from the whole dataset.

For a quick debug run:

```bash
python3 traffic_forecasting/build_slice_series.py --max-rows 100000
```

## Run baseline forecasts

```bash
python3 traffic_forecasting/forecast_4_slices.py
```

Default models:

- `naive`
- `seasonal_naive`
- `moving_average`

Default horizons:

- `1` step = 10 minutes
- `6` steps = 1 hour
- `12` steps = 2 hours

Outputs:

```text
traffic_forecasting/outputs/predictions.csv
traffic_forecasting/reports/metrics.csv
```

By default, models are evaluated with `--evaluation-mode direct`: one common train/test split, one forecast origin, and the same requested horizons for every model. This is the mode to use for benchmark comparison.

For baseline-only robustness analysis, rolling evaluation is also available:

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --models naive,seasonal_naive,moving_average \
  --evaluation-mode rolling
```

## Run N-HiTS

N-HiTS is the first advanced model ported from the notebooks. It uses `neuralforecast` and trains once on the train split, then predicts the maximum requested horizon.

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --models nhits \
  --horizons 1,6,12
```

For a quick smoke test:

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --models nhits \
  --test-size 48 \
  --horizons 1,6 \
  --nhits-max-steps 2 \
  --nhits-train-tail 500
```

N-HiTS supports the common `evaluation=direct` benchmark mode.

## Benchmark models

To compare baselines and N-HiTS with the same protocol:

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --models naive,seasonal_naive,moving_average,nhits,prophet,lstm \
  --evaluation-mode direct \
  --horizons 1,6,12
```

Recommended first benchmark command:

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --models naive,seasonal_naive,moving_average,nhits,prophet,lstm \
  --evaluation-mode direct \
  --test-size 144 \
  --horizons 1,6,12 \
  --nhits-input-size 72 \
  --nhits-max-steps 200 \
  --nhits-train-tail 5000 \
  --prophet-train-tail 5000 \
  --lstm-train-tail 2000 \
  --lstm-window 36 \
  --lstm-hidden-size 16 \
  --lstm-epochs 5 \
  --predictions-csv traffic_forecasting/outputs/predictions_benchmark.csv \
  --metrics-csv traffic_forecasting/reports/metrics_benchmark.csv
```

Use GPU when available:

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --models nhits,lstm \
  --device auto
```

Force CUDA and fail if it is not available through PyTorch/Lightning:

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --models nhits,lstm \
  --device cuda
```

Fair-history benchmark with the same train tail for trained models:

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --models naive,seasonal_naive,moving_average,nhits,prophet,lstm \
  --evaluation-mode direct \
  --device auto \
  --test-size 144 \
  --horizons 1,6,12 \
  --nhits-train-tail 10000 \
  --prophet-train-tail 10000 \
  --lstm-train-tail 10000 \
  --nhits-input-size 288 \
  --nhits-max-steps 300 \
  --lstm-window 144 \
  --lstm-hidden-size 64 \
  --lstm-epochs 20
```

Use `--nhits-train-tail 0 --prophet-train-tail 0 --lstm-train-tail 0` to train on all available history.

## Run Panel N-HiTS By `ip_id`

N-HiTS can also be trained on the `slice/ip_id` panel, then predictions are aggregated back to the four slices for the same metric format as the slice-only benchmark.

Smoke test on a small dense panel:

```bash
python3 traffic_forecasting/build_ip_slice_series.py \
  --max-rows 100000 \
  --top-k-per-slice 2 \
  --dense \
  --output-csv traffic_forecasting/data/ip_slice_traffic_sample_top2_dense.csv \
  --slice-output-csv traffic_forecasting/data/slice_traffic_sample_top2_from_ip.csv

python3 traffic_forecasting/forecast_ip_slice_nhits.py \
  --input-csv traffic_forecasting/data/ip_slice_traffic_sample_top2_dense.csv \
  --test-size 12 \
  --horizons 1,6 \
  --input-size 12 \
  --max-steps 2 \
  --train-tail 200 \
  --device cpu
```

Two-week windowed GPU run across the full-year real-series panel:

```bash
python3 traffic_forecasting/forecast_ip_slice_nhits.py \
  --input-csv traffic_forecasting/data/ip_slice_traffic_min2016_real_full_dense.csv \
  --device cuda \
  --test-size 144 \
  --horizons 1,6,12 \
  --input-size 2016 \
  --step-size 144 \
  --train-tail 0 \
  --max-steps 500 \
  --predictions-csv traffic_forecasting/outputs/predictions_ip_slice_min2016_real_window2016_stride144_gpu.csv \
  --metrics-csv traffic_forecasting/reports/metrics_ip_slice_min2016_real_window2016_stride144_gpu.csv
```

Preferred variant after clustering: train one independent panel model per slice.

```bash
python3 traffic_forecasting/forecast_ip_slice_nhits.py \
  --input-csv traffic_forecasting/data/ip_slice_traffic_min2016_real_full_dense.csv \
  --training-scope per-slice \
  --device cuda \
  --test-size 144 \
  --horizons 1,6,12 \
  --input-size 2016 \
  --step-size 144 \
  --train-tail 0 \
  --max-steps 500 \
  --predictions-csv traffic_forecasting/outputs/predictions_ip_slice_min2016_real_per_slice_window2016_stride144_gpu.csv \
  --metrics-csv traffic_forecasting/reports/metrics_ip_slice_min2016_real_per_slice_window2016_stride144_gpu.csv
```

With `test-size=144`, `input-size=2016`, `horizon=12`, and `step-size=144`, the current full-year panel gives:

```text
659 series
40164 train timestamps
174635 training windows
test starts at 2024-07-13 22:00
```

The split is strict: N-HiTS receives only rows before the test start, so no training window target can overlap the final test horizon.

These panel metrics are computed on the retained real `ip_id` series only. Compare them against baselines rebuilt from the matching `slice_traffic_min2016_real_full_from_ip.csv`, not against the full-slice benchmark.

## Build Subnet/Slice Series

CESNET provides `ids_relationship.csv`, which maps:

```text
id_ip -> id_institution -> id_institution_subnet
```

The project does not have real antenna identifiers. `id_institution_subnet` is the closest defensible network-level proxy, so the preferred panel granularity is:

```text
id_institution_subnet + slice + timestamp
```

Build the observed subnet/slice panel:

```bash
python3 traffic_forecasting/build_subnet_slice_series.py \
  --output-csv traffic_forecasting/data/subnet_slice_traffic_10min_long.csv \
  --slice-output-csv traffic_forecasting/data/slice_traffic_from_subnet_10min.csv
```

Build the dense two-week-capable panel for windowed models:

```bash
python3 traffic_forecasting/build_subnet_slice_series.py \
  --min-total-points 2016 \
  --dense \
  --output-csv traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv \
  --slice-output-csv traffic_forecasting/data/slice_traffic_from_subnet_min2016.csv
```

Current dense output:

```text
179 series
69 subnets
45 institutions
7,215,132 rows
```

Audit report:

```text
traffic_forecasting/reports/subnet_slice_data_audit.md
```

Build the preferred explicit two-week plus six-hour window index.
This protocol alternates train/validation/test blocks and rotates the assignment by institutional subnet, so validation and test cover the whole year globally while each subnet/slice series keeps disjoint train/validation/test timestamps:

```bash
python3 traffic_forecasting/build_window_index.py \
  --panel-csv traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv \
  --input-size 2016 \
  --horizon 36 \
  --stride 144 \
  --split-mode rotating_blocks \
  --split-block-size 4032 \
  --block-pattern train,train,val,train,train,test \
  --rotate-by id_institution_subnet \
  --output-csv traffic_forecasting/reports/subnet_slice_window_index_2016_36_stride144_rotating_subnet.csv
```

Window protocol report:

```text
traffic_forecasting/reports/subnet_slice_window_protocol_rotating.md
```

Current rotating split:

```text
train: 16,758 examples
validation: 4,172 examples
test: 4,130 examples
179 subnet/slice series in each split
target periods cover 2023-10-23 -> 2024-07-14 05:50 globally
series_with_split_overlap: 0
```

Run fast slice-level baselines on the same retained subnet coverage:

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --input-csv traffic_forecasting/data/slice_traffic_from_subnet_min2016.csv \
  --models naive,seasonal_naive,moving_average \
  --evaluation-mode direct \
  --test-size 144 \
  --horizons 1,6,12 \
  --predictions-csv traffic_forecasting/outputs/predictions_subnet_slice_baselines.csv \
  --metrics-csv traffic_forecasting/reports/metrics_subnet_slice_baselines.csv
```

Run N-HiTS on the subnet/slice panel:

```bash
python3 traffic_forecasting/forecast_ip_slice_nhits.py \
  --input-csv traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv \
  --model-name nhits_subnet_slice \
  --training-scope per-slice \
  --device cuda \
  --test-size 144 \
  --horizons 1,6,12 \
  --input-size 2016 \
  --step-size 144 \
  --train-tail 0 \
  --max-steps 500 \
  --predictions-csv traffic_forecasting/outputs/predictions_subnet_slice_nhits_window2016_stride144_gpu.csv \
  --metrics-csv traffic_forecasting/reports/metrics_subnet_slice_nhits_window2016_stride144_gpu.csv
```

Compare subnet panel N-HiTS against subnet slice baselines:

```bash
python3 traffic_forecasting/evaluate_forecasts.py \
  --metrics-csv \
    traffic_forecasting/reports/metrics_subnet_slice_baselines.csv \
    traffic_forecasting/reports/metrics_subnet_slice_nhits_window2016_stride144_gpu.csv \
  --output-csv traffic_forecasting/reports/metrics_subnet_slice_ranked.csv
```

Historical `ip_id` panel commands below are kept for reproducibility, but subnet/slice is now the preferred panel granularity.

Run slice-level baselines on the same retained real `ip_id` coverage:

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
```

Compare the panel model against that matching slice-level benchmark:

```bash
python3 traffic_forecasting/evaluate_forecasts.py \
  --metrics-csv \
    traffic_forecasting/reports/metrics_min2016_real_slice_baseline_gpu.csv \
    traffic_forecasting/reports/metrics_ip_slice_min2016_real_per_slice_window2016_stride144_gpu.csv \
  --output-csv traffic_forecasting/reports/metrics_min2016_real_slice_vs_panel_ranked.csv
```

## Run Prophet

Prophet is trained independently for each slice on log-transformed traffic. It is useful as an interpretable statistical baseline with trend and seasonality.

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --models prophet \
  --evaluation-mode direct \
  --horizons 1,6,12
```

## Run LSTM

LSTM is implemented with PyTorch and trained independently for each slice on normalized `log1p(n_bytes)`.

```bash
python3 traffic_forecasting/forecast_4_slices.py \
  --models lstm \
  --evaluation-mode direct \
  --horizons 1,6,12
```

Current recommended LSTM settings from a short tuning run:

```text
--lstm-train-tail 2000 --lstm-window 36 --lstm-hidden-size 16 --lstm-epochs 5
```

Tune LSTM:

```bash
python3 traffic_forecasting/tune_lstm.py \
  --train-tails 1000,2000 \
  --windows 36,72 \
  --hidden-sizes 16,32 \
  --epochs 5,10
```

## Tune N-HiTS

Use the tuning helper to compare a small grid of N-HiTS configurations:

```bash
python3 traffic_forecasting/tune_nhits.py \
  --input-sizes 36,72,144,288 \
  --train-tails 1000,2000,5000,10000 \
  --max-steps 100,200
```

On the current direct benchmark, the best global configuration found so far is:

```text
--nhits-input-size 72 --nhits-train-tail 5000 --nhits-max-steps 200
```

N-HiTS improves several slice/horizon pairs, but it does not dominate every baseline on the current single-origin benchmark. This is useful: the final model selection should be based on measured performance, not model complexity.

## Tune Baselines And Prophet

Tune moving-average windows, seasonal periods, and Prophet train tails:

```bash
python3 traffic_forecasting/tune_benchmarks.py \
  --moving-average-windows 3,6,12,24,72,144,288 \
  --seasonal-periods 72,144,288,1008 \
  --prophet-train-tails 1000,2000,5000,10000
```

Default output:

```text
traffic_forecasting/reports/tuning_baselines_prophet.csv
```

Current benchmark result after tuning and adding LSTM:

- `lstm`: best on 4 slice/horizon pairs.
- `moving_average`: best on 3 slice/horizon pairs.
- `prophet`: best on 3 slice/horizon pairs.
- `seasonal_naive`: best on 1 slice/horizon pair.
- `naive`: best on 1 slice/horizon pair.

This confirms that the most suitable solution is likely a model-selection layer by slice and horizon, not a single global model.

## Rank metrics

```bash
python3 traffic_forecasting/evaluate_forecasts.py
```

Default output:

```text
traffic_forecasting/reports/metrics_ranked.csv
```

## Notes

Generated data, outputs, reports and model artifacts under `traffic_forecasting/` are ignored by Git.

The first simulation integration should use the simplest model that is accurate and stable enough. PatchTST is planned as the Transformer-family state-of-the-art comparison, but it should remain optional until it clearly improves the results.
