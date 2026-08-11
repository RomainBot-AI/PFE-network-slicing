# PFE — Network Slicing: Traffic Forecasting & SDN Allocation

This repository holds two connected parts of a network-slicing study on real
CESNET traffic:

1. **Traffic forecasting** — predict per-slice demand on a 10-minute
   subnet/slice traffic panel, comparing baselines, statistical, machine
   learning, and deep learning models under one temporal protocol.
2. **SDN slicing simulation** — an emulated Ryu + Mininet network whose
   per-slice bandwidth is allocated by a PPO agent, which can consume the
   forecast as a proactive demand signal.

Four slices are used throughout: `URLLC`, `URLLC_eMBB_MIX`, `eMBB`, `mMTC`.

## Repository layout

```text
src/nsf/            forecasting package (data, backtest, models, evaluation, export)
scripts/            command-line entry points (thin wrappers over src/nsf)
configs/            declarative experiment configuration (data, models, experiments, backtest)
tests/              unit tests, including anti-leakage tests
Dataset Preparing/  builds the 4-slice clustered dataset from raw CESNET aggregates
simulation/         Ryu SDN controller, Mininet topology, and PPO allocation agent
models/             preprocessing artifacts for slice construction (kmeans, scaler, mapping)
data/               raw / interim / processed data (contents ignored by Git)
traffic_forecasting/data/   local home of the benchmark panel CSV (ignored by Git)
experiments/runs/   generated benchmark outputs (ignored by Git)
implementation/     separate parallel RAN-simulation codebase (self-contained; under review)
ForecastingDoc/     ids_relationship.csv, mapping CESNET ids to institutions/subnets
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` covers the forecasting pipeline. Additional requirement sets:
`requirements-data.txt` (slice clustering), `simulation/requirements-ppo.txt`
(PPO agent). Run all commands from the repository root; the `Makefile` exports
`PYTHONPATH=src` automatically.

## Data

Large datasets are not versioned in Git. The forecasting benchmarks expect the
dense 10-minute subnet/slice panel at:

```text
traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
```

with columns `unique_id`, `ds`, `y`, `slice`. Raw CESNET per-IP aggregates live
under `data/raw/` and the clustered simulation dataset at
`simulation/mininet/cesnet_points_clustered_4slices.csv`.

## Slice dataset construction

`Dataset Preparing/` turns raw CESNET per-IP aggregates into the 4-slice
dataset via feature engineering + MiniBatchKMeans, producing the clustered
simulation CSV and the model artifacts under `models/`. See
`Dataset Preparing/README.md`.

## Forecasting

Config-driven benchmarks run through `scripts/` and `src/nsf/`. Common commands
(see the `Makefile` for the full list):

```bash
make preprocess-panel PYTHON=.venv/bin/python
make benchmark-deterministic PYTHON=.venv/bin/python
make benchmark-probabilistic-lightgbm PYTHON=.venv/bin/python
make benchmark-probabilistic-deepar PYTHON=.venv/bin/python
make model-comparison PYTHON=.venv/bin/python
```

Or directly, e.g.:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_deterministic_benchmark
PYTHONPATH=src .venv/bin/python -m scripts.run_probabilistic_lightgbm --config configs/experiment/probabilistic_lightgbm_1d.yaml
```

### Protocol

```text
dataset:     subnet/slice panel (id_institution_subnet + slice)
series:      179          slices: 4
frequency:   10 minutes
input_size:  2016 steps = 14 days (reference); 1d / 7d kept as sensitivity
horizon:     36 steps = 6 hours
folds:       5 rolling-origin folds, stride 144 steps (1 day)
scope:       per-slice training for learned panel models
```

No random split. Scaling and feature statistics are fit on train only; the same
temporal folds are shared by every model; metrics are reported per horizon and
per slice, with a leakage audit written alongside each run. Outputs land under
`experiments/runs/` (`predictions*.csv`, `benchmark_summary*.csv`, `timing.csv`,
`model_metadata.csv`, `leakage_audit.csv`).

### Models

- Baselines: persistence, seasonal naive (daily/weekly), moving average
- Statistical: Prophet
- Machine learning: LightGBM (point and quantile)
- Deep learning: LSTM, N-HiTS, PatchTST, DeepAR

### Results (14-day reference protocol)

Deterministic global ranking (by RMSE; WAPE and MASE also reported):

| model | RMSE | WAPE | MASE | note |
| --- | ---: | ---: | ---: | --- |
| Prophet tuned | 27.31M | 0.904 | 0.545 | best global RMSE and MASE |
| PatchTST tuned | 28.20M | 1.018 | 0.574 | best transformer-family result |
| LSTM 5000w | 29.02M | 0.992 | 0.563 | best recurrent deep baseline |
| LightGBM tuned | 29.31M | 0.876 | 0.600 | best global WAPE |
| N-HiTS tuned | 33.87M | 1.811 | 0.650 | not a winner on this panel |

Probabilistic (selection by interval score, not median accuracy):

| model | interval score | width | coverage | median RMSE |
| --- | ---: | ---: | ---: | ---: |
| LightGBM quantile 14d | 33.37M | 6.36M | 0.835 | — |
| DeepAR 14d | wider | wider | 0.855 | 26.11M |

Input-history sensitivity: Prophet and LightGBM are best with 14-day history;
LSTM, N-HiTS, and PatchTST are best globally with 1-day history under their
tuned settings.

### Retained models

```text
Deterministic final model:       Prophet 14d
Operational deterministic model:  LightGBM 14d (best WAPE, cheap to deploy)
Deep-learning baseline:           LSTM 5000w
Transformer comparison:           PatchTST 14d
Probabilistic simulation signal:  LightGBM quantile 14d (q90)
Probabilistic deep baseline:      DeepAR 14d
```

## Forecast-driven simulation

Export the retained probabilistic forecast to a slice-level simulation input:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.export_probabilistic_forecast_for_simulation
```

This writes `simulation/forecast_inputs/slice_demand_forecast_lightgbm_q90.csv`,
consumed by the PPO agent as a conservative `q90` demand signal. The SDN
controller, Mininet topology, and PPO agent, and how to run them, are documented
in `simulation/README.md`.

## Tests

```bash
make test PYTHON=.venv/bin/python
```
