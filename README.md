# PFE — Network Slicing: Traffic Forecasting & SDN Allocation

A network-slicing study on real CESNET traffic. It predicts per-slice demand
with time-series models, then uses that forecast to let a reinforcement-learning
agent allocate bandwidth **proactively** across four slices — `URLLC`,
`URLLC_eMBB_MIX`, `eMBB`, `mMTC` — in an emulated SDN.

The core idea: instead of reacting to congestion after it happens, forecast the
demand of the next hour and reserve capacity before the spike arrives.

## The pipeline

```text
  raw CESNET traffic
         │
   ┌─────▼───────────────┐
   │ 1. Prepare datasets  │  Dataset Preparing/  →  clustered slice dataset
   └─────┬───────────────┘                         + benchmark panel
         │
   ┌─────▼───────────────┐
   │ 2. Forecast demand   │  src/nsf + scripts/  →  LightGBM q90 demand signal
   └─────┬───────────────┘                         (CSV for the simulation)
         │
   ┌─────▼───────────────┐        ┌──────────────────────────────┐
   │ 3. Simulate (SDN)    │  ───▶  │ 3a. reactive (no forecast)    │
   │    Ryu + Mininet +   │        │ 3b. proactive (with forecast) │
   │    PPO allocation    │        └──────────────────────────────┘
   └──────────────────────┘
```

Stage 3 has a second, self-contained variant under `implementation/` that runs
the whole forecast→control→energy/QoS loop offline (no Docker) — see
[Alternative: offline simulation](#alternative-offline-simulation).

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

Run commands from the repository root; the `Makefile` exports `PYTHONPATH=src`
automatically. Component-specific dependencies live with their code
(`Dataset Preparing/requirements.txt`, `simulation/requirements-ppo.txt`).

Large datasets are not versioned in Git — see [Data](#data) for where each file
goes.

---

## 1. Prepare the datasets

Two datasets drive the project, both built from raw CESNET aggregates in
`data/raw/`.

**Slice dataset for the simulation** — cluster the per-IP traffic into the four
slices:

```bash
python3 -m pip install -r "Dataset Preparing/requirements.txt"
python3 "Dataset Preparing/cluster_4_slices.py"
```

This writes `simulation/mininet/cesnet_points_clustered_4slices.csv` and the
clustering artifacts under `models/`. Details in `Dataset Preparing/README.md`.

**Forecasting panel** — build the dense subnet/slice panel from the clustered
dataset and the `id → subnet` mapping (`data/reference/ids_relationship.csv`):

```bash
make build-subnet-panel PYTHON=.venv/bin/python
```

This writes `traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv`
(columns `unique_id`, `ds`, `y`, `slice`) — the panel the benchmarks read. See
[Data](#data).

## 2. Forecast per-slice demand

Train and compare the forecasting models, then export the retained signal.

```bash
# Prepare panel + run benchmarks (see the Makefile for all targets)
make preprocess-panel PYTHON=.venv/bin/python
make benchmark-deterministic PYTHON=.venv/bin/python
make benchmark-probabilistic-lightgbm PYTHON=.venv/bin/python

# Export the retained LightGBM q90 forecast as a slice-level demand signal
PYTHONPATH=src .venv/bin/python -m scripts.export_probabilistic_forecast_for_simulation
```

The export writes
`simulation/forecast_inputs/slice_demand_forecast_lightgbm_q90.csv`. The `q90`
(90th-percentile) quantile is a deliberately conservative demand estimate, so
the controller slightly over-provisions and protects the SLA against an incoming
spike. See [Forecasting reference](#forecasting-reference) for the protocol,
models, and results.

This stage is only required for the **proactive** run (3b). You can skip it and
go straight to the reactive baseline (3a).

## 3. Run the SDN simulation

The simulation uses Ryu (SDN controller + REST API), Mininet (topology and
traffic), and a PPO agent that decides per-slice activation and bandwidth. Full
details and health-check commands are in `simulation/README.md`.

Start the controller and topology (two terminals, from `simulation/`):

```bash
# terminal 1: Ryu + Mininet container
docker compose up --build

# terminal 2: open a shell in the container, then start the topology inside it
docker exec -it mininet bash
python3 topology.py
```

### 3a. Reactive baseline (no forecast)

The PPO agent allocates using only the **observed** load — the reactive
reference to compare against:

```bash
RYU_CONTROLLER_IP=127.0.0.1 RYU_REST_PORT=8080 python3 simulation/ppo.py
```

### 3b. Proactive (with forecast)

Feed the exported `q90` signal so the agent plans for demand **before** it
arrives. `FORECAST_MODE=max` uses `max(observed, forecast)` per slice — the
recommended conservative mode:

```bash
FORECAST_CSV=simulation/forecast_inputs/slice_demand_forecast_lightgbm_q90.csv \
FORECAST_QUANTILE=q90 FORECAST_MODE=max \
RYU_CONTROLLER_IP=127.0.0.1 RYU_REST_PORT=8080 python3 simulation/ppo.py
```

Comparing 3a and 3b shows the benefit of proactive allocation: fewer SLA
violations on critical slices when traffic ramps up, at the cost of slightly
higher provisioning.

## Alternative: offline simulation

`implementation/` is a self-contained project that closes the same loop
(forecast → slice control → PPO → energy/QoS) **offline**, replaying the dataset
in a pure-Python environment — no Docker, one command:

```bash
cd implementation
uv venv .venv && uv pip install --python .venv -e '.[dev]'
.venv/bin/python main.py --model lightgbm --num_rans 4 --episodes 10
```

It also benchmarks every predictor at once (`--model all`). See
`implementation/README.md`.

---

## Repository layout

```text
Dataset Preparing/   builds the 4-slice clustered dataset from raw CESNET aggregates
src/nsf/             forecasting package (data, backtest, models, evaluation, export)
scripts/             command-line entry points (thin wrappers over src/nsf)
configs/             declarative experiment configuration
simulation/          Ryu SDN controller, Mininet topology, and PPO allocation agent
implementation/      self-contained offline forecast-in-the-loop PPO simulation
models/              slice-construction artifacts (kmeans, scaler, cluster→slice map)
tests/               unit tests, including anti-leakage tests
data/                raw / interim / processed data (contents ignored by Git)
data/reference/      ids_relationship.csv: CESNET id → institution/subnet mapping
experiments/runs/    generated benchmark outputs (ignored by Git)
```

## Data

Large datasets are not versioned in Git:

```text
data/raw/                                                    raw CESNET per-IP aggregates (input)
simulation/mininet/cesnet_points_clustered_4slices.csv       stage-1 slice dataset (generated)
traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv   forecasting panel (generated by `make build-subnet-panel`)
```

## Forecasting reference

Protocol (one temporal setup shared by every model — no random split):

```text
dataset:     subnet/slice panel (id_institution_subnet + slice)
series:      179          slices: 4
frequency:   10 minutes
input_size:  2016 steps = 14 days (reference); 1d / 7d kept as sensitivity
horizon:     36 steps = 6 hours
folds:       5 rolling-origin folds, stride 144 steps (1 day)
```

Scaling and features are fit on train only; metrics are reported per horizon and
per slice with a leakage audit per run. Outputs land under `experiments/runs/`.

Models: baselines (persistence, seasonal naive, moving average), Prophet,
LightGBM (point and quantile), LSTM, N-HiTS, PatchTST, DeepAR.

Results on the 14-day reference protocol:

| model | RMSE | WAPE | MASE | note |
| --- | ---: | ---: | ---: | --- |
| Prophet tuned | 27.31M | 0.904 | 0.545 | best global RMSE and MASE |
| PatchTST tuned | 28.20M | 1.018 | 0.574 | best transformer-family result |
| LSTM 5000w | 29.02M | 0.992 | 0.563 | best recurrent deep baseline |
| LightGBM tuned | 29.31M | 0.876 | 0.600 | best global WAPE |
| N-HiTS tuned | 33.87M | 1.811 | 0.650 | not a winner on this panel |

For the proactive simulation, the retained demand signal is **LightGBM quantile
14d (`q90`)** — best probabilistic interval score with compact, well-calibrated
intervals.

## Tests

```bash
make test PYTHON=.venv/bin/python
```
