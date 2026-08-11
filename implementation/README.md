# Offline network-slicing PPO simulation

A self-contained simulation that closes the full network-slicing loop offline:
**forecast per-slice demand → SDN slice on/off control → PPO reinforcement
learning → energy and QoS evaluation**. Unlike `../simulation/` (live Ryu +
Mininet in Docker), this project replays the real CESNET subnet/slice dataset in
a pure-Python Gym-style environment, so it runs end-to-end from one command with
no network emulation.

The physical/energy and QoS model and the four SDN control algorithms follow the
5G RAN power model of Phyu et al. (2023). It is an installable project of its own
(`pyproject.toml`, `uv.lock`), separate from the root forecasting package.

## Layout

```text
main.py                        CLI entry point (builds a RunConfig)
src/simulator/ran_simulator.py 5G/6G RAN physics and energy model (power in W, per-slice latency/QoS)
src/environment/sdn_controller_env.py  Gym-style SDN double-controller env (Algorithms 1–4)
src/agents/ppo_agent.py        PyTorch Actor-Critic PPO, multi-Bernoulli policy, GAE, clipped loss
src/models/                    traffic predictors behind one interface (see below)
src/pipeline/config.py         RunConfig dataclass and default paths
src/pipeline/trainer_evaluator.py  loads data, 80/20 split, trains predictor + PPO, evaluates, plots
src/pipeline/macro_ran.py      groups subnets into K macro-RANs (K-means/quantiles) for speed
src/visualization/             figure generators (per-model plots, benchmark, Pareto)
tests/                         predictor and pipeline smoke tests
data/experiments/              saved PNG results from earlier beta/lambda sweeps
```

The predictors share their windowing logic through two small bases:
`StationWindowPredictor` (per-station train/predict loop) with
`TabularLagPredictor` (ridge, lightgbm) and `SequencePredictor` (lstm, nhits) on
top; `passthrough` (oracle) and `prophet` are standalone. Model backends are
imported lazily, so a passthrough/ridge run does not require torch/lightgbm/prophet.

## Setup

This project ships a `uv.lock`; `uv` is the recommended installer.

```bash
uv venv .venv
uv pip install --python .venv -e '.[dev]'          # core + pytest (torch, lightgbm included)
uv pip install --python .venv -e '.[dev,forecasting]'  # + prophet (needed only for --model prophet)
```

Plain `pip` works too: `python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'`.
Core dependencies (numpy, pandas, scikit-learn, matplotlib, torch, lightgbm)
cover every model except prophet, which is the sole optional extra.

## Usage

Run from the `implementation/` directory:

```bash
.venv/bin/python main.py --model lightgbm --num_rans 4 --episodes 10
.venv/bin/python main.py --model all --num_rans 4        # benchmark every predictor
```

A missing optional backend (e.g. prophet) is skipped in `--model all` rather
than aborting the whole benchmark.

Key arguments:

```text
--model        passthrough | ridge | lightgbm | lstm | nhits | prophet | all
--subnet       all | topN | <id>            (ignored when --num_rans > 0)
--num_rans     group the 69 subnets into K macro-RANs (0 = per-subnet)
--steps        max steps per episode (0 = full dataset)
--episodes     PPO training episodes
--beta         weight of QoS satisfaction in the reward
--lambda_loss  penalty for QoS violation / overload
--dataset      dataset CSV path (defaults to the repo dataset, see below)
--output_dir   figure output directory (defaults to implementation/data/plots)
```

## Data

Defaults resolve automatically to the repository dataset:

```text
../traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
```

Columns: `ds`, `id_institution_subnet`, `slice`, `y`. Override with `--dataset`.

## Method

- **Reward**: `r_t = 1/f_b + beta * eta - lambda_loss * L`, trading off energy
  efficiency (`1/f_b`, power in Watts), QoS satisfaction (`eta`), and traffic
  loss (`L`).
- **Control (Algorithms 1–4)**: PPO decides on/off per specialised slice
  (EcoSlice 1 always on) → threshold filter turns off slices below
  `alpha * Max_i` (causal rolling 95th percentile, no future leakage) → priority
  routing (URLLC > URLLC_eMBB_MIX > mMTC > eMBB) into EcoSlice 1, spilling to
  EcoSlice 2 above 75% → safety fallback re-activates the heaviest slice on
  overload.
- **Predictors** share the `BaseTrafficPredictor` interface and are scored by
  MAE / RMSE / NMAE; `passthrough` is a perfect-foresight oracle baseline.
- **Split**: chronological 80/20 train/test, no shuffling.

## Outputs

Figures are written under `--output_dir` (default `data/plots/`), with a flat
copy under `data/plots/_artifacts/`. Per-run figures include traffic prediction,
active-slice timeline, bandwidth allocation, QoS/SLA analysis, energy
consumption, PPO train-vs-test, and an all-models benchmark chart.
`data/experiments/` holds saved results from earlier `beta`/`lambda_loss` sweeps.

Console logs and figure labels are in French; code and documentation are in English.

## Tests

```bash
.venv/bin/python -m pytest
```

The suite generates a small synthetic panel and checks each predictor plus a
full pipeline run, so it needs no external dataset.
