# PFE Network Slicing Forecasting

Code for deterministic and probabilistic traffic forecasting experiments on network slicing time series.

The repository contains the reproducible pipeline code, experiment configurations, and the small preprocessing models used for slice construction. Large datasets and generated experiment runs are intentionally kept outside Git.

## Setup

Create a virtual environment and install the forecasting dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements-forecasting.txt
```

Run commands from the repository root. The `Makefile` sets `PYTHONPATH=src` automatically.

## Data

Large traffic datasets are not versioned in Git. See `DATA.md` for the expected local files and paths.

The main benchmark configs expect:

```text
traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
```

## Deterministic Benchmarks

Run the main deterministic benchmark:

```bash
make benchmark-deterministic PYTHON=.venv/bin/python
```

Run model-specific benchmarks:

```bash
make benchmark-lstm PYTHON=.venv/bin/python
make benchmark-prophet PYTHON=.venv/bin/python
make benchmark-patchtst PYTHON=.venv/bin/python
```

The deterministic benchmark compares baselines, statistical models, machine learning, and deep learning models using the same temporal folds, horizons, and metrics.

## Probabilistic Benchmarks

Run LightGBM quantile forecasting:

```bash
make benchmark-probabilistic-lightgbm PYTHON=.venv/bin/python
```

Run DeepAR distributional forecasting:

```bash
make benchmark-probabilistic-deepar PYTHON=.venv/bin/python
```

The default probabilistic configs use a 14-day history and a 6-hour horizon. One-day history configs are also available:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_probabilistic_lightgbm --config configs/experiment/probabilistic_lightgbm_1d.yaml
PYTHONPATH=src .venv/bin/python -m scripts.run_probabilistic_deepar --config configs/experiment/probabilistic_deepar_1d.yaml
```

Smoke tests for quick validation:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.run_probabilistic_lightgbm --config configs/experiment/probabilistic_lightgbm_smoke.yaml
PYTHONPATH=src .venv/bin/python -m scripts.run_probabilistic_deepar --config configs/experiment/probabilistic_deepar_smoke.yaml
```

## Outputs

Experiment runs are written under `experiments/runs/` and are ignored by Git. Typical outputs include:

- `predictions_probabilistic.csv`
- `benchmark_summary.csv`
- `benchmark_summary_by_slice.csv`
- `benchmark_probabilistic_summary.csv`
- `benchmark_probabilistic_summary_by_slice.csv`
- `timing.csv`
- `model_metadata.csv`
- `leakage_audit.csv`

## Models Included

The `models/` directory contains the small preprocessing artifacts used for 4-slice construction:

- `kmeans_4clusters.pkl`
- `scaler_4clusters.pkl`
- `cluster_to_slice.pkl`

