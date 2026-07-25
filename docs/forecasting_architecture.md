# Forecasting architecture

This repository is broader than forecasting: it also contains data slicing,
simulation, Mininet/Ryu, and PPO allocation code. The forecasting architecture is
therefore added as a research layer instead of replacing the existing project.

## Migration principle

- Keep existing commands in `traffic_forecasting/` working.
- Add `src/nsf/` as the clean package for new forecasting code.
- Add `scripts/` as thin CLI wrappers; business logic should live in `src/nsf/`.
- Keep generated data, reports, and experiment runs ignored by Git.
- Move logic gradually from old scripts into `src/nsf/` only when it is tested.
- Treat slice-level and `ip_id` scripts in `traffic_forecasting/` as legacy
  material unless explicitly needed for comparison or traceability.

## Target layout

```text
configs/              declarative experiment configuration
src/nsf/              importable forecasting package
scripts/              thin command-line entry points
experiments/runs/     ignored run outputs
tests/                unit tests, especially anti-leakage tests
docs/                 methodology and memory-facing notes
traffic_forecasting/  legacy-compatible scripts and generated outputs
```

## Methodological guardrails

- No random split for forecasting.
- Same temporal protocol for all slices and panel series.
- Scaling and feature statistics must be fit on train only.
- Report metrics by horizon, not only averaged across horizons.
- Treat daily and weekly seasonal naive baselines as separate baselines.
- Prefer subnet/slice panel experiments for the next benchmark.
- Keep probabilistic allocation separate from the deterministic benchmark until
  deterministic selection is stable.

## Current Canonical Path

The validated forecasting path is subnet/slice only:

```text
traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
```

Use the package-backed commands:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.make_eda_report
PYTHONPATH=src .venv/bin/python -m scripts.prepare_panel_dataset
PYTHONPATH=src .venv/bin/python -m scripts.run_panel_backtest
```

Slice-level wrappers were removed from `scripts/` to avoid duplicating the
legacy benchmark in the clean package. The historical scripts remain available
directly under `traffic_forecasting/`.
