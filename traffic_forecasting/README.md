# Traffic Forecasting Legacy Workspace

This directory contains the first forecasting scripts used during exploration.
It is kept only for traceability and for reproducing older slice-level or
`ip_id` experiments.

The current thesis pipeline is no longer driven from this directory. Use the
canonical subnet/slice panel code under:

```text
src/nsf/
scripts/
configs/experiment/
```

## Current Canonical Protocol

The retained forecasting setup is:

```text
dataset: traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
granularity: id_institution_subnet + slice
frequency: 10 minutes
input history: 2016 steps = 14 days
horizon: 36 steps = 6 hours
folds: 5 rolling-origin folds
```

Main commands:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.make_eda_report
PYTHONPATH=src .venv/bin/python -m scripts.prepare_panel_dataset
PYTHONPATH=src .venv/bin/python -m scripts.run_deterministic_benchmark
PYTHONPATH=src .venv/bin/python -m scripts.run_probabilistic_lightgbm
PYTHONPATH=src .venv/bin/python -m scripts.export_probabilistic_forecast_for_simulation
```

## Final Model Status

Report-level conclusions are maintained in:

```text
docs/final_forecasting_conclusion.md
docs/forecasting_final_model_analysis.md
docs/probabilistic_benchmark_status.md
reports/final_forecasting_tables.md
```

Current recommendations:

```text
deterministic final model: Prophet 14d
operational deterministic baseline: LightGBM 14d
probabilistic simulation signal: LightGBM quantile 14d q90
probabilistic deep baseline: DeepAR 14d
transformer comparison: PatchTST
```

## Legacy Scripts

The scripts in this directory predate the final `src/nsf` architecture:

```text
build_slice_series.py
build_ip_slice_series.py
build_subnet_slice_series.py
forecast_4_slices.py
forecast_ip_slice_nhits.py
tune_benchmarks.py
tune_lstm.py
tune_nhits.py
evaluate_forecasts.py
```

They are not the source of truth for the final benchmark tables. Avoid using
old slice-level summaries or horizon `1,6,12` results for the thesis narrative;
those were replaced by the subnet/slice protocol above.
