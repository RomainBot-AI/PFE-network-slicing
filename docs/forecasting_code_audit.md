# Forecasting code audit

This audit records the cleanup after introducing the `src/nsf` research package.

## Canonical code

New forecasting work should use:

```text
src/nsf/
scripts/make_eda_report.py
scripts/make_eda_html.py
scripts/prepare_panel_dataset.py
scripts/run_panel_backtest.py
configs/experiment/subnet_slice_*.yaml
```

The canonical dataset is:

```text
traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
```

## Cleanup performed

Removed package-level slice duplicates:

```text
src/nsf/backtest/engine.py
scripts/build_slice_series.py
scripts/build_window_index.py
scripts/run_backtest.py
```

Removed the unused `read_slice_series` helper from `src/nsf/data/loading.py`.

Removed slice-level Makefile targets:

```text
build-slice-series
build-window-index
backtest-slice
```

## Duplication intentionally kept

`traffic_forecasting/common.py` still duplicates a few helpers that now also
exist in `src/nsf`. This is intentional for now because old scripts import it
directly and should remain runnable without package installation.

`traffic_forecasting/forecast_4_slices.py`, tuning scripts, and `ip_id` scripts
are retained as historical/legacy experiments. They are not part of the main
subnet/slice thesis pipeline.

## Rule going forward

Add new research code under `src/nsf`. Keep `traffic_forecasting/` stable unless
we are deliberately migrating one legacy script into the package with tests.
