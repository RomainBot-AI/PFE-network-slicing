.PHONY: build-subnet-panel eda-subnet-panel eda-html preprocess-panel backtest-panel tune-lightgbm tune-lstm tune-prophet tune-patchtst benchmark-lstm benchmark-prophet benchmark-patchtst benchmark-deterministic benchmark-report model-comparison run-history-sensitivity history-sensitivity history-tables probabilistic-selection rank-metrics test

PYTHON ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

build-subnet-panel:
	$(PYTHON) traffic_forecasting/build_subnet_slice_series.py

eda-subnet-panel:
	$(PYTHON) -m scripts.make_eda_report

eda-html:
	$(PYTHON) -m scripts.make_eda_html

preprocess-panel:
	$(PYTHON) -m scripts.prepare_panel_dataset

backtest-panel:
	$(PYTHON) -m scripts.run_panel_backtest

tune-lightgbm:
	$(PYTHON) -m scripts.tune_lightgbm

tune-lstm:
	$(PYTHON) -m scripts.tune_lstm

tune-prophet:
	$(PYTHON) -m scripts.tune_prophet

tune-patchtst:
	$(PYTHON) -m scripts.tune_patchtst

benchmark-deterministic:
	$(PYTHON) -m scripts.run_deterministic_benchmark

benchmark-lstm:
	$(PYTHON) -m scripts.run_lstm_benchmark

benchmark-prophet:
	$(PYTHON) -m scripts.run_prophet_benchmark

benchmark-patchtst:
	$(PYTHON) -m scripts.run_patchtst_benchmark

benchmark-report:
	$(PYTHON) -m scripts.make_benchmark_report

model-comparison:
	$(PYTHON) -m scripts.make_model_comparison

run-history-sensitivity:
	$(PYTHON) -m scripts.run_history_sensitivity

history-sensitivity:
	$(PYTHON) -m scripts.make_history_sensitivity

history-tables:
	$(PYTHON) -m scripts.make_history_tables

probabilistic-selection:
	$(PYTHON) -m scripts.make_probabilistic_selection

rank-metrics:
	$(PYTHON) -m scripts.evaluate

test:
	$(PYTHON) -m pytest
