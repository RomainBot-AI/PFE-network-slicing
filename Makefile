.PHONY: build-subnet-panel eda-subnet-panel eda-html preprocess-panel preprocessing-report backtest-panel tune-lightgbm tune-lstm tune-prophet tune-patchtst tune-nhits benchmark-lstm benchmark-prophet benchmark-patchtst benchmark-nhits benchmark-deterministic benchmark-probabilistic-lightgbm benchmark-probabilistic-deepar benchmark-probabilistic-nhits benchmark-probabilistic-patchtst benchmark-probabilistic-prophet benchmark-report model-comparison run-history-sensitivity history-sensitivity history-tables probabilistic-selection export-forecast final-artifacts test

PYTHON ?= python3
export PYTHONPATH := forecasting/src:forecasting:$(PYTHONPATH)

build-subnet-panel:
	$(PYTHON) -m scripts.build_subnet_panel --dense --output-csv forecasting/data/subnet_slice_traffic_min2016_dense.csv

eda-subnet-panel:
	$(PYTHON) -m scripts.make_eda_report

eda-html:
	$(PYTHON) -m scripts.make_eda_html

preprocess-panel:
	$(PYTHON) -m scripts.prepare_panel_dataset

preprocessing-report:
	$(PYTHON) -m scripts.make_preprocessing_report

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

tune-nhits:
	$(PYTHON) -m scripts.tune_nhits

benchmark-deterministic:
	$(PYTHON) -m scripts.run_deterministic_benchmark

benchmark-probabilistic-lightgbm:
	$(PYTHON) -m scripts.run_probabilistic_lightgbm

benchmark-probabilistic-deepar:
	$(PYTHON) -m scripts.run_probabilistic_deepar

benchmark-probabilistic-nhits:
	$(PYTHON) -m scripts.run_probabilistic_nhits

benchmark-probabilistic-patchtst:
	$(PYTHON) -m scripts.run_probabilistic_patchtst

benchmark-probabilistic-prophet:
	$(PYTHON) -m scripts.run_probabilistic_prophet

benchmark-lstm:
	$(PYTHON) -m scripts.run_lstm_benchmark

benchmark-prophet:
	$(PYTHON) -m scripts.run_prophet_benchmark

benchmark-patchtst:
	$(PYTHON) -m scripts.run_patchtst_benchmark

benchmark-nhits:
	$(PYTHON) -m scripts.run_nhits_benchmark

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

export-forecast:
	$(PYTHON) -m scripts.export_probabilistic_forecast_for_simulation

final-artifacts:
	$(PYTHON) -m scripts.make_final_forecasting_artifacts

test:
	$(PYTHON) -m pytest
