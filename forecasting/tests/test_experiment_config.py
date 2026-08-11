from pathlib import Path

from nsf.config import load_experiment_config


def test_load_yaml_experiment_config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
seed: 7
data:
  panel_csv: panel.csv
  frequency: 10min
backtest:
  input_size: 12
  horizon: 3
  n_folds: 2
  fold_stride: 4
models:
  - name: persistence
output:
  run_dir: forecasting/experiments/runs/test
""",
        encoding="utf-8",
    )

    config = load_experiment_config(path)

    assert config.seed == 7
    assert config.data.panel_csv == "panel.csv"
    assert config.backtest.horizon == 3
    assert config.models[0].name == "persistence"
