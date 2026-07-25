from pathlib import Path

import pandas as pd
import yaml

from nsf.benchmark.deterministic import load_benchmark_config, run_deterministic_benchmark
from nsf.evaluation.deterministic import mase


def test_mase_uses_external_scale():
    assert mase([10.0, 12.0], [9.0, 15.0], scale=2.0) == 1.0


def test_deterministic_benchmark_baseline_outputs(tmp_path: Path):
    rows = []
    timestamps = pd.date_range("2024-01-01", periods=40, freq="10min")
    for unique_id, slice_name, offset in [("subnet_1__URLLC", "URLLC", 0), ("subnet_2__eMBB", "eMBB", 100)]:
        for idx, ts in enumerate(timestamps):
            rows.append({"unique_id": unique_id, "ds": ts, "y": float(idx + offset), "slice": slice_name})
    panel_csv = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(panel_csv, index=False)

    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "data": {"panel_csv": str(panel_csv), "frequency": "10min"},
                "backtest": {"input_size": 12, "horizon": 2, "n_folds": 2, "fold_stride": 4, "step_size": 4, "expanding": False},
                "features": {"seasonal_scale_period": 3, "train_origin_stride": 2, "lags": [1, 2, 3]},
                "models": [{"name": "persistence"}],
                "output": {"run_dir": str(tmp_path / "run")},
            }
        ),
        encoding="utf-8",
    )

    paths = run_deterministic_benchmark(load_benchmark_config(config_path))
    predictions = pd.read_csv(paths["predictions"])
    metrics = pd.read_csv(paths["metrics"])
    summary = pd.read_csv(paths["benchmark_summary"])

    assert len(predictions) == 8
    assert {"MAE", "RMSE", "WAPE", "MASE"}.issubset(metrics.columns)
    assert summary["model"].tolist() == ["persistence"]
