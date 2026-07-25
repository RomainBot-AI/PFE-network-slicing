from pathlib import Path

import pandas as pd
import yaml

from nsf.preprocessing.panel_dataset import load_preprocess_config, prepare_panel_feature_dataset


def test_prepare_panel_feature_dataset_uses_train_only_lags_and_scalers(tmp_path: Path):
    rows = []
    timestamps = pd.date_range("2024-01-01", periods=30, freq="10min")
    for unique_id, slice_name, offset in [("subnet_1__URLLC", "URLLC", 0), ("subnet_2__eMBB", "eMBB", 100)]:
        for idx, ts in enumerate(timestamps):
            rows.append(
                {
                    "unique_id": unique_id,
                    "ds": ts,
                    "y": float(idx + offset),
                    "slice": slice_name,
                    "id_institution": 1,
                    "id_institution_subnet": int(offset + 1),
                }
            )
    panel_csv = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(panel_csv, index=False)

    config_path = tmp_path / "preprocess.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "data": {"panel_csv": str(panel_csv), "frequency": "10min"},
                "backtest": {
                    "input_size": 6,
                    "horizon": 2,
                    "n_folds": 2,
                    "fold_stride": 3,
                    "step_size": 3,
                    "expanding": False,
                },
                "features": {"lags": [1, 3, 6], "calendar": True, "log_target": True, "scale_per_series": True},
                "output": {"output_dir": str(tmp_path / "processed")},
            }
        ),
        encoding="utf-8",
    )

    paths = prepare_panel_feature_dataset(load_preprocess_config(config_path))
    features = pd.read_csv(paths["features"])
    scalers = pd.read_csv(paths["scalers"])
    audit = pd.read_csv(paths["feature_audit"])

    first = features[(features["fold"] == 0) & (features["unique_id"] == "subnet_1__URLLC") & (features["horizon"] == 1)].iloc[0]
    assert first["lag_1"] == 24.0
    assert first["lag_3"] == 22.0
    assert first["lag_6"] == 19.0
    assert first["y"] == 25.0
    assert len(scalers) == 4
    assert audit["train_end_before_target"].all()
    assert audit["max_lag_available_in_train"].all()
