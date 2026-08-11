"""End-to-end pipeline smoke test on a synthetic panel."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from src.pipeline.config import RunConfig
from src.pipeline.trainer_evaluator import run_single_model_pipeline


def test_single_model_pipeline_runs(synthetic_csv, tmp_path):
    config = RunConfig(
        model_name="passthrough",
        subnet_choice="0",
        max_steps=25,
        episodes=1,
        dataset_path=synthetic_csv,
        output_dir=str(tmp_path / "plots"),
    )
    results = run_single_model_pipeline(config)

    for key in ("model_name", "energy_opt_test", "energy_gain_test", "qos_test", "nmae_test"):
        assert key in results
    assert results["model_name"] == "passthrough"
    assert results["nmae_test"] == 0.0  # oracle predictor
    assert 0.0 <= results["qos_test"] <= 1.0
    assert (tmp_path / "plots" / "passthrough").exists()
