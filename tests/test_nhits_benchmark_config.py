from pathlib import Path

import yaml

from nsf.benchmark.nhits_benchmark import load_nhits_benchmark_config


def test_load_nhits_benchmark_config(tmp_path: Path):
    config_path = tmp_path / "nhits.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 7,
                "data": {"panel_csv": "panel.csv", "frequency": "10min"},
                "backtest": {"input_size": 12, "horizon": 3, "n_folds": 2, "fold_stride": 4, "step_size": 4, "expanding": False},
                "model": {
                    "max_steps": 5,
                    "mlp_units": [[16, 16]],
                    "stack_types": ["identity"],
                    "n_blocks": [1],
                    "n_pool_kernel_size": [2],
                    "n_freq_downsample": [1],
                },
                "training": {"device": "cpu", "slices": ["mMTC"]},
                "output": {"output_dir": "runs/nhits"},
            }
        ),
        encoding="utf-8",
    )

    config = load_nhits_benchmark_config(config_path)

    assert config.seed == 7
    assert config.model.max_steps == 5
    assert config.model.mlp_units == ((16, 16),)
    assert config.training.slices == ("mMTC",)
    assert config.output_dir == "runs/nhits"
