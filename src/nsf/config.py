"""Experiment configuration loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    panel_csv: str
    frequency: str = "10min"


@dataclass(frozen=True)
class BacktestConfig:
    input_size: int = 2016
    horizon: int = 36
    step_size: int = 144
    n_folds: int = 5
    fold_stride: int = 144
    mode: str = "rolling_origin"
    expanding: bool = False


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    run_dir: str = "experiments/runs/subnet_slice_baseline"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    backtest: BacktestConfig
    models: tuple[ModelConfig, ...]
    output: OutputConfig


def _read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = _read_config(path)
    models = tuple(
        ModelConfig(name=item["name"], params=dict(item.get("params", {})))
        for item in raw.get("models", [])
    )
    if not models:
        raise ValueError("Config must define at least one model")
    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        data=DataConfig(**raw["data"]),
        backtest=BacktestConfig(**raw["backtest"]),
        models=models,
        output=OutputConfig(**raw.get("output", {})),
    )


def resolved_config_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "data": {
            "panel_csv": config.data.panel_csv,
            "frequency": config.data.frequency,
        },
        "backtest": {
            "input_size": config.backtest.input_size,
            "horizon": config.backtest.horizon,
            "step_size": config.backtest.step_size,
            "n_folds": config.backtest.n_folds,
            "fold_stride": config.backtest.fold_stride,
            "mode": config.backtest.mode,
            "expanding": config.backtest.expanding,
        },
        "models": [{"name": model.name, "params": model.params} for model in config.models],
        "output": {"run_dir": config.output.run_dir},
    }
