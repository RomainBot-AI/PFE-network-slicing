"""Run configuration for the offline slicing simulation.

Default paths resolve relative to this file so the project runs from any machine:
the dataset defaults to the repository's canonical panel, and figures default to
``implementation/data/plots``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_IMPL_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATASET_PATH = str(_REPO_ROOT / "forecasting" / "data" / "subnet_slice_traffic_min2016_dense.csv")
DEFAULT_OUTPUT_DIR = str(_IMPL_ROOT / "data" / "plots")


@dataclass
class RunConfig:
    """Parameters for a single simulation/benchmark run."""

    model_name: str = "passthrough"
    subnet_choice: str = "all"
    num_rans: int = 0
    max_steps: Optional[int] = None
    episodes: int = 1
    beta: float = 10.0
    lambda_loss: float = 50.0
    log_freq: int = 1000
    dataset_path: str = DEFAULT_DATASET_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    seed: int = 42
