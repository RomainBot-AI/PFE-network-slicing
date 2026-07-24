"""Run the probabilistic DeepAR benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from nsf.benchmark.probabilistic_deepar import load_probabilistic_deepar_config, run_probabilistic_deepar_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probabilistic DeepAR benchmark")
    parser.add_argument(
        "--config",
        default="configs/experiment/probabilistic_deepar_14d.yaml",
        help="Path to a probabilistic DeepAR benchmark YAML config.",
    )
    args = parser.parse_args()
    paths = run_probabilistic_deepar_benchmark(load_probabilistic_deepar_config(args.config))
    print("Probabilistic DeepAR outputs:")
    for name, path in paths.items():
        print(f"- {name}: {Path(path)}")


if __name__ == "__main__":
    main()
