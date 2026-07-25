"""Run the probabilistic PatchTST quantile benchmark."""

from __future__ import annotations

import argparse

from nsf.benchmark.probabilistic_neural_quantile import load_probabilistic_patchtst_config, run_probabilistic_neural_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probabilistic PatchTST benchmark")
    parser.add_argument(
        "--config",
        default="configs/experiment/probabilistic_patchtst_14d.yaml",
        help="Path to a probabilistic PatchTST benchmark YAML config.",
    )
    args = parser.parse_args()
    paths = run_probabilistic_neural_benchmark(load_probabilistic_patchtst_config(args.config))
    print("Probabilistic PatchTST outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
