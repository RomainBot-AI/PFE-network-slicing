"""Run the probabilistic N-HiTS quantile benchmark."""

from __future__ import annotations

import argparse

from nsf.benchmark.probabilistic_neural_quantile import load_probabilistic_nhits_config, run_probabilistic_neural_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probabilistic N-HiTS benchmark")
    parser.add_argument(
        "--config",
        default="configs/experiment/probabilistic_nhits_1d.yaml",
        help="Path to a probabilistic N-HiTS benchmark YAML config.",
    )
    args = parser.parse_args()
    paths = run_probabilistic_neural_benchmark(load_probabilistic_nhits_config(args.config))
    print("Probabilistic N-HiTS outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
