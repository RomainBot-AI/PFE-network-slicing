"""Run the Prophet interval probabilistic benchmark."""

from __future__ import annotations

import argparse

from nsf.benchmark.probabilistic_prophet import load_probabilistic_prophet_config, run_probabilistic_prophet_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probabilistic Prophet benchmark")
    parser.add_argument(
        "--config",
        default="configs/experiment/probabilistic_prophet_14d.yaml",
        help="Path to a probabilistic Prophet benchmark YAML config.",
    )
    args = parser.parse_args()
    paths = run_probabilistic_prophet_benchmark(load_probabilistic_prophet_config(args.config))
    print("Probabilistic Prophet outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
