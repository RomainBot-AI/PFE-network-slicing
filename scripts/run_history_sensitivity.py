#!/usr/bin/env python3
"""Run all input-history sensitivity benchmarks from one YAML file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from nsf.benchmark.deterministic import load_benchmark_config, run_deterministic_benchmark
from nsf.benchmark.lstm_benchmark import load_lstm_benchmark_config, run_lstm_benchmark
from nsf.benchmark.nhits_benchmark import load_nhits_benchmark_config, run_nhits_benchmark
from nsf.benchmark.patchtst_benchmark import load_patchtst_benchmark_config, run_patchtst_benchmark
from nsf.benchmark.prophet_benchmark import load_prophet_benchmark_config, run_prophet_benchmark
from nsf.visualization.benchmark_html import build_benchmark_html
from nsf.visualization.history_sensitivity import build_history_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/history_sensitivity_runs.yaml")
    parser.add_argument("--force", action="store_true", help="Rerun benchmarks even when benchmark_summary.csv exists.")
    return parser.parse_args()


def _run(kind: str, config_path: str) -> None:
    if kind == "deterministic":
        run_deterministic_benchmark(load_benchmark_config(config_path))
    elif kind == "lstm":
        run_lstm_benchmark(load_lstm_benchmark_config(config_path))
    elif kind == "nhits":
        run_nhits_benchmark(load_nhits_benchmark_config(config_path))
    elif kind == "patchtst":
        run_patchtst_benchmark(load_patchtst_benchmark_config(config_path))
    elif kind == "prophet":
        run_prophet_benchmark(load_prophet_benchmark_config(config_path))
    else:
        raise ValueError(f"Unknown run kind: {kind}")


def _is_complete(output_dir: str) -> bool:
    run_dir = Path(output_dir)
    required = [
        "benchmark_summary.csv",
        "benchmark_summary_by_slice.csv",
        "metrics.csv",
        "predictions.csv",
        "run_meta.json",
    ]
    return all((run_dir / name).exists() for name in required)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    args = parse_args()
    raw: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    skip_existing = bool(raw.get("skip_existing", True)) and not args.force
    make_reports = bool(raw.get("make_reports", True))
    output_dir = str(raw.get("comparison_output_dir", "reports"))

    for index, run in enumerate(raw["runs"], start=1):
        run_id = str(run["id"])
        kind = str(run["kind"])
        config_path = str(run["config"])
        run_dir = str(run["output_dir"])
        if skip_existing and _is_complete(run_dir):
            print(f"[{index}/{len(raw['runs'])}] {run_id}: skip existing", flush=True)
        else:
            print(f"[{index}/{len(raw['runs'])}] {run_id}: start", flush=True)
            _run(kind, config_path)
            print(f"[{index}/{len(raw['runs'])}] {run_id}: done", flush=True)
        if make_reports and _is_complete(run_dir):
            report = build_benchmark_html(run_dir)
            print(f"[{index}/{len(raw['runs'])}] {run_id}: report {report}", flush=True)

    paths = build_history_sensitivity(output_dir)
    print("History sensitivity outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
