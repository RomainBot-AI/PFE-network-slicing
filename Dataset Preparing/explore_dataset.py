#!/usr/bin/env python3
"""Lightweight CESNET dataset explorer.

This script intentionally uses only the Python standard library so it can run
before the data-science dependencies are installed.
"""

import csv
import glob
import os


DATA_ROOT = os.path.join("data", "raw")
IP_SAMPLE_ROOT = os.path.join(DATA_ROOT, "ip_sample", "ip_addresses_sample")


def read_csv_head(path, n=3):
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = []
        for idx, row in enumerate(reader):
            if idx >= n:
                break
            rows.append(row)
    return header, rows


def summarize_csv(path):
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, header


def summarize_aggregate_folder(folder_name):
    folder = os.path.join(IP_SAMPLE_ROOT, folder_name)
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    total_rows = 0
    min_rows = None
    max_rows = 0
    min_file = None
    max_file = None
    min_time = None
    max_time = None
    schemas = {}

    for path in paths:
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            header = tuple(next(reader))
            schemas[header] = schemas.get(header, 0) + 1
            row_count = 0
            for row in reader:
                row_count += 1
                if row:
                    time_id = int(row[0])
                    min_time = time_id if min_time is None else min(min_time, time_id)
                    max_time = time_id if max_time is None else max(max_time, time_id)

        total_rows += row_count
        if min_rows is None or row_count < min_rows:
            min_rows = row_count
            min_file = os.path.basename(path)
        if row_count > max_rows:
            max_rows = row_count
            max_file = os.path.basename(path)

    print(folder_name)
    print(f"  files: {len(paths)}")
    print(f"  total_rows: {total_rows}")
    print(f"  avg_rows_per_file: {total_rows / max(1, len(paths)):.1f}")
    print(f"  min_rows: {min_rows} ({min_file})")
    print(f"  max_rows: {max_rows} ({max_file})")
    print(f"  id_time_range: {min_time}..{max_time}")
    print(f"  schema_variants: {len(schemas)}")
    for schema, count in schemas.items():
        print(f"  schema_count: {count}")
        print(f"  columns: {', '.join(schema)}")
        break


def summarize_numeric_10_minutes():
    columns = [
        "n_flows",
        "n_packets",
        "n_bytes",
        "tcp_udp_ratio_bytes",
        "dir_ratio_bytes",
        "avg_duration",
        "avg_ttl",
    ]
    stats = {
        column: {"n": 0, "sum": 0.0, "min": None, "max": None}
        for column in columns
    }

    paths = sorted(glob.glob(os.path.join(IP_SAMPLE_ROOT, "agg_10_minutes", "*.csv")))
    for path in paths:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for column in columns:
                    value = float(row[column])
                    stat = stats[column]
                    stat["n"] += 1
                    stat["sum"] += value
                    stat["min"] = value if stat["min"] is None else min(stat["min"], value)
                    stat["max"] = value if stat["max"] is None else max(stat["max"], value)

    print("agg_10_minutes numeric summary")
    for column in columns:
        stat = stats[column]
        mean = stat["sum"] / stat["n"] if stat["n"] else 0.0
        print(
            f"  {column}: n={stat['n']} min={stat['min']} "
            f"max={stat['max']} mean={mean:.4f}"
        )


def main():
    print("Metadata CSVs")
    for path in [
        os.path.join(DATA_ROOT, "times", "times", "times_10_minutes.csv"),
        os.path.join(DATA_ROOT, "times", "times", "times_1_hour.csv"),
        os.path.join(DATA_ROOT, "times", "times", "times_1_day.csv"),
        os.path.join(DATA_ROOT, "weekends_and_holidays.csv"),
        os.path.join(IP_SAMPLE_ROOT, "identifiers.csv"),
    ]:
        rows, header = summarize_csv(path)
        print(f"  {path}: rows={rows}, columns={header}")

    print()
    print("Aggregate folders")
    for folder_name in ["agg_10_minutes", "agg_1_hour", "agg_1_day"]:
        summarize_aggregate_folder(folder_name)

    print()
    summarize_numeric_10_minutes()

    sample_path = sorted(
        glob.glob(os.path.join(IP_SAMPLE_ROOT, "agg_10_minutes", "*.csv"))
    )[0]
    header, rows = read_csv_head(sample_path)
    print()
    print(f"Sample file: {sample_path}")
    print(f"  columns: {header}")
    for row in rows:
        print(f"  row: {row}")


if __name__ == "__main__":
    main()
