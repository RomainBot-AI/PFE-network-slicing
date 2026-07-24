# Data Files

Large datasets are not stored in this Git repository.

## Required Benchmark Dataset

Most benchmark configs expect this local file:

```text
traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
```

This file is the dense 10-minute subnet/slice traffic panel used by the deterministic and probabilistic forecasting benchmarks.

Expected columns include:

- `unique_id`
- `ds`
- `y`
- `slice`

## Local Placement

Create the data directory if needed:

```bash
mkdir -p traffic_forecasting/data
```

Then place the benchmark CSV at:

```text
traffic_forecasting/data/subnet_slice_traffic_min2016_dense.csv
```

## Other Local Data

Some preprocessing scripts may use files under:

```text
data/raw/
data/interim/
data/processed/
```

These directories are ignored by Git because they can contain large raw or generated datasets.

## Generated Outputs

Benchmark outputs are generated locally under:

```text
experiments/runs/
```

This directory is ignored by Git. Regenerate outputs with the commands in `README.md`.

