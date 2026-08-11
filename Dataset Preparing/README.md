# Dataset preparation

Builds the reproducible 4-slice dataset from CESNET per-IP aggregates,
labelling traffic as `URLLC`, `URLLC_eMBB_MIX`, `eMBB`, or `mMTC`.

## Scripts

- `explore_dataset.py` — dependency-free inspection of the available CSV files,
  schemas, row counts, and basic statistics.
- `cluster_4_slices.py` — the pipeline: reads per-IP aggregates, engineers
  interpretable traffic features, fits a MiniBatchKMeans on a balanced sample
  with a RobustScaler, predicts a cluster per row, and maps clusters to the four
  slice labels.

## Usage

```bash
python3 -m pip install -r requirements-data.txt
python3 "Dataset Preparing/explore_dataset.py"
python3 "Dataset Preparing/cluster_4_slices.py"
```

Quick run on a few files:

```bash
python3 "Dataset Preparing/cluster_4_slices.py" \
  --max-files 20 \
  --output-csv /tmp/cesnet_points_clustered_4slices.csv \
  --models-dir /tmp/pfe-models \
  --reports-dir /tmp/pfe-reports
```

## Inputs and outputs

Input: `data/raw/ip_sample/ip_addresses_sample/agg_10_minutes/` (one CSV per IP).

Outputs:

```text
simulation/mininet/cesnet_points_clustered_4slices.csv
models/scaler_4clusters.pkl
models/kmeans_4clusters.pkl
models/cluster_to_slice.pkl
reports/cluster_4_slices_profile.csv
reports/cluster_4_slices_report.json
```
