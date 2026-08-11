#!/usr/bin/env python3
"""Build a reproducible 4-slice clustering dataset from CESNET aggregates.

The notebook is useful for experimentation, but this script is the pipeline
that should be rerun and compared. It:

1. reads the per-IP CESNET aggregate CSV files;
2. creates interpretable traffic features;
3. fits a scalable KMeans variant on a balanced sample;
4. predicts a cluster for every row;
5. maps clusters to network-slicing labels with documented heuristics;
6. exports the simulation CSV, model artifacts, and a quality report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler


SEED = 42
SLICE_NAMES = ["URLLC", "URLLC_eMBB_MIX", "eMBB", "mMTC"]

RAW_COLUMNS = [
    "id_time",
    "n_flows",
    "n_packets",
    "n_bytes",
    "n_dest_asn",
    "n_dest_ports",
    "n_dest_ip",
    "tcp_udp_ratio_packets",
    "tcp_udp_ratio_bytes",
    "dir_ratio_packets",
    "dir_ratio_bytes",
    "avg_duration",
    "avg_ttl",
]

MODEL_FEATURES = [
    "log_n_flows",
    "log_n_packets",
    "log_n_bytes",
    "log_avg_pkt_size",
    "log_bytes_per_flow",
    "log_pkts_per_flow",
    "log_avg_duration",
    "tcp_udp_ratio_bytes",
    "dir_ratio_bytes",
    "dest_ports_per_flow",
    "dest_ip_per_flow",
    "avg_ttl_norm",
    "burst_cv",
]

EXPORT_ENGINEERED_COLUMNS = [
    "avg_pkt_size",
    "bytes_per_flow",
    "pkts_per_flow",
    "burst_cv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agg-dir",
        default="data/raw/ip_sample/ip_addresses_sample/agg_10_minutes",
        help="Directory containing one aggregate CSV per IP.",
    )
    parser.add_argument(
        "--times-csv",
        default="data/raw/times/times/times_10_minutes.csv",
        help="CSV mapping id_time to timestamp.",
    )
    parser.add_argument(
        "--output-csv",
        default="simulation/mininet/cesnet_points_clustered_4slices.csv",
        help="Clustered CSV consumed by simulation/mininet/topology.py.",
    )
    parser.add_argument("--models-dir", default="dataset-prep/models", help="Directory for model artifacts.")
    parser.add_argument("--reports-dir", default="dataset-prep/reports", help="Directory for clustering reports.")
    parser.add_argument("--sample-per-file", type=int, default=300)
    parser.add_argument("--metrics-sample-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-files", type=int, default=None, help="Debug limit for number of IP files.")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def list_aggregate_files(agg_dir: str, max_files: int | None = None) -> List[Path]:
    paths = sorted(Path(agg_dir).glob("*.csv"))
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {agg_dir}")
    return paths


def ip_id_from_path(path: Path) -> int:
    return int(path.stem)


def add_features(df: pd.DataFrame, ip_id: int) -> pd.DataFrame:
    df = df.copy()
    df["ip_id"] = ip_id

    numeric_cols = [c for c in RAW_COLUMNS if c != "id_time"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["id_time"] = pd.to_numeric(df["id_time"], errors="coerce").astype("Int64")

    eps = 1e-9
    df["avg_pkt_size"] = df["n_bytes"] / (df["n_packets"] + eps)
    df["bytes_per_flow"] = df["n_bytes"] / (df["n_flows"] + eps)
    df["pkts_per_flow"] = df["n_packets"] / (df["n_flows"] + eps)

    df["log_n_flows"] = np.log1p(df["n_flows"])
    df["log_n_packets"] = np.log1p(df["n_packets"])
    df["log_n_bytes"] = np.log1p(df["n_bytes"])
    df["log_avg_pkt_size"] = np.log1p(df["avg_pkt_size"])
    df["log_bytes_per_flow"] = np.log1p(df["bytes_per_flow"])
    df["log_pkts_per_flow"] = np.log1p(df["pkts_per_flow"])
    df["log_avg_duration"] = np.log1p(df["avg_duration"].clip(lower=0))

    df["dest_ports_per_flow"] = (df["n_dest_ports"] / (df["n_flows"] + eps)).clip(0, 1)
    df["dest_ip_per_flow"] = (df["n_dest_ip"] / (df["n_flows"] + eps)).clip(0, 1)
    df["avg_ttl_norm"] = (df["avg_ttl"] / 255.0).clip(0, 1)

    bytes_log = df["log_n_bytes"]
    burst_cv = float(bytes_log.std(ddof=0) / max(bytes_log.mean(), eps))
    df["burst_cv"] = np.clip(burst_cv, 0, 10)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=MODEL_FEATURES + ["id_time"])


def load_featured_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=RAW_COLUMNS)
    return add_features(df, ip_id_from_path(path))


def build_training_sample(paths: List[Path], sample_per_file: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    samples = []
    for path in paths:
        df = load_featured_file(path)
        if len(df) > sample_per_file:
            random_state = int(rng.integers(0, np.iinfo(np.int32).max))
            df = df.sample(n=sample_per_file, random_state=random_state)
        samples.append(df)
    return pd.concat(samples, ignore_index=True)


def fit_clusterer(sample: pd.DataFrame, batch_size: int, seed: int) -> Tuple[RobustScaler, MiniBatchKMeans, np.ndarray]:
    scaler = RobustScaler()
    x_sample = scaler.fit_transform(sample[MODEL_FEATURES].astype(float))
    kmeans = MiniBatchKMeans(
        n_clusters=4,
        random_state=seed,
        n_init=20,
        batch_size=batch_size,
        reassignment_ratio=0.01,
    )
    sample_clusters = kmeans.fit_predict(x_sample)
    return scaler, kmeans, sample_clusters


def compute_cluster_profile(sample: pd.DataFrame, sample_clusters: np.ndarray) -> pd.DataFrame:
    profiled = sample.copy()
    profiled["cluster"] = sample_clusters
    profile_cols = list(dict.fromkeys(MODEL_FEATURES + EXPORT_ENGINEERED_COLUMNS + [
        "n_flows",
        "n_packets",
        "n_bytes",
        "avg_duration",
    ]))
    profile = profiled.groupby("cluster")[profile_cols].median()
    profile["count_sample"] = profiled.groupby("cluster").size()
    return profile.reset_index()


def minmax(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / span


def build_slice_indicators(profile: pd.DataFrame) -> pd.DataFrame:
    p = profile.set_index("cluster").copy()

    volume = minmax(p["log_n_bytes"]) + minmax(p["log_bytes_per_flow"]) + minmax(p["log_avg_pkt_size"])
    density = minmax(p["log_n_packets"]) + minmax(p["log_pkts_per_flow"])
    burst = minmax(p["burst_cv"])
    duration_low = 1 - minmax(p["log_avg_duration"])
    small_payload = 1 - minmax(p["log_avg_pkt_size"])
    diversity = minmax(p["dest_ports_per_flow"]) + minmax(p["dest_ip_per_flow"])

    indicators = pd.DataFrame(index=p.index)
    indicators["volume_score"] = volume
    indicators["density_score"] = density
    indicators["burst_score"] = burst
    indicators["duration_low_score"] = duration_low
    indicators["small_payload_score"] = small_payload
    indicators["diversity_score"] = diversity
    indicators["embb_score"] = 1.8 * volume + 0.5 * density
    indicators["urllc_score"] = 1.8 * duration_low + 0.9 * burst + 0.5 * density
    indicators["mmtc_score"] = 1.3 * diversity + 0.9 * small_payload - 0.7 * minmax(p["log_n_bytes"])
    indicators["mix_score"] = 0.9 * volume + 0.8 * density + 0.8 * burst + 0.3 * duration_low
    return indicators


def map_clusters_to_slices(profile: pd.DataFrame) -> Dict[int, str]:
    """Map arbitrary cluster ids to slice labels with explicit slice semantics.

    CESNET does not provide ground-truth slice labels. The assignment is therefore
    a documented interpretation of cluster profiles, not supervised labeling:

    * eMBB: highest volume / payload-heavy cluster.
    * URLLC: shortest-duration cluster, used as a latency proxy.
    * mMTC: low-volume, small-payload, high-destination-diversity cluster.
    * URLLC_eMBB_MIX: remaining hybrid/intermediate cluster.
    """
    indicators = build_slice_indicators(profile)
    remaining = set(int(c) for c in indicators.index)
    rationale = []

    embb_cluster = int(indicators.loc[list(remaining), "embb_score"].idxmax())
    remaining.remove(embb_cluster)
    rationale.append({
        "slice": "eMBB",
        "cluster": embb_cluster,
        "reason": "highest volume/payload score",
        "score": float(indicators.loc[embb_cluster, "embb_score"]),
    })

    urllc_cluster = int(indicators.loc[list(remaining), "urllc_score"].idxmax())
    remaining.remove(urllc_cluster)
    rationale.append({
        "slice": "URLLC",
        "cluster": urllc_cluster,
        "reason": "highest low-duration and burst score",
        "score": float(indicators.loc[urllc_cluster, "urllc_score"]),
    })

    mmtc_cluster = int(indicators.loc[list(remaining), "mmtc_score"].idxmax())
    remaining.remove(mmtc_cluster)
    rationale.append({
        "slice": "mMTC",
        "cluster": mmtc_cluster,
        "reason": "highest low-volume small-payload diversity score",
        "score": float(indicators.loc[mmtc_cluster, "mmtc_score"]),
    })

    mix_cluster = int(next(iter(remaining)))
    rationale.append({
        "slice": "URLLC_eMBB_MIX",
        "cluster": mix_cluster,
        "reason": "remaining intermediate/hybrid traffic profile",
        "score": float(indicators.loc[mix_cluster, "mix_score"]),
    })

    profile.attrs["slice_indicators"] = indicators.reset_index().to_dict(orient="records")
    profile.attrs["slice_assignment_rationale"] = rationale
    return {
        embb_cluster: "eMBB",
        urllc_cluster: "URLLC",
        mmtc_cluster: "mMTC",
        mix_cluster: "URLLC_eMBB_MIX",
    }


def compute_metrics(x_sample: np.ndarray, sample_clusters: np.ndarray, metrics_sample_size: int, seed: int) -> Dict[str, float]:
    if len(x_sample) > metrics_sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_sample), size=metrics_sample_size, replace=False)
        x_eval = x_sample[idx]
        y_eval = sample_clusters[idx]
    else:
        x_eval = x_sample
        y_eval = sample_clusters

    return {
        "silhouette": float(silhouette_score(x_eval, y_eval)),
        "davies_bouldin": float(davies_bouldin_score(x_eval, y_eval)),
        "calinski_harabasz": float(calinski_harabasz_score(x_eval, y_eval)),
        "metrics_sample_size": int(len(x_eval)),
    }


def predict_and_export(
    paths: List[Path],
    times_csv: str,
    output_csv: str,
    scaler: RobustScaler,
    kmeans: MiniBatchKMeans,
    cluster_to_slice: Dict[int, str],
) -> Dict[str, int]:
    times = pd.read_csv(times_csv)
    if "time_id" in times.columns and "id_time" not in times.columns:
        times = times.rename(columns={"time_id": "id_time"})
    if "time" in times.columns:
        times = times.rename(columns={"time": "timestamp"})

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    counts = {name: 0 for name in SLICE_NAMES}
    total = 0
    wrote_header = False

    export_cols = [
        "ip_id",
        "id_time",
        "timestamp",
        "n_bytes",
        "n_packets",
        "n_flows",
        "tcp_udp_ratio_bytes",
        "dir_ratio_bytes",
        *EXPORT_ENGINEERED_COLUMNS,
        "cluster",
        "slice",
    ]

    for path in paths:
        df = load_featured_file(path)
        x = scaler.transform(df[MODEL_FEATURES].astype(float))
        df["cluster"] = kmeans.predict(x)
        df["slice"] = df["cluster"].map(cluster_to_slice)
        df = df.merge(times[["id_time", "timestamp"]], on="id_time", how="left")

        for slice_name, count in df["slice"].value_counts().items():
            counts[slice_name] += int(count)
        total += len(df)

        df[export_cols].to_csv(output_path, mode="a", index=False, header=not wrote_header)
        wrote_header = True

    counts["total"] = total
    return counts


def main() -> None:
    args = parse_args()
    paths = list_aggregate_files(args.agg_dir, args.max_files)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)

    sample = build_training_sample(paths, args.sample_per_file, args.seed)
    scaler, kmeans, sample_clusters = fit_clusterer(sample, args.batch_size, args.seed)

    x_sample = scaler.transform(sample[MODEL_FEATURES].astype(float))
    metrics = compute_metrics(x_sample, sample_clusters, args.metrics_sample_size, args.seed)

    profile = compute_cluster_profile(sample, sample_clusters)
    cluster_to_slice = map_clusters_to_slices(profile)
    profile["slice"] = profile["cluster"].map(cluster_to_slice)

    counts = predict_and_export(
        paths=paths,
        times_csv=args.times_csv,
        output_csv=args.output_csv,
        scaler=scaler,
        kmeans=kmeans,
        cluster_to_slice=cluster_to_slice,
    )

    joblib.dump(scaler, Path(args.models_dir) / "scaler_4clusters.pkl")
    joblib.dump(kmeans, Path(args.models_dir) / "kmeans_4clusters.pkl")
    joblib.dump(cluster_to_slice, Path(args.models_dir) / "cluster_to_slice.pkl")

    profile_path = Path(args.reports_dir) / "cluster_4_slices_profile.csv"
    report_path = Path(args.reports_dir) / "cluster_4_slices_report.json"
    profile.to_csv(profile_path, index=False)

    report = {
        "seed": args.seed,
        "agg_dir": args.agg_dir,
        "files": len(paths),
        "sample_rows": int(len(sample)),
        "features": MODEL_FEATURES,
        "cluster_to_slice": cluster_to_slice,
        "slice_counts": counts,
        "metrics": metrics,
        "slice_indicators": profile.attrs.get("slice_indicators", []),
        "slice_assignment_rationale": profile.attrs.get("slice_assignment_rationale", []),
        "outputs": {
            "clustered_csv": args.output_csv,
            "profile_csv": str(profile_path),
            "report_json": str(report_path),
            "models_dir": args.models_dir,
        },
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
