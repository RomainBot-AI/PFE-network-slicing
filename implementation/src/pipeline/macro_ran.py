#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial clustering and aggregation of subnets into K macro-RANs.

Groups the subnets into K clusters with K-means on their per-slice traffic
profile (log1p scale, with a quantile fallback when clusters are too imbalanced),
then sums the demand of every subnet in a group for each (timestamp, slice). This
keeps 100% of the network load while cutting the number of time steps, so it
represents K regional gNodeBs / macro-cells and trains much faster.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("macro_ran")


def build_subnet_to_macro_map(
    raw: pd.DataFrame,
    num_rans: int,
    col_ran: str = "id_institution_subnet",
    col_slice: str = "slice",
    col_value: str = "y",
    seed: int = 42,
    max_imbalance_ratio: float = 4.0
) -> Dict[int, int]:
    """Build the mapping ``{subnet_id -> macro_ran_id}`` (0..K-1).

    Uses K-means on the log1p scale with a quantile-based fallback when the
    resulting clusters are too imbalanced.
    """
    profile = raw.pivot_table(index=col_ran, columns=col_slice, values=col_value, aggfunc="sum", fill_value=0.0)
    profile["__total__"] = profile.sum(axis=1)

    if len(profile) <= num_rans:
        logger.info(f"  → {len(profile)} subnets pour {num_rans} macro-RAN demandés : pas de clustering nécessaire")
        return {int(sid): i for i, sid in enumerate(profile.index)}

    log_profile = np.log1p(profile.values)
    X = StandardScaler().fit_transform(log_profile)
    km = KMeans(n_clusters=num_rans, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    sizes = pd.Series(labels).value_counts()
    imbalance = sizes.max() / max(sizes.min(), 1)

    if imbalance > max_imbalance_ratio:
        logger.warning(
            f"  → K-means (log1p) déséquilibré (ratio {imbalance:.1f}x) "
            f"— bascule automatique sur un découpage équitable par quantiles"
        )
        log_total = np.log1p(profile["__total__"].values)
        ranks = pd.Series(log_total, index=profile.index).rank(method="first")
        labels = pd.qcut(ranks, q=num_rans, labels=False).values
        sizes = pd.Series(labels).value_counts()

    mapping = {int(sid): int(lbl) for sid, lbl in zip(profile.index, labels)}
    logger.info(
        f"  ✓ {len(profile)} subnets regroupés avec succès en {num_rans} Macro-RAN "
        f"(tailles : {pd.Series(labels).value_counts().sort_index().to_dict()})"
    )
    return mapping


def aggregate_to_macro_rans(
    raw: pd.DataFrame,
    subnet_to_macro: Dict[int, int],
    col_time: str = "ds",
    col_ran: str = "id_institution_subnet",
    col_slice: str = "slice",
    col_value: str = "y"
) -> pd.DataFrame:
    """Aggregate the raw panel by macro-RAN, summing traffic per (timestamp, macro_ran, slice)."""
    df = raw[[col_time, col_ran, col_slice, col_value]].copy()
    df["macro_ran"] = df[col_ran].map(subnet_to_macro)
    if df["macro_ran"].isna().any():
        df = df.dropna(subset=["macro_ran"])
    df["macro_ran"] = df["macro_ran"].astype(int)

    agg = df.groupby([col_time, "macro_ran", col_slice], as_index=False)[col_value].sum()
    # Rename macro_ran -> id_institution_subnet for compatibility with the environment.
    agg = agg.rename(columns={"macro_ran": col_ran})
    return agg
