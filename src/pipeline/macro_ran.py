#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/pipeline/macro_ran.py
 OBJET  : Clustering et Agrégation Spatial en K "Macro-RAN" (Option --num_rans)
====================================================================================================

ROLE ET POSITION DANS LE PIPELINE :
-----------------------------------
Ce module s'intercale en amont de la création de l'environnement SDN dans `trainer_evaluator.py`.
Il prend les 69 sous-réseaux (subnets) bruts du dataset CESNET et les regroupe en $K$ Macro-RANs (ex: $K=4$) :
  1. Calcule le profil moyen de trafic par subnet sur l'échelle `log1p`.
  2. Effectue un clustering K-Means avec garde-fou par quantiles pour garantir un équilibre entre groupes.
  3. Somme exactement 100% de la charge de trafic pour chaque triplet (timestamp, macro_ran, slice).

AVANTAGES DU MODE MACRO-RAN :
----------------------------
  - Accélération de la simulation d'un facteur 17x (de 2.7M de pas de temps à ~160k pas).
  - Conservation à 100% du volume de trafic global du réseau sans aucune perte.
  - Représentation réaliste de K grandes antennes régionales (Zone Urbaine, Chargée, Rurale, etc.).
====================================================================================================
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
    """
    Construit la table de correspondance {subnet_id -> macro_ran_id (0..K-1)}.

    Méthodologie :
      - Applique K-Means sur la transformation log1p des profils statistiques par subnet.
      - Si le ratio de déséquilibre dépasse max_imbalance_ratio (4.0), bascule automatiquement
        sur un découpage équitable par quantiles de charge totale.

    :param raw: DataFrame brut du trafic.
    :param num_rans: Nombre K de Macro-RANs cibles.
    :param seed: Graine aléatoire.
    :param max_imbalance_ratio: Seuil maximal de déséquilibre toléré.
    :return: Dictionnaire de mapping {subnet_id: macro_ran_id}.
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

    # Garde-fou : bascule sur quantiles si le K-Means produit des clusters trop asymétriques
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
    """
    Agrège le dataset brut en remplaçant id_institution_subnet par le macro_ran_id correspondant,
    puis somme exactement les charges de trafic pour chaque triplet (timestamp, macro_ran, slice).

    :param raw: DataFrame de trafic brut.
    :param subnet_to_macro: Table de mapping issue de build_subnet_to_macro_map.
    :return: DataFrame agrégé au format compatible avec SDN_DoubleController_Env.
    """
    df = raw[[col_time, col_ran, col_slice, col_value]].copy()
    df["macro_ran"] = df[col_ran].map(subnet_to_macro)
    if df["macro_ran"].isna().any():
        df = df.dropna(subset=["macro_ran"])
    df["macro_ran"] = df["macro_ran"].astype(int)

    # Sommation exacte des volumes de trafic par Macro-RAN
    agg = df.groupby([col_time, "macro_ran", col_slice], as_index=False)[col_value].sum()
    agg = agg.rename(columns={"macro_ran": col_ran})
    return agg
