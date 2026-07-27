# Dataset preparation

Objectif : transformer les agrégats CESNET par IP en un CSV annoté avec 4 slices :

- `URLLC`
- `URLLC_eMBB_MIX`
- `eMBB`
- `mMTC`

## Source de vérité

Les notebooks historiques ne font plus partie du pipeline final. Ils ont servi a
l'exploration initiale, mais les résultats reproductibles du rapport doivent
être générés depuis les scripts versionnés de ce dossier.

## Exploration rapide

Sans dépendance externe :

```bash
python3 "Dataset Preparing/explore_dataset.py"
```

Ce script vérifie les fichiers disponibles, les schémas CSV, les volumes de lignes
et quelques statistiques simples.

## Clustering 4 slices

Installer les dépendances :

```bash
python3 -m pip install -r requirements-data.txt
```

Lancer le pipeline complet :

```bash
python3 "Dataset Preparing/cluster_4_slices.py"
```

Sorties principales :

- `simulation/mininet/cesnet_points_clustered_4slices.csv`
- `models/scaler_4clusters.pkl`
- `models/kmeans_4clusters.pkl`
- `models/cluster_to_slice.pkl`
- `reports/cluster_4_slices_profile.csv`
- `reports/cluster_4_slices_report.json`

Pour tester rapidement sur quelques fichiers :

```bash
python3 "Dataset Preparing/cluster_4_slices.py" \
  --max-files 20 \
  --output-csv /tmp/cesnet_points_clustered_4slices.csv \
  --models-dir /tmp/pfe-models \
  --reports-dir /tmp/pfe-reports
```
