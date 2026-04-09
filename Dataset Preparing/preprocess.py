# %%#Imports
import os, glob, tarfile
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


SEED = 42
np.random.seed(SEED)


# %% [markdown]
# ## 1) Localiser les archives et extraire
# Ce bloc cherche automatiquement `ip_addresses_sample.tar.gz` et `times.tar.gz` dans `/content` et dans Google Drive (si monté).


os.makedirs("data/ip_sample", exist_ok=True)
os.makedirs("data/times", exist_ok=True)

ip_tar = "ip_addresses_sample.tar.gz"
times_tar = "times.tar.gz"

print("IP tar exists   :", os.path.exists(ip_tar))
print("TIMES tar exists:", os.path.exists(times_tar))

# %% [markdown]
# ## 2) Construire df_points (agg_10_minutes)
# On charge `times_10_minutes.csv` puis les fichiers `agg_10_minutes/*.csv` et on fusionne pour obtenir `timestamp`.

#Charger le mapping temps 10 minutes
time_path = "data/raw/times/times/times_10_minutes.csv"
df_t = pd.read_csv(time_path)
if "time_id" in df_t.columns and "id_time" not in df_t.columns:
    df_t = df_t.rename(columns={"time_id": "id_time"})

agg_files = sorted(glob.glob("data/raw/ip_sample/ip_addresses_sample/agg_10_minutes/*.csv"))
if len(agg_files) == 0:
    raise FileNotFoundError("Aucun fichier agg_10_minutes trouvé.")

DEBUG = False
DEBUG_N_FILES = 300
use_files = agg_files[:DEBUG_N_FILES] if DEBUG else agg_files

dfs = []
bad = 0

for f in use_files:
    try:
        d = pd.read_csv(f)

        # ip_id est dans le nom du fichier: ".../agg_10_minutes/1802574.csv"
        ip_id = os.path.splitext(os.path.basename(f))[0]
        d["ip_id"] = int(ip_id)

        dfs.append(d)
    except Exception as e:
        bad += 1

df_ip = pd.concat(dfs, ignore_index=True)

# Harmonisation id_time si besoin
if "time_id" in df_ip.columns and "id_time" not in df_ip.columns:
    df_ip = df_ip.rename(columns={"time_id": "id_time"})

if "id_time" in df_ip.columns and "id_time" in df_t.columns:
    df_points = df_ip.merge(df_t, on="id_time", how="left")
else:
    df_points = df_ip.copy()

print("df_points:", df_points.shape, "| bad files:", bad)
print("Has ip_id ?", "ip_id" in df_points.columns, "| unique ip_id:", df_points["ip_id"].nunique())
print("Has timestamp ?", "timestamp" in df_points.columns)
# %% [markdown]
# ## 3) Feature engineering + Clustering (k=4)
# On construit des features cohérentes avec :
# - **taille paquet** (`avg_pkt_size`)
# - **intensité / volume** (`n_bytes`, `bytes_per_flow`)
# - **interactivité/pattern** (`burst_cv` proxy)
# 
# Puis KMeans `k=4` + mapping cluster -> slice via scoring sur centroïdes (assignation 1-1).

# %%
#Clustering pipeline

FEATURES_BASE = [
    "n_bytes",
    "n_packets",
    "n_flows",
    "tcp_udp_ratio_bytes",
    "dir_ratio_bytes",
]

# Vérifier que les colonnes existent, sinon proposer les colonnes disponibles
missing_base = [c for c in FEATURES_BASE if c not in df_points.columns]
if missing_base:
    print("Colonnes manquantes (FEATURES_BASE):", missing_base)
    print("Colonnes dispo (extrait):", list(df_points.columns)[:80])
    raise KeyError("Adapte FEATURES_BASE aux noms réels de tes colonnes CESNET puis relance.")

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-9


    df["avg_pkt_size"]   = df["n_bytes"] / (df["n_packets"] + eps)
    df["bytes_per_flow"] = df["n_bytes"] / (df["n_flows"] + eps)
    df["pkts_per_flow"]  = df["n_packets"] / (df["n_flows"] + eps)

    for col in ["n_bytes","n_packets","n_flows","avg_pkt_size","bytes_per_flow","pkts_per_flow"]:
        df[col] = np.log1p(df[col].astype(float))

    df["burst_cv"] = 0.0
    if "ip_id" in df.columns:
        g = df.groupby("ip_id")["n_bytes"]
        mean_per_ip = g.transform("mean")
        std_per_ip  = g.transform("std").fillna(0.0)

        df["burst_cv"] = std_per_ip / (mean_per_ip + 1e-9)
        df["burst_cv"] = np.clip(df["burst_cv"], 0, 10)

    return df

df_feat = add_engineered_features(df_points)

FEATURES = FEATURES_BASE + ["avg_pkt_size","bytes_per_flow","pkts_per_flow","burst_cv"]

# Clean
df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES).copy()
X = df_feat[FEATURES].astype(float).values

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print("X_scaled:", X_scaled.shape)

# KMeans
k = 4
kmeans = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
df_feat["cluster"] = kmeans.fit_predict(X_scaled)

# Silhouette
m = min(20000, len(df_feat))
idx = np.random.choice(len(df_feat), size=m, replace=False)
sil = silhouette_score(X_scaled[idx], df_feat["cluster"].values[idx])
print("silhouette (approx):", round(float(sil), 4))
print(df_feat["cluster"].value_counts())

# Centroïdes
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids, columns=FEATURES)
centroids_df["cluster"] = np.arange(k)

# Scoring slice
def score_slices(row):

    nb = row["n_bytes"]
    ap = row["avg_pkt_size"]
    nf = row["n_flows"]
    bc = row["burst_cv"]

    score_embb = (2.0*nb + 1.5*ap - 1.0*bc - 0.3*nf)

    score_mmtc = (-2.0*nb - 1.0*ap - 0.5*nf - 0.2*bc)

    score_urllc = (-1.2*ap + 1.2*nf + 2.0*bc - 0.6*nb)

    score_mix = (1.5*nb + 1.2*ap + 0.8*nf + 2.0*bc)

    return pd.Series({
        "eMBB": score_embb,
        "mMTC": score_mmtc,
        "URLLC": score_urllc,
        "URLLC_eMBB_MIX": score_mix,
    })

scores = centroids_df.set_index("cluster").apply(score_slices, axis=1)

assigned = {}
used = set()
order = scores.max(axis=1).sort_values(ascending=False).index.tolist()
for cl in order:
    prefs = scores.loc[cl].sort_values(ascending=False)
    for s in prefs.index:
        if s not in used:
            assigned[cl] = s
            used.add(s)
            break

cluster_to_slice = assigned
print("cluster_to_slice:", cluster_to_slice)

df_feat["slice"] = df_feat["cluster"].map(cluster_to_slice)

print("\nCounts per slice:")


pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)

# %% [markdown]
# ## 5) Export
# Export CSV propre avec `cluster` + `slice`. Code robuste aux variations de noms (`ip_id`, `timestamp`, etc.).

# %%
#Export
OUT_PATH = "cesnet_points_clustered_4slices.csv"

id_candidates = ["ip_id", "id_ip", "ip", "src_ip"]
time_candidates = ["timestamp", "time", "datetime", "date_time"]
id_time_candidates = ["id_time", "time_id"]

def first_existing(cands, df):
    for c in cands:
        if c in df.columns:
            return c
    return None

col_ip = first_existing(id_candidates, df_feat)
col_ts = first_existing(time_candidates, df_feat)
col_id_time = first_existing(id_time_candidates, df_feat)

if col_ts is None:
    try:
        df_t2 = pd.read_csv("data/raw/times/times/times_10_minutes.csv")
        if "time_id" in df_t2.columns and "id_time" not in df_t2.columns:
            df_t2 = df_t2.rename(columns={"time_id": "id_time"})
        if "id_time" in df_feat.columns and "id_time" in df_t2.columns:
            df_feat = df_feat.merge(df_t2, on="id_time", how="left")
            col_ts = first_existing(time_candidates, df_feat)
    except Exception as e:
        print("timestamp non reconstruit:", e)

cols_out = []
for c in [col_ip, col_id_time, col_ts]:
    if c is not None:
        cols_out.append(c)

for c in FEATURES_BASE:
    if c in df_feat.columns:
        cols_out.append(c)

for c in ["avg_pkt_size","bytes_per_flow","pkts_per_flow","burst_cv"]:
    if c in df_feat.columns:
        cols_out.append(c)

cols_out += ["cluster","slice"]
cols_out = list(dict.fromkeys(cols_out))

missing = [c for c in cols_out if c not in df_feat.columns]
if missing:
    raise KeyError(f"Colonnes manquantes pour export: {missing}")

df_feat[cols_out].to_csv(OUT_PATH, index=False)
print("saved:", OUT_PATH, "| rows:", len(df_feat), "| cols:", len(cols_out))
print("Export columns:", cols_out)



