# 5G/6G RAN Network Slicing Simulator, Forecasting & PPO Agent

Ce dépôt contient le code source, les pipelines de prédiction temporelle, l'environnement de simulation SDN et les scripts de rapportage du projet de Fin d'Études (PFE) **"Network Slicing, prédiction de trafic et allocation proactive des ressources par IA"** (CY Tech).

---

## 🎯 Architecture Générale du Projet

Le projet relie l'analyse de données de trafic réseau réel à la prise de décision automatisée par Apprentissage par Renforcement (RL) :

```
                     ┌─────────────────────────────────────────────────────────────┐
                     │              PARTIE I : DATA & FORECASTING                  │
                     └──────────────────────────────┬──────────────────────────────┘
                                                    │
 ┌───────────────────────────┐        ┌─────────────▼─────────────┐        ┌───────────────────────────┐
 │   Traces Réelles CESNET   │ ─────> │    Clustering KMeans K=4  │ ─────> │   Panel Subnet / Slice    │
 │ (6,86M lignes de trafic)  │        │ (eMBB, URLLC, mMTC, MIX)  │        │ (179 séries sur 69 subnets)│
 └───────────────────────────┘        └───────────────────────────┘        └─────────────┬─────────────┘
                                                                                         │
                                                                           ┌─────────────▼─────────────┐
                                                                           │  Modèles de Forecasting   │
                                                                           │ (LightGBM, LSTM, N-HiTS,  │
                                                                           │  Prophet, LightGBM q90)   │
                                                                           └─────────────┬─────────────┘
                                                                                         │
                     ┌─────────────────────────────────────────────────────────────┐     │
                     │          PARTIE II : SIMULATION SDN & AGENT PPO             │ <───┘
                     └──────────────────────────────┬──────────────────────────────┘
                                                    │
 ┌───────────────────────────┐        ┌─────────────▼─────────────┐        ┌───────────────────────────┐
 │   Agrégation 4 Macro-RANs │ ─────> │ Architecture SDN 2 Niveaux│ ─────> │   Agent PPO (RL PyTorch)  │
 │  (Urbain, Chargé, Rural)  │        │ (Filtre, Veto, EcoSlice)  │        │  (Campagne Finale V12)    │
 └───────────────────────────┘        └───────────────────────────┘        └───────────────────────────┘
```

---

## 📁 Structure du Code Source

```
.
├── main.py                             # Point d'entrée CLI principal
├── pyproject.toml                      # Gestion des dépendances Python (uv)
├── README.md                           # Documentation du projet
├── data/
│   ├── experiments_v12/                # Dossiers des résultats des simulations V12
panel)
└── src/
    ├── agents/
    │   └── ppo_agent.py                # Agent RL PPO en PyTorch (Acteur-Critique multi-binaire)
    ├── environment/
    │   └── sdn_controller_env.py       # Environnement Gymnasium du contrôleur SDN (Algorithmes 1 à 4)
    ├── models/
    │   ├── base_predictor.py           # Classe abstraite de base pour les prédicteurs
    │   ├── lightgbm_predictor.py       # Prédicteur GBDT LightGBM (déterministe et quantile q90)
    │   ├── lstm_predictor.py         # Prédicteur Réseau Récurrent PyTorch LSTM
    │   ├── nhits_predictor.py        # Prédicteur Hiérarchique PyTorch N-HiTS
    │   ├── prophet_predictor.py        # Prédicteur Prophet par couple (slice, station)
    │   ├── passthrough_predictor.py    # Baseline Oracle à information parfaite
    │   └── predictor_factory.py        # Usine (Factory) d'instanciation des prédicteurs
    ├── pipeline/
    │   ├── macro_ran.py                # Regroupement spatial K-Means des subnets en Macro-RANs
    │   └── trainer_evaluator.py        # Orchestrateur des boucles d'entraînement PPO et d'évaluation
    ├── simulator/
    │   └── ran_simulator.py            # Modèle physique énergétique et calcul de la QoS 5G/6G
    └── visualization/
        └── plot_generator.py         # Génération automatique des graphiques et timelines
```

---

## ⚙️ Organisation et Fonctionnement du Code

1. **Prédicteurs de Trafic (`src/models/`)** :
   - Reçoivent les historiques de trafic et génèrent la prédiction $\hat{l}^{t+1}$ pour le pas suivant.
   - Fournissent également des prédictions probabilistes (quantile $q90$) pour anticiper les pics de charge sans sous-dimensionnement.

2. **Environnement SDN (`src/environment/sdn_controller_env.py`)** :
   - Encapsule le moteur physique du RAN (`src/simulator/ran_simulator.py`).
   - Reçoit l'action d'extinction brute de l'agent PPO et applique le **filtre des seuils prédictifs** ($\alpha_{\text{seuil}} \cdot \text{Max}_i$) et le **veto de sécurité URLLC**.
   - Gère la redirection du trafic des tranches éteintes vers **EcoSlice 1** et **EcoSlice Bis**.

3. **Agent PPO (`src/agents/ppo_agent.py`)** :
   - Implémente l'algorithme *Proximal Policy Optimization* (Acteur-Critique en PyTorch).
   - En mode entraînement (`stochastic`), échantillonne selon une loi de Bernoulli.
   - En mode évaluation/test (`deterministic`), applique la décision stricte par seuil d'activation $p_i \ge 0.5$.

4. **Orchestrateur de Simulation (`src/pipeline/trainer_evaluator.py`)** :
   - Divise le dataset en jeu d'entraînement et jeu de test (80/20).
   - Lance la boucle d'épisodes, collecte les trajectoires, entraîne l'agent PPO et génère la suite de graphiques et timelines d'évaluation.

---

## 🚀 Installation & Exécution (`uv`)

### 1. Prérequis et Installation
Le projet utilise le gestionnaire de paquets **`uv`** pour garantir la reproductibilité rapide des dépendances.

```bash
# Cloner le dépôt et se placer dans le dossier
cd dataset_creation

# Synchroniser l'environnement virtuel et installer les dépendances
uv sync
```

---

### 2. Exécution du Projet (`main.py`)

La simulation et les entraînements se lancent directement via la CLI de `main.py`.

#### Syntaxe générale :
```bash
uv run main.py [OPTIONS]
```

#### Options disponibles :

| Option | Description | Valeurs possibles | Valeur par défaut |
| :--- | :--- | :--- | :--- |
| `--model` | Modèle de prédiction à utiliser | `lightgbm`, `lstm`, `nhits`, `prophet`, `passthrough`, `all` | `lightgbm` |
| `--num_rans` | Nombre de Macro-RANs pour le découpage spatial | `0` (subnets bruts), `4` (4 Macro-RANs) | `4` |
| `--beta` | Poids de la satisfaction QoS dans la récompense | Flottant ($\ge 0$) | `5.0` |
| `--lambda_loss` | Pénalité de surcharge dans la récompense | Flottant ($\ge 0$) | `10.0` |
| `--episodes` | Nombre d'épisodes d'entraînement PPO | Entier ($\ge 1$) | `15` |
| `--eval_mode` | Mode d'évaluation lors de la phase de test | `deterministic`, `stochastic` | `deterministic` |
| `--seed` | Graine aléatoire pour la reproductibilité | Entier | `42` |

---

### 💡 Exemples d'Utilisation

```bash
# 1. Lancement standard avec LightGBM en régime d'équilibre (beta=5.0)
uv run main.py --model lightgbm --num_rans 4 --beta 5.0 --lambda_loss 10.0 --episodes 15

# 2. Lancement en régime favorisant l'économie d'énergie (beta=2.0)
uv run main.py --model lightgbm --num_rans 4 --beta 2.0 --lambda_loss 10.0 --episodes 15

# 3. Lancement d'un modèle neuronal (LSTM)
uv run main.py --model lstm --num_rans 4 --beta 5.0 --lambda_loss 10.0 --episodes 15

# 4. Benchmark complet (Exécution de tous les modèles à la suite)
uv run main.py --model all --num_rans 4 --beta 5.0 --lambda_loss 10.0 --episodes 15
```

Les graphiques, logs et fichiers d'évaluation sont automatiquement enregistrés dans le dossier `./data/plots/<model_name>/` et archivés dans `./data/experiments_v12/`.
