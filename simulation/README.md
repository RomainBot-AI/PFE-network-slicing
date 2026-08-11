# SDN slicing simulation

This simulation uses:

- Ryu as the SDN controller and REST API.
- Mininet to emulate the topology and generate traffic.
- PPO to update the per-slice bandwidth allocation.

## Components

```text
controller.py   Ryu OpenFlow controller and REST API (ports, stats, energy)
mininet/topology.py   Mininet topology and tc slice classes (1:10..1:13)
ppo.py          centralised PPO agent that allocates per-slice bandwidth,
                enforces SLAs and slice priority, and optionally consumes a
                proactive demand forecast
forecast_inputs/   exported slice demand forecast consumed by ppo.py
docker-compose.yml, Dockerfile.mininet   Ryu + Mininet container
```

## Prerequisites

- Docker Engine.
- Docker Compose plugin.
- Python dependencies for the PPO agent:

```bash
python3 -m pip install -r simulation/requirements-ppo.txt
```

The clustered dataset must exist at:

```text
simulation/mininet/cesnet_points_clustered_4slices.csv
```

Generate it with:

```bash
python3 "Dataset Preparing/cluster_4_slices.py"
```

## Start Ryu and Mininet container

From `simulation/`:

```bash
docker compose up --build
```

Keep this terminal open.

## Start the Mininet topology

In another terminal, from `simulation/`:

```bash
docker exec -it mininet bash
python3 topology.py
```

## Start the PPO agent

In a third terminal, from the repository root:

```bash
source .venv/bin/activate
RYU_CONTROLLER_IP=172.18.0.10 RYU_REST_PORT=8080 python3 simulation/ppo.py
```

If the Docker bridge IP is not reachable from the host, use the published REST
port instead:

```bash
RYU_CONTROLLER_IP=127.0.0.1 RYU_REST_PORT=8080 python3 simulation/ppo.py
```

## Optional proactive forecast input

Export the retained LightGBM quantile forecast to a slice-level simulation
input:

```bash
PYTHONPATH=src python3 scripts/export_probabilistic_forecast_for_simulation.py
```

This creates:

```text
simulation/forecast_inputs/slice_demand_forecast_lightgbm_q90.csv
```

Run PPO with the conservative `q90` forecast:

```bash
FORECAST_CSV=simulation/forecast_inputs/slice_demand_forecast_lightgbm_q90.csv FORECAST_QUANTILE=q90 FORECAST_MODE=max RYU_CONTROLLER_IP=127.0.0.1 RYU_REST_PORT=8080 python3 simulation/ppo.py
```

Forecast modes:

- `max`: use `max(observed_load, forecast_load)` per slice. This is the
  recommended conservative mode.
- `forecast`: use forecast demand only.
- `add`: add observed and forecast demand.

## Useful checks

```bash
curl http://127.0.0.1:8080/getports
curl http://127.0.0.1:8080/getstats
curl http://127.0.0.1:8080/getenergy
```
