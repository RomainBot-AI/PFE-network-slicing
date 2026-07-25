from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_ppo_module():
    path = Path(__file__).resolve().parents[1] / "simulation" / "ppo.py"
    spec = importlib.util.spec_from_file_location("simulation_ppo", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_forecast_demand_provider_combines_loads(tmp_path) -> None:
    ppo = _load_ppo_module()
    csv_path = tmp_path / "forecast.csv"
    csv_path.write_text(
        "\n".join(
            [
                "timestamp,slice,q10_mbps,q50_mbps,q90_mbps",
                "2024-01-01 00:00:00,URLLC,1,2,3",
                "2024-01-01 00:00:00,URLLC_eMBB_MIX,4,5,6",
                "2024-01-01 00:00:00,eMBB,7,8,9",
                "2024-01-01 00:00:00,mMTC,10,11,12",
            ]
        ),
        encoding="utf-8",
    )
    provider = ppo.ForecastDemandProvider(
        str(csv_path),
        ("URLLC", "URLLC_eMBB_MIX", "eMBB", "mMTC"),
        quantile="q90",
        loop=False,
    )

    combined, forecast = provider.combine(np.array([5.0, 1.0, 20.0, 0.0]), mode="max")

    np.testing.assert_allclose(forecast, np.array([3.0, 6.0, 9.0, 12.0]))
    np.testing.assert_allclose(combined, np.array([5.0, 6.0, 20.0, 12.0]))
