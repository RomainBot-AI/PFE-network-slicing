"""Small IO helpers used by the forecasting package."""

from __future__ import annotations

from pathlib import Path


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
