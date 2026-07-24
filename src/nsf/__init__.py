"""Network-slice forecasting package.

The package is a clean research-facing layer around the existing
``traffic_forecasting`` scripts. Old script entry points remain supported while
new code moves here module by module.
"""

from nsf.constants import SLICES

__all__ = ["SLICES"]
