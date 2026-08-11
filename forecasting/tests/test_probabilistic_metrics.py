from __future__ import annotations

import numpy as np

from nsf.evaluation.probabilistic import (
    interval_coverage,
    interval_score,
    interval_width,
    normalized_interval_width,
    pinball_loss,
)


def test_pinball_loss_zero_for_exact_quantile_prediction() -> None:
    y = np.array([0.0, 10.0, 20.0])

    assert pinball_loss(y, y, 0.1) == 0.0
    assert pinball_loss(y, y, 0.5) == 0.0
    assert pinball_loss(y, y, 0.9) == 0.0


def test_interval_metrics() -> None:
    y = np.array([10.0, 20.0, 30.0, 40.0])
    lower = np.array([5.0, 15.0, 35.0, 35.0])
    upper = np.array([15.0, 25.0, 45.0, 45.0])

    assert interval_coverage(y, lower, upper) == 0.75
    assert interval_width(lower, upper) == 10.0
    assert normalized_interval_width(y, lower, upper) == 0.4
    assert interval_score(y, lower, upper, alpha=0.2) > interval_width(lower, upper)
