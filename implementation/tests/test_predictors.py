"""Predictors produce one non-negative ``pred_<slice>`` column per slice."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.predictor_factory import get_traffic_predictor
from tests.conftest import SLICES

TABULAR_AND_SEQUENCE = ["passthrough", "ridge", "lightgbm", "lstm", "nhits"]


@pytest.mark.parametrize("model_name", TABULAR_AND_SEQUENCE)
def test_predictor_fit_predict(model_name, synthetic_pivoted):
    predictor = get_traffic_predictor(model_name)
    predictor.fit(synthetic_pivoted)
    out = predictor.predict_pivoted(synthetic_pivoted)

    assert list(predictor.slice_names) == SLICES
    for s in SLICES:
        assert f"pred_{s}" in out.columns
        preds = out[f"pred_{s}"].to_numpy()
        assert len(preds) == len(synthetic_pivoted)
        assert np.all(preds >= 0.0)
        assert np.all(np.isfinite(preds))


def test_prophet_optional(synthetic_pivoted):
    pytest.importorskip("prophet")
    predictor = get_traffic_predictor("prophet")
    predictor.fit(synthetic_pivoted)
    out = predictor.predict_pivoted(synthetic_pivoted)
    for s in SLICES:
        assert f"pred_{s}" in out.columns


def test_passthrough_is_oracle(synthetic_pivoted):
    metrics = get_traffic_predictor("passthrough").evaluate(synthetic_pivoted)
    assert metrics == {"MAE": 0.0, "RMSE": 0.0, "NMAE": 0.0}


def test_ridge_alias():
    from src.models.ridge_predictor import MLTrafficPredictor, RidgeTrafficPredictor

    assert MLTrafficPredictor is RidgeTrafficPredictor
