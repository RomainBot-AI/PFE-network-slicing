import numpy as np
import pandas as pd

from nsf.preprocessing.scaler import fit_log_zscore, transform_log_zscore


def test_log_zscore_uses_train_statistics_only():
    train = pd.Series([1.0, 2.0, 3.0])
    test = pd.Series([1_000_000.0])

    params = fit_log_zscore(train)
    transformed_train = transform_log_zscore(train, params)
    transformed_test = transform_log_zscore(test, params)

    expected_mean = np.log1p(train).mean()
    assert params.mean == expected_mean
    assert abs(float(transformed_train.mean())) < 1e-12
    assert transformed_test[0] > 10.0
