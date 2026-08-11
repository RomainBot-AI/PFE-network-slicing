import numpy as np

from nsf.datasets.windowing import make_supervised_windows
from nsf.splitting.rolling_origin import make_rolling_windows


def test_make_supervised_windows_keeps_targets_after_inputs():
    x, y = make_supervised_windows(np.arange(10), input_size=3, horizon=2, step_size=2)

    assert x.tolist() == [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]
    assert y.tolist() == [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]


def test_rolling_windows_are_chronological_and_non_leaking():
    windows = make_rolling_windows(total_points=10, input_size=3, horizon=2, step_size=2)

    assert len(windows) == 3
    for window in windows:
        assert window.input_end_idx < window.target_start_idx
        assert window.target_end_idx - window.target_start_idx + 1 == 2
