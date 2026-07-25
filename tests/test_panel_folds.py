import pandas as pd

from nsf.splitting.panel_folds import leakage_audit, make_panel_folds


def test_panel_folds_are_chronological_and_non_overlapping():
    timestamps = pd.date_range("2024-01-01", periods=100, freq="10min")

    folds = make_panel_folds(
        timestamps=timestamps,
        input_size=20,
        horizon=5,
        n_folds=3,
        fold_stride=10,
        expanding=False,
    )
    audit = leakage_audit(folds)

    assert [fold.fold for fold in folds] == [0, 1, 2]
    assert audit["train_ends_before_target"].all()
    assert audit["train_target_overlap_points"].sum() == 0
    assert set(audit["input_size"]) == {20}
    assert set(audit["horizon"]) == {5}


def test_expanding_panel_folds_keep_past_only():
    timestamps = pd.date_range("2024-01-01", periods=100, freq="10min")

    folds = make_panel_folds(
        timestamps=timestamps,
        input_size=20,
        horizon=5,
        n_folds=2,
        fold_stride=10,
        expanding=True,
    )

    assert folds[0].train_start_idx == 0
    assert folds[1].train_start_idx == 0
    assert folds[0].train_end_idx < folds[0].target_start_idx
