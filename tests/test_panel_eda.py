import pandas as pd

from nsf.eda.panel import concentration_by_slice, overview, summarize_by_series, summarize_by_slice


def test_panel_eda_summarizes_subnet_slice_panel():
    df = pd.DataFrame(
        {
            "unique_id": ["subnet_1__URLLC"] * 3 + ["subnet_2__eMBB"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3, freq="10min").tolist() * 2,
            "y": [1.0, 0.0, 3.0, 10.0, 20.0, 0.0],
            "slice": ["URLLC"] * 3 + ["eMBB"] * 3,
            "id_institution": [1] * 3 + [2] * 3,
            "id_institution_subnet": [1] * 3 + [2] * 3,
        }
    )

    info = overview(df, freq="10min")
    series = summarize_by_series(df, freq="10min")
    slices = summarize_by_slice(series)
    concentration = concentration_by_slice(series)

    assert info["series"] == 2
    assert info["missing_global_timestamps"] == 0
    assert set(series["zero_share"].round(2)) == {0.33}
    assert set(slices["slice"]) == {"URLLC", "eMBB"}
    assert concentration["traffic_share"].max() == 1.0
