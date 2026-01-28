import pandas as pd

from bid_predictor_ui.acceptance_explorer import view


def test_build_download_dataframe_orders_and_renames_columns():
    table_data = [
        {"Feature": "offer_id", "Bid 1": 101, "Bid 2": 202},
        {"Feature": "Acceptance Probability", "Bid 1": 0.25, "Bid 2": 0.5},
    ]
    columns = [
        {"name": "Feature", "id": "Feature"},
        {"name": "Bid 1", "id": "Bid 1"},
        {"name": "Bid 2", "id": "Bid 2"},
    ]

    df = view._build_download_dataframe(table_data, columns)

    expected = pd.DataFrame(table_data)
    expected = expected[["Feature", "Bid 1", "Bid 2"]]
    pd.testing.assert_frame_equal(df, expected)


def test_build_download_dataframe_handles_empty_inputs():
    df = view._build_download_dataframe([], [])

    assert df.empty
