import pandas as pd

from bid_predictor_ui.snapshots import build_snapshot_options


def test_build_snapshot_options_formats_and_sorts_numeric_values():
    series = pd.Series([3, None, 1, "2", 3, "10"])

    options = build_snapshot_options(series)

    assert options == [
        {"label": "Snapshot 1", "value": "1"},
        {"label": "Snapshot 2", "value": "2"},
        {"label": "Snapshot 3", "value": "3"},
        {"label": "Snapshot 10", "value": "10"},
    ]


def test_build_snapshot_options_handles_non_numeric_entries():
    snapshots = ["A", "B", "A"]

    options = build_snapshot_options(snapshots)

    assert options == [
        {"label": "Snapshot A", "value": "A"},
        {"label": "Snapshot B", "value": "B"},
    ]
