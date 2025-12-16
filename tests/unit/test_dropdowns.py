import pandas as pd

from bid_predictor_ui.dropdowns import choose_dropdown_value, options_from_series


def test_options_from_series_deduplicates_and_sorts():
    series = pd.Series(["b", None, "a", "b", "c"])

    options = options_from_series(series)

    assert options == [
        {"label": "a", "value": "a"},
        {"label": "b", "value": "b"},
        {"label": "c", "value": "c"},
    ]


def test_choose_dropdown_value_prefers_requested_then_current():
    options = [
        {"label": "a", "value": "a"},
        {"label": "b", "value": "b"},
    ]

    assert choose_dropdown_value(options, "b", "a") == "b"
    assert choose_dropdown_value(options, None, "a") == "a"
    assert choose_dropdown_value(options, None, None) == "a"
    assert choose_dropdown_value([], None, None) is None
