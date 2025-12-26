import pandas as pd

from bid_predictor_ui.performance_tracker import view


def _row_labels(row):
    return [tile.children[0].children for tile in row.children]


def _build_sample_overview_rows():
    dataset = pd.DataFrame(
        {
            "offer_status": ["TICKETED", "EXPIRED", "TICKETED", "EXPIRED"],
            "accept_prob": [90, 10, 40, 60],
            "carrier_code": ["AA", "AA", "AA", "AA"],
            "hours_before_departure": [5, 6, 7, 8],
        }
    )
    sections, error = view._performance_overview_tiles(
        dataset, threshold=0.5, carrier="ALL", hours_range=[0, 100]
    )
    assert error is None
    return sections


def test_performance_overview_counts_rows_match_requested_order():
    sections = _build_sample_overview_rows()
    counts_section = sections[0]
    summary_row = counts_section.children[1]
    detail_row = counts_section.children[2]

    assert _row_labels(summary_row) == [
        "Total Number of Items",
        "Number of Actual Positive",
        "Number of Actual Negatives",
    ]
    assert _row_labels(detail_row) == [
        "Number of True Positive",
        "Number of False Positive",
        "Number of True Negatives",
        "Number of False Negatives",
    ]


def test_performance_overview_metrics_rows_match_requested_order():
    sections = _build_sample_overview_rows()
    metrics_section = sections[1]
    metric_rows = metrics_section.children[1:]

    assert [_row_labels(row) for row in metric_rows] == [
        ["Accuracy", "Balanced Accuracy", "Prevalence"],
        ["F-Score", "FM Index", "Negative F-Score", "Negative FM Index"],
        ["Precision", "Recall", "False Negative Rate"],
        ["Negative Precision", "Negative Recall", "False Positive Rate"],
    ]
