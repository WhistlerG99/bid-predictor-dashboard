import pandas as pd

from bid_predictor_ui.performance_history import data as history_data


def _build_sample_dataset():
    return pd.DataFrame(
        {
            "accept_prob_timestamp": [
                "2024-01-01T05:00:00",
                "2024-01-01T06:00:00",
                "2024-01-02T07:00:00",
            ],
            "offer_status": ["TICKETED", "EXPIRED", "TICKETED"],
            "accept_prob": [0.9, 0.2, 0.4],
            "carrier_code": ["AA", "AA", "BB"],
        }
    )


def test_compute_daily_performance_history_includes_overall_and_carrier_rows():
    dataset = _build_sample_dataset()

    history = history_data.compute_daily_performance_history(dataset, threshold=0.5)

    assert not history.empty
    assert set(history["carrier_code"]) == {"AA", "BB", "ALL"}

    aa_day = history[
        (history["carrier_code"] == "AA")
        & (history[history_data.HISTORY_DATE_COLUMN] == pd.Timestamp("2024-01-01"))
    ].iloc[0]
    assert aa_day["total"] == 2
    assert aa_day["tp"] == 1
    assert aa_day["tn"] == 1
    assert aa_day["accuracy"] == 1.0


def test_update_performance_history_refreshes_recent_dates(tmp_path):
    dataset = _build_sample_dataset()
    history_path = tmp_path / "history.parquet"
    source_path = tmp_path / "source.parquet"

    initial = history_data.compute_daily_performance_history(dataset.iloc[:1], threshold=0.5)
    initial.to_parquet(history_path, index=False)
    dataset.to_parquet(source_path, index=False)

    updated = history_data.update_performance_history_from_source(
        str(history_path),
        str(source_path),
        refresh_days=2,
        threshold=0.5,
    )

    assert updated is not None
    assert updated[history_data.HISTORY_DATE_COLUMN].nunique() == 2
    assert (updated[history_data.HISTORY_DATE_COLUMN] == pd.Timestamp("2024-01-02")).any()
