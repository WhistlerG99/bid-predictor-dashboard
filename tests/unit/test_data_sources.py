import pandas as pd

from bid_predictor_ui.data_sources import (
    clear_data_cache,
    load_dataset_from_source,
    load_dataset_with_offer_status,
)


def teardown_function(function):
    clear_data_cache()


def test_load_dataset_from_source_caches_and_reloads(tmp_path):
    data_path = tmp_path / "acceptance.csv"
    pd.DataFrame(
        {
            "accept_prob": [0.5],
            "accept_prob_timestamp": ["2024-01-01T00:00:00"],
            "offer_id": [1],
        }
    ).to_csv(data_path, index=False)

    normalizer_calls: list[int] = []

    def normalizer(frame: pd.DataFrame) -> pd.DataFrame:
        normalizer_calls.append(len(frame))
        return frame

    df_first = load_dataset_from_source(str(data_path), normalizer=normalizer)
    assert len(df_first) == 1

    df_cached = load_dataset_from_source(str(data_path), normalizer=normalizer)
    assert df_cached.equals(df_first)
    assert normalizer_calls == [1]

    pd.DataFrame(
        {
            "accept_prob": [0.5, 0.8],
            "accept_prob_timestamp": [
                "2024-01-01T00:00:00",
                "2024-01-02T00:00:00",
            ],
            "offer_id": [1, 2],
        }
    ).to_csv(data_path, index=False)

    df_reloaded = load_dataset_from_source(
        str(data_path), normalizer=normalizer, reload=True
    )

    assert len(df_reloaded) == 2
    assert normalizer_calls == [1, 2]


def test_load_dataset_with_offer_status_skips_enrichment_when_present(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "acceptance.csv"
    pd.DataFrame(
        {
            "accept_prob": [0.5],
            "offer_id": [1],
            "offer_status": ["TICKETED"],
        }
    ).to_csv(data_path, index=False)

    calls = {"count": 0}

    def fake_enrich(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        calls["count"] += 1
        return df

    monkeypatch.setattr(
        "bid_predictor_ui.data_sources.enrich_with_offer_status", fake_enrich
    )

    result = load_dataset_with_offer_status(str(data_path))

    assert "offer_status" in result.columns
    assert calls["count"] == 0


def test_load_dataset_with_offer_status_enriches_when_missing(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "acceptance.csv"
    pd.DataFrame(
        {
            "accept_prob": [0.5],
            "offer_id": [1],
        }
    ).to_csv(data_path, index=False)

    def fake_enrich(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        updated = df.copy()
        updated["offer_status"] = "pending"
        return updated

    monkeypatch.setattr(
        "bid_predictor_ui.data_sources.enrich_with_offer_status", fake_enrich
    )

    result = load_dataset_with_offer_status(str(data_path))

    assert result["offer_status"].tolist() == ["pending"]
