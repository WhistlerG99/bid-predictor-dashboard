import pandas as pd

from bid_predictor_ui.data_sources import clear_data_cache, load_dataset_from_source


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
