from bid_predictor_ui.performance_history import view


def test_resolve_performance_history_uri_prefers_explicit(monkeypatch):
    monkeypatch.setenv("PERFORMANCE_HISTORY_S3_URI", "s3://env/history.parquet")

    assert (
        view._resolve_performance_history_uri("s3://explicit/history.parquet")
        == "s3://explicit/history.parquet"
    )


def test_resolve_performance_history_uri_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("PERFORMANCE_HISTORY_S3_URI", "s3://env/history.parquet")

    assert (
        view._resolve_performance_history_uri(None) == "s3://env/history.parquet"
    )


def test_resolve_performance_history_uri_empty_when_missing(monkeypatch):
    monkeypatch.delenv("PERFORMANCE_HISTORY_S3_URI", raising=False)

    assert view._resolve_performance_history_uri(None) == ""
