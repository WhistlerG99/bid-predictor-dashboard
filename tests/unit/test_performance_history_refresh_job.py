from bid_predictor_ui.performance_history import refresh_job


def test_run_performance_history_refresh_calls_update(monkeypatch):
    called = {}

    def fake_update(history_uri, source_uri, *, refresh_days, cache_client=None, **_):
        called["history_uri"] = history_uri
        called["source_uri"] = source_uri
        called["refresh_days"] = refresh_days
        called["cache_client"] = cache_client
        return None

    monkeypatch.setattr(
        refresh_job, "update_performance_history_from_source", fake_update
    )

    result = refresh_job.run_performance_history_refresh(
        "history.parquet", "source/path", 7
    )

    assert result is True
    assert called == {
        "history_uri": "history.parquet",
        "source_uri": "source/path",
        "refresh_days": 7,
        "cache_client": None,
    }


def test_run_performance_history_refresh_requires_configuration(monkeypatch):
    called = {"count": 0}

    def fake_update(*args, **kwargs):
        called["count"] += 1
        return None

    monkeypatch.setattr(
        refresh_job, "update_performance_history_from_source", fake_update
    )

    result = refresh_job.run_performance_history_refresh("", "source/path", 7)

    assert result is False
    assert called["count"] == 0
