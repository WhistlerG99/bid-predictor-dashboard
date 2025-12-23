from types import SimpleNamespace

from bid_predictor_ui import model_registry


def test_list_registered_model_names(monkeypatch):
    class FakeClient:
        def search_registered_models(self):
            return [
                SimpleNamespace(name="alpha"),
                SimpleNamespace(name="beta"),
                SimpleNamespace(name="alpha"),
                SimpleNamespace(name=None),
            ]

    monkeypatch.setattr(
        model_registry.mlflow.tracking,
        "MlflowClient",
        lambda: FakeClient(),
    )

    assert model_registry.list_registered_model_names() == ["alpha", "beta"]


def test_list_model_stage_or_versions(monkeypatch):
    class FakeClient:
        def search_model_versions(self, query):
            assert query == "name='example'"
            return [
                SimpleNamespace(current_stage="Production", version="2"),
                SimpleNamespace(current_stage="Staging", version=1),
                SimpleNamespace(current_stage="None", version=3),
                SimpleNamespace(current_stage=None, version=None),
            ]

    monkeypatch.setattr(
        model_registry.mlflow.tracking,
        "MlflowClient",
        lambda: FakeClient(),
    )

    stages, versions = model_registry.list_model_stage_or_versions("example")
    assert stages == ["Production", "Staging"]
    assert versions == ["1", "2", "3"]


def test_build_model_stage_or_version_options(monkeypatch):
    monkeypatch.setattr(
        model_registry,
        "list_model_stage_or_versions",
        lambda model_name: (["Production"], ["7"]),
    )

    assert model_registry.build_model_stage_or_version_options("example") == [
        {"label": "Production", "value": "Production"},
        {"label": "Version 7", "value": "7"},
    ]
