"""Helpers for interacting with the MLflow model registry."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import mlflow


def list_registered_model_names() -> List[str]:
    """Return sorted registered model names from MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()
    except Exception:
        return []

    names = {model.name for model in models if getattr(model, "name", None)}
    return sorted(names)


def list_model_stage_or_versions(model_name: str) -> Tuple[List[str], List[str]]:
    """Return available stages and versions for a registered model."""
    if not model_name:
        return [], []

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return [], []

    stages = {
        version.current_stage
        for version in versions
        if getattr(version, "current_stage", None)
        and str(version.current_stage).lower() != "none"
    }
    version_numbers = {
        str(version.version)
        for version in versions
        if getattr(version, "version", None)
    }

    return _sort_stages(stages), _sort_versions(version_numbers)


def build_model_name_options() -> List[dict]:
    """Return dropdown-friendly options for registered models."""
    return [{"label": name, "value": name} for name in list_registered_model_names()]


def build_model_stage_or_version_options(model_name: str) -> List[dict]:
    """Return dropdown options for stages and versions of a model."""
    stages, versions = list_model_stage_or_versions(model_name)
    options = [{"label": stage, "value": stage} for stage in stages]
    options.extend(
        {"label": f"Version {version}", "value": version} for version in versions
    )
    return options


def _sort_stages(stages: Iterable[str]) -> List[str]:
    return sorted(stages, key=lambda stage: stage.lower())


def _sort_versions(versions: Iterable[str]) -> List[str]:
    def _version_key(value: str) -> Tuple[int, str]:
        if value.isdigit():
            return (0, f"{int(value):06d}")
        return (1, value)

    return sorted(versions, key=_version_key)
