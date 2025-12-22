"""MLflow model loader callbacks for the snapshot explorer."""
from __future__ import annotations

from typing import Mapping, Optional

from dash import Dash, Input, Output, State, no_update

from ..data import get_model_feature_config, load_model_cached
from ..feature_config import build_ui_feature_config


def register_model_loader_callbacks(app: Dash) -> None:
    """Register callbacks that load MLflow models for predictions."""

    @app.callback(
        Output("model-uri-store", "data"),
        Output("feature-config-store", "data"),
        Output("mlflow-model-status", "children"),
        Input("mlflow-model-load", "n_clicks"),
        State("mlflow-model-uri-input", "value"),
        prevent_initial_call=True,
    )
    def load_model(
        n_clicks: int,
        model_uri: Optional[str],
    ) -> tuple[object, object, str]:
        if not n_clicks:
            return no_update, no_update, ""

        if not model_uri:
            return no_update, no_update, "Enter an MLflow model URI before loading."

        try:
            load_model_cached(model_uri)
        except Exception as exc:  # pragma: no cover - user feedback
            return no_update, no_update, f"Failed to load model: {exc}"

        raw_config: Optional[Mapping[str, object]] = get_model_feature_config(model_uri)
        ui_config = build_ui_feature_config(raw_config)
        return model_uri, ui_config, f"Loaded model: {model_uri}"


__all__ = ["register_model_loader_callbacks"]
