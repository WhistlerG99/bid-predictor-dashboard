"""Shared UI elements and callbacks for flight selection helpers."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from uuid import uuid4

import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate


def build_random_selection_button(button_id: str) -> html.Div:
    """Render a styled random selection button used across tabs."""

    return html.Div(
        [
            html.Button(
                "Random flight",
                id=button_id,
                n_clicks=0,
                style={
                    "width": "100%",
                    "backgroundColor": "#1b4965",
                    "color": "white",
                    "border": "none",
                    "padding": "0.6rem",
                    "borderRadius": "6px",
                },
            ),
        ],
        style={"marginBottom": "0.75rem"},
    )


def build_selection_history_dropdown(dropdown_id: str) -> html.Div:
    """Render a dropdown for previously selected flights."""

    return html.Div(
        [
            html.Label("Selection history", style={"fontWeight": "600"}),
            dcc.Dropdown(
                id=dropdown_id,
                placeholder="Previously selected flights",
                options=[],
                value=None,
                style={"width": "100%"},
            ),
        ],
        style={"marginBottom": "0.75rem"},
    )


def register_selection_history_callbacks(
    app: Dash,
    *,
    dataset_store_id: str,
    random_button_id: str,
    history_dropdown_id: str,
    history_store_id: str,
    selection_request_store_id: str,
    carrier_dropdown_id: str,
    flight_dropdown_id: str,
    travel_date_dropdown_id: str,
    upgrade_dropdown_id: str,
    loader: Callable[[object], pd.DataFrame],
) -> None:
    """Share callbacks that handle random selection and history tracking."""

    @app.callback(
        Output(selection_request_store_id, "data"),
        Input(random_button_id, "n_clicks"),
        Input(history_dropdown_id, "value"),
        State(dataset_store_id, "data"),
        State(history_store_id, "data"),
        prevent_initial_call=True,
    )
    def handle_selection_requests(
        random_clicks: Optional[int],
        history_selection: Optional[str],
        dataset_config: Optional[object],
        history: Optional[List[Dict[str, str]]],
    ):
        """Dispatch random and history-based selection requests."""

        trigger = ctx.triggered_id
        if trigger == random_button_id:
            if not dataset_config:
                raise PreventUpdate

            dataset = loader(dataset_config)
            required = {
                "carrier_code",
                "flight_number",
                "travel_date",
                "upgrade_type",
            }
            if not required.issubset(dataset.columns):
                raise PreventUpdate

            candidates = dataset.dropna(subset=required)[list(required)].copy()
            if candidates.empty:
                raise PreventUpdate

            candidates["flight_number"] = candidates["flight_number"].astype(str)
            candidates["carrier_code"] = candidates["carrier_code"].astype(str)
            travel_dates = pd.to_datetime(candidates["travel_date"], errors="coerce")
            candidates = candidates.assign(travel_date=travel_dates)
            candidates = candidates.dropna()
            candidates["travel_date"] = candidates["travel_date"].dt.strftime("%Y-%m-%d")
            candidates = candidates.drop_duplicates()
            if candidates.empty:
                raise PreventUpdate

            selection = candidates.sample(n=1).iloc[0]
            return {
                "carrier": selection["carrier_code"],
                "flight_number": selection["flight_number"],
                "travel_date": selection["travel_date"],
                "upgrade_type": selection["upgrade_type"],
                "trigger": str(uuid4()),
            }

        if trigger == history_dropdown_id:
            if not history_selection:
                raise PreventUpdate

            history = history or []
            for entry in history:
                if entry.get("id") == history_selection:
                    return {
                        "carrier": entry["carrier"],
                        "flight_number": entry["flight_number"],
                        "travel_date": entry["travel_date"],
                        "upgrade_type": entry["upgrade_type"],
                        "trigger": str(uuid4()),
                    }
            raise PreventUpdate

        raise PreventUpdate

    @app.callback(
        Output(history_store_id, "data"),
        Output(history_dropdown_id, "options"),
        Input(carrier_dropdown_id, "value"),
        Input(flight_dropdown_id, "value"),
        Input(travel_date_dropdown_id, "value"),
        Input(upgrade_dropdown_id, "value"),
        Input(dataset_store_id, "data"),
        State(history_store_id, "data"),
        prevent_initial_call=True,
    )
    def update_selection_history(
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        dataset_config: Optional[object],
        history: Optional[List[Dict[str, str]]],
    ):
        """Record the most recent flight selection and update the history dropdown."""

        trigger = ctx.triggered_id
        if trigger == dataset_store_id:
            return [], []

        if not (carrier and flight_number and travel_date and upgrade_type):
            raise PreventUpdate

        history = history or []
        entry_id = "|".join(
            [
                str(carrier),
                str(flight_number),
                str(travel_date),
                str(upgrade_type),
            ]
        )
        label = f"{carrier} {flight_number} · {travel_date} · {upgrade_type}"
        new_entry = {
            "id": entry_id,
            "carrier": str(carrier),
            "flight_number": str(flight_number),
            "travel_date": str(travel_date),
            "upgrade_type": str(upgrade_type),
            "label": label,
        }

        filtered_history = [entry for entry in history if entry.get("id") != entry_id]
        filtered_history.insert(0, new_entry)
        filtered_history = filtered_history[:20]

        options = [
            {"label": entry["label"], "value": entry["id"]}
            for entry in filtered_history
        ]
        return filtered_history, options

    @app.callback(
        Output(history_dropdown_id, "value"),
        Input(dataset_store_id, "data"),
        prevent_initial_call=True,
    )
    def reset_selection_history_value(_: Optional[object]):
        """Clear the selection history dropdown when the dataset changes."""

        return None


__all__ = [
    "build_random_selection_button",
    "build_selection_history_dropdown",
    "register_selection_history_callbacks",
]
