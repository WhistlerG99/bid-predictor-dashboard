"""Dropdown population callbacks for the feature sensitivity tab."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from dash import Dash, Input, Output, State
from dash.exceptions import PreventUpdate

from ..data import load_dataset_cached
from ..dropdowns import choose_dropdown_value
from ..scenario import (
    build_carrier_options,
    build_flight_number_options,
    build_travel_date_options,
    build_upgrade_options,
)
from ..selection_controls import register_selection_history_callbacks


def register_filter_callbacks(app: Dash) -> None:
    """Register callbacks that populate scenario selection dropdowns."""

    @app.callback(
        Output("scenario-carrier-dropdown", "options"),
        Output("scenario-carrier-dropdown", "value"),
        Input("dataset-path-store", "data"),
        Input("scenario-selection-request-store", "data"),
        State("scenario-carrier-dropdown", "value"),
    )
    def populate_scenario_carriers(
        dataset_path: Optional[str],
        selection_request: Optional[Dict[str, str]],
        current_value: Optional[str],
    ):
        """Extract the carriers present in the currently loaded dataset.

        When the dataset path changes the tab should present a fresh list of
        carriers.  Failures to read the file silently fall back to an empty
        dropdown, allowing the UI to surface a user-friendly warning elsewhere.
        """
        if not dataset_path:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_carrier_options(dataset)
        requested_carrier = None
        if selection_request:
            requested_carrier = selection_request.get("carrier")
        value = choose_dropdown_value(options, requested_carrier, current_value)
        return options, value

    @app.callback(
        Output("scenario-flight-number-dropdown", "options"),
        Output("scenario-flight-number-dropdown", "value"),
        Input("scenario-carrier-dropdown", "value"),
        Input("scenario-selection-request-store", "data"),
        State("dataset-path-store", "data"),
        State("scenario-flight-number-dropdown", "value"),
    )
    def populate_scenario_flight_numbers(
        carrier: Optional[str],
        selection_request: Optional[Dict[str, str]],
        dataset_path: Optional[str],
        current_value: Optional[str],
    ):
        """Populate flight numbers for the chosen carrier.

        Each time the carrier changes the callback reloads the dataset and
        returns all matching flight numbers so the next dropdown stays in sync
        with the selection hierarchy.
        """
        if not dataset_path or not carrier:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_flight_number_options(dataset, carrier)
        requested_flight = None
        if selection_request:
            requested_flight = selection_request.get("flight_number")
            if requested_flight is not None:
                requested_flight = str(requested_flight)
        current_value = str(current_value) if current_value is not None else None
        value = choose_dropdown_value(options, requested_flight, current_value)
        return options, value

    @app.callback(
        Output("scenario-travel-date-dropdown", "options"),
        Output("scenario-travel-date-dropdown", "value"),
        Input("scenario-flight-number-dropdown", "value"),
        Input("scenario-selection-request-store", "data"),
        State("scenario-carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("scenario-travel-date-dropdown", "value"),
    )
    def populate_scenario_travel_dates(
        flight_number: Optional[str],
        selection_request: Optional[Dict[str, str]],
        carrier: Optional[str],
        dataset_path: Optional[str],
        current_value: Optional[str],
    ):
        """Populate travel dates for the chosen carrier and flight.

        The available dates come directly from the dataset; invalid or missing
        data produces an empty list so the downstream controls reset
        gracefully.
        """
        if not dataset_path or not carrier or not flight_number:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_travel_date_options(dataset, carrier, flight_number)
        requested_date = None
        if selection_request:
            requested_date = selection_request.get("travel_date")
        value = choose_dropdown_value(options, requested_date, current_value)
        return options, value

    @app.callback(
        Output("scenario-upgrade-dropdown", "options"),
        Output("scenario-upgrade-dropdown", "value"),
        Input("scenario-travel-date-dropdown", "value"),
        Input("scenario-selection-request-store", "data"),
        State("scenario-carrier-dropdown", "value"),
        State("scenario-flight-number-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("scenario-upgrade-dropdown", "value"),
    )
    def populate_scenario_upgrades(
        travel_date: Optional[str],
        selection_request: Optional[Dict[str, str]],
        carrier: Optional[str],
        flight_number: Optional[str],
        dataset_path: Optional[str],
        current_value: Optional[str],
    ):
        """Populate upgrade types for the selected flight and travel date.

        By cascading all upstream filters into the dataset query the callback
        ensures that only valid upgrade types appear once a flight is fully
        specified.  Any data access errors again fall back to empty options so
        the app can communicate problems through a dedicated feedback area.
        """
        if not dataset_path or not carrier or not flight_number or not travel_date:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_upgrade_options(dataset, carrier, flight_number, travel_date)
        requested_upgrade = None
        if selection_request:
            requested_upgrade = selection_request.get("upgrade_type")
        value = choose_dropdown_value(options, requested_upgrade, current_value)
        return options, value

    register_selection_history_callbacks(
        app,
        dataset_store_id="dataset-path-store",
        random_button_id="scenario-random-selection-button",
        history_dropdown_id="scenario-selection-history-dropdown",
        history_store_id="scenario-selection-history-store",
        selection_request_store_id="scenario-selection-request-store",
        carrier_dropdown_id="scenario-carrier-dropdown",
        flight_dropdown_id="scenario-flight-number-dropdown",
        travel_date_dropdown_id="scenario-travel-date-dropdown",
        upgrade_dropdown_id="scenario-upgrade-dropdown",
        loader=load_dataset_cached,
    )


__all__ = ["register_filter_callbacks"]
