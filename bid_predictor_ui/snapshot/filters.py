"""Dropdown population callbacks for the snapshot explorer."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import pandas as pd
from dash import Dash, Input, Output, State
from dash.exceptions import PreventUpdate

from ..data import load_dashboard_dataset
from ..dropdowns import choose_dropdown_value, options_from_series
from ..selection_controls import register_selection_history_callbacks
from ..snapshots import build_snapshot_options


def register_filter_callbacks(app: Dash) -> None:
    """Register callbacks that populate the snapshot filter dropdowns."""

    @app.callback(
        Output("carrier-dropdown", "options"),
        Output("carrier-dropdown", "value"),
        Input("dataset-path-store", "data"),
        Input("snapshot-selection-request-store", "data"),
        State("carrier-dropdown", "value"),
    )
    def populate_carriers(
        dataset_path: Optional[Mapping[str, object] | str],
        selection_request: Optional[Dict[str, str]],
        current_value: Optional[str],
    ):
        """Populate the carrier dropdown when a dataset is loaded."""

        if not dataset_path:
            return [], None

        dataset = load_dashboard_dataset(dataset_path)
        if "carrier_code" not in dataset.columns:
            return [], None

        carriers = dataset["carrier_code"].dropna().drop_duplicates().sort_values()
        options = options_from_series(carriers)
        requested_carrier = None
        if selection_request:
            requested_carrier = selection_request.get("carrier")

        value = choose_dropdown_value(options, requested_carrier, current_value)
        return options, value

    @app.callback(
        Output("flight-number-dropdown", "options"),
        Output("flight-number-dropdown", "value"),
        Input("carrier-dropdown", "value"),
        Input("snapshot-selection-request-store", "data"),
        State("dataset-path-store", "data"),
        State("flight-number-dropdown", "value"),
    )
    def populate_flight_numbers(
        carrier: Optional[str],
        selection_request: Optional[Dict[str, str]],
        dataset_path: Optional[Mapping[str, object] | str],
        current_value: Optional[str],
    ):
        """Filter flight numbers based on the selected carrier."""

        if not dataset_path or not carrier:
            return [], None

        dataset = load_dashboard_dataset(dataset_path)
        if not {"carrier_code", "flight_number"}.issubset(dataset.columns):
            return [], None

        mask = dataset["carrier_code"] == carrier
        flights = (
            dataset.loc[mask, "flight_number"].dropna().drop_duplicates().sort_values()
        )
        options = [{"label": str(number), "value": str(number)} for number in flights]
        requested_flight = None
        if selection_request:
            requested_flight = selection_request.get("flight_number")
            if requested_flight is not None:
                requested_flight = str(requested_flight)

        current_value = str(current_value) if current_value is not None else None
        value = choose_dropdown_value(options, requested_flight, current_value)
        return options, value

    @app.callback(
        Output("travel-date-dropdown", "options"),
        Output("travel-date-dropdown", "value"),
        Input("flight-number-dropdown", "value"),
        Input("snapshot-selection-request-store", "data"),
        State("carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("travel-date-dropdown", "value"),
    )
    def populate_travel_dates(
        flight_number: Optional[str],
        selection_request: Optional[Dict[str, str]],
        carrier: Optional[str],
        dataset_path: Optional[Mapping[str, object] | str],
        current_value: Optional[str],
    ):
        """Filter travel dates once the flight number has been chosen."""

        if not dataset_path or not carrier or not flight_number:
            return [], None

        dataset = load_dashboard_dataset(dataset_path)
        required = {"carrier_code", "flight_number", "travel_date"}
        if not required.issubset(dataset.columns):
            return [], None

        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
        )
        travel_dates = (
            pd.to_datetime(dataset.loc[mask, "travel_date"], errors="coerce")
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        options = options_from_series(
            pd.Series([date.strftime("%Y-%m-%d") for date in travel_dates])
        )
        requested_date = None
        if selection_request:
            requested_date = selection_request.get("travel_date")

        value = choose_dropdown_value(options, requested_date, current_value)
        return options, value

    @app.callback(
        Output("upgrade-dropdown", "options"),
        Output("upgrade-dropdown", "value"),
        Input("travel-date-dropdown", "value"),
        Input("snapshot-selection-request-store", "data"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("upgrade-dropdown", "value"),
    )
    def populate_upgrade_types(
        travel_date: Optional[str],
        selection_request: Optional[Dict[str, str]],
        carrier: Optional[str],
        flight_number: Optional[str],
        dataset_path: Optional[Mapping[str, object] | str],
        current_value: Optional[str],
    ):
        """List upgrade types for the fully specified flight selection."""

        if not dataset_path or not carrier or not flight_number or not travel_date:
            return [], None

        dataset = load_dashboard_dataset(dataset_path)
        if "upgrade_type" not in dataset.columns:
            return [], None

        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
        )
        upgrades = (
            dataset.loc[mask, "upgrade_type"].dropna().drop_duplicates().sort_values()
        )
        options = options_from_series(upgrades)
        requested_upgrade = None
        if selection_request:
            requested_upgrade = selection_request.get("upgrade_type")

        value = choose_dropdown_value(options, requested_upgrade, current_value)
        return options, value

    @app.callback(
        Output("snapshot-dropdown", "options"),
        Output("snapshot-dropdown", "value"),
        Input("upgrade-dropdown", "value"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("travel-date-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_snapshots(
        upgrade_type: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        dataset_path: Optional[Mapping[str, object] | str],
    ):
        """List available snapshots after the user selects an upgrade type."""

        if (
            not dataset_path
            or not carrier
            or not flight_number
            or not travel_date
            or not upgrade_type
        ):
            return [], None

        dataset = load_dashboard_dataset(dataset_path)
        if "snapshot_num" not in dataset.columns:
            return [], None

        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
            & (dataset["upgrade_type"] == upgrade_type)
        )
        snapshots = dataset.loc[mask, "snapshot_num"]
        options = build_snapshot_options(snapshots)
        value = options[0]["value"] if options else None
        return options, value

    register_selection_history_callbacks(
        app,
        dataset_store_id="dataset-path-store",
        random_button_id="random-selection-button",
        history_dropdown_id="selection-history-dropdown",
        history_store_id="selection-history-store",
        selection_request_store_id="snapshot-selection-request-store",
        carrier_dropdown_id="carrier-dropdown",
        flight_dropdown_id="flight-number-dropdown",
        travel_date_dropdown_id="travel-date-dropdown",
        upgrade_dropdown_id="upgrade-dropdown",
        loader=load_dashboard_dataset,
    )


__all__ = ["register_filter_callbacks"]
