"""Callbacks and helpers for the acceptance probability explorer tab."""
from __future__ import annotations

from copy import deepcopy
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from dash import Dash, Input, Output, State, html, no_update
from dotenv import load_dotenv

from ..dropdowns import choose_dropdown_value, options_from_series
from ..snapshots import build_snapshot_options
from ..data_sources import (
    DEFAULT_ACCEPTANCE_TABLE,
    load_dataset_from_source,
)
from ..feature_config import DEFAULT_UI_FEATURE_CONFIG
from ..formatting import prepare_bid_record
from ..plotting import build_prediction_plot, filter_snapshots_by_frequency
from ..tables import build_bid_table
from ..selection_controls import register_selection_history_callbacks

load_dotenv()

_ACCEPTANCE_TABLE_FEATURES = [
    "offer_id",
    "conf_num",
    "days_before_departure",
    "hours_before_departure",
    "seats_available",
    "created_timestamp",
    "usd_base_amount_25%",
    "usd_base_amount_50%",
    "usd_base_amount_75%",
    "usd_base_amount_max",
    "num_offers",
    "bid_rank",
    "Current Time",
]


def _default_acceptance_config() -> Optional[Mapping[str, object]]:
    s3_uri = os.getenv("S3_DATASET_LISTING_URI")
    if not s3_uri:
        return None
    try:
        hours = int(os.getenv("S3_DATASET_LOOKBACK_HOURS", "120"))
    except (TypeError, ValueError):
        hours = 120
    if hours <= 0:
        hours = 120
    return {"source": "path", "path": s3_uri, "hours": hours}


def _acceptance_feature_config() -> Dict[str, List[str]]:
    config = deepcopy(DEFAULT_UI_FEATURE_CONFIG)
    priority_features = ["offer_id", "conf_num"]
    tail_features = ["hours_before_departure", "Current Time", "Acceptance Probability"]

    display_features: List[str] = []
    seen: set[str] = set()

    for feature in priority_features:
        if feature in seen:
            continue
        seen.add(feature)
        display_features.append(feature)

    base_candidates = list(config.get("display_features", [])) + _ACCEPTANCE_TABLE_FEATURES
    for feature in base_candidates:
        if feature in tail_features or feature in seen:
            continue
        seen.add(feature)
        display_features.append(feature)

    for feature in tail_features:
        if feature in seen:
            continue
        seen.add(feature)
        display_features.append(feature)
    config["display_features"] = display_features

    readonly = []
    seen_readonly: set[str] = set()
    for feature in list(config.get("readonly_features", [])) + _ACCEPTANCE_TABLE_FEATURES:
        if feature in seen_readonly:
            continue
        seen_readonly.add(feature)
        readonly.append(feature)
    config["readonly_features"] = readonly
    return config


def _scale_acceptance_probabilities(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    non_na = numeric.dropna()
    if not non_na.empty and non_na.max() <= 1:
        numeric = numeric * 100.0
    return numeric


def _normalize_acceptance_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        raise ValueError("The loaded dataset is empty.")

    base_amount_aliases = {
        "usd_base_amount_25_percent": "usd_base_amount_25%",
        "usd_base_amount_50_percent": "usd_base_amount_50%",
        "usd_base_amount_75_percent": "usd_base_amount_75%",
    }
    for alias, canonical in base_amount_aliases.items():
        if canonical not in dataset.columns and alias in dataset.columns:
            dataset[canonical] = dataset[alias]

    if "accept_prob_timestamp" in dataset.columns:
        timestamps = pd.to_datetime(dataset["accept_prob_timestamp"], errors="coerce")
        dataset["current_timestamp"] = timestamps
    if "travel_date" not in dataset.columns and "departure_timestamp" in dataset.columns:
        dataset["travel_date"] = pd.to_datetime(
            dataset["departure_timestamp"], errors="coerce"
        ).dt.date
    if "hours_before_departure" not in dataset.columns and "days_before_departure" in dataset.columns:
        days = pd.to_numeric(dataset["days_before_departure"], errors="coerce")
        dataset["hours_before_departure"] = days * 24

    group_keys = [
        key
        for key in [
            "carrier_code",
            "flight_number",
            "travel_date",
            "upgrade_type",
        ]
        if key in dataset.columns
    ]
    if "snapshot_num" not in dataset.columns and "current_timestamp" in dataset.columns:
        if group_keys:
            dataset["snapshot_num"] = (
                dataset.groupby(group_keys)["current_timestamp"]
                .rank(method='dense', ascending=True)
                .astype(int)
            )
        else:
            dataset["snapshot_num"] = (
                dataset["current_timestamp"]
                .rank(method='dense', ascending=True)
                .astype(int)
            )
    if "Bid #" not in dataset.columns and "offer_id" in dataset.columns:
        if group_keys:
            dataset["Bid #"] = pd.NA
            for _, group in dataset.groupby(group_keys, sort=False):
                offers = group["offer_id"]
                try:
                    ordered_offers = (
                        pd.Series(offers.dropna().unique())
                        .sort_values(kind="mergesort")
                        .tolist()
                    )
                except Exception:
                    ordered_offers = (
                        pd.Series(offers.dropna().astype(str).unique())
                        .sort_values(kind="mergesort")
                        .tolist()
                    )
                label_map = {value: idx + 1 for idx, value in enumerate(ordered_offers)}
                dataset.loc[group.index, "Bid #"] = offers.map(label_map)

            missing = dataset["Bid #"].isna()
            if missing.any():
                dataset.loc[missing, "Bid #"] = pd.factorize(dataset.loc[missing, "offer_id"])[0] + 1
            dataset["Bid #"] = dataset["Bid #"].astype(int)
        else:
            dataset["Bid #"] = pd.factorize(dataset["offer_id"])[0] + 1
    probability_source = None
    if "Acceptance Probability" in dataset.columns:
        probability_source = dataset["Acceptance Probability"]
    elif "acceptance_prob" in dataset.columns:
        probability_source = dataset["acceptance_prob"]
    elif "accept_prob" in dataset.columns:
        probability_source = dataset["accept_prob"]

    if probability_source is not None:
        scaled = _scale_acceptance_probabilities(probability_source)
        dataset["Acceptance Probability"] = scaled
        dataset["acceptance_prob"] = scaled
        dataset["accept_prob"] = scaled

    dataset["offer_status"] = dataset.get("offer_status", "pending")
    return dataset


def _select_first_series(
    dataset: pd.DataFrame, columns: Iterable[str]
) -> Optional[pd.Series]:
    for column in columns:
        if column in dataset.columns:
            return dataset[column]
    return None


def load_acceptance_dataset(
    config: str | Mapping[str, object], *, reload: bool = False
) -> pd.DataFrame:
    """Load acceptance data from a file path or Redshift table."""

    if not config:
        raise ValueError("No acceptance dataset source provided.")

    return load_dataset_from_source(
        config,
        normalizer=_normalize_acceptance_dataset,
        reload=reload,
    )


def _build_summary(
    carrier: Optional[str],
    flight_number: Optional[str],
    travel_date: Optional[str],
    upgrade: Optional[str],
) -> html.Div:
    return html.Ul(
        [
            html.Li(f"Carrier: {carrier}") if carrier else html.Li("Carrier not set"),
            html.Li(f"Flight: {flight_number}")
            if flight_number
            else html.Li("Flight not set"),
            html.Li(f"Travel date: {travel_date}")
            if travel_date
            else html.Li("Travel date not set"),
            html.Li(f"Upgrade type: {upgrade}")
            if upgrade
            else html.Li("Upgrade type not set"),
        ],
        style={"paddingLeft": "1.2rem", "margin": "0"},
    )


def _prepare_records(snapshot_df: pd.DataFrame) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for record in snapshot_df.to_dict("records"):
        cleaned = dict(record)
        if "Acceptance Probability" not in cleaned:
            cleaned["Acceptance Probability"] = cleaned.get("acceptance_prob") or cleaned.get(
                "accept_prob"
            )
        if "Current Time" not in cleaned:
            cleaned["Current Time"] = cleaned.get("current_timestamp") or cleaned.get(
                "accept_prob_timestamp"
            )
        current_value = cleaned.get("Current Time")
        if pd.isna(current_value):
            cleaned["Current Time"] = ""
        elif hasattr(current_value, "isoformat"):
            cleaned["Current Time"] = current_value.isoformat()
        records.append(prepare_bid_record(cleaned))
    return records


def _build_predictions(df: pd.DataFrame) -> Dict[str, object]:
    probability_map: Dict[str, object] = {}
    probabilities = _select_first_series(
        df, ["Acceptance Probability", "acceptance_prob", "accept_prob"]
    )
    if probabilities is None:
        probabilities = []
    for idx, value in enumerate(probabilities):
        probability_map[f"bid_{idx}"] = value
    return {"probabilities": probability_map, "derived_features": []}


def _render_table(
    records: Optional[List[Dict[str, object]]],
    predictions: Dict[str, object],
    feature_config: Optional[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    probability_map: Dict[str, object] = {}
    derived_values = None
    if isinstance(predictions, dict):
        if "probabilities" in predictions or "derived_features" in predictions:
            probability_map = dict(predictions.get("probabilities", {}))
            derived_values = predictions.get("derived_features")
        else:
            probability_map = dict(predictions)

    columns, data_rows, style_rules = build_bid_table(
        records,
        probability_map,
        feature_config=feature_config,
        derived_feature_values=derived_values,
        show_comp_features=True,
    )
    for column in columns:
        column["editable"] = False
    for column in columns:
        style_rules.append(
            {"if": {"column_id": column.get("id")}, "pointerEvents": "none"}
        )
    style_rules.append(
        {"if": {"row_index": "odd"}, "backgroundColor": "#f3f4f6"}
    )
    style_rules.append(
        {"if": {"row_index": "even"}, "backgroundColor": "#ffffff"}
    )
    return columns, data_rows, style_rules


def register_acceptance_callbacks(app: Dash) -> None:
    """Register callbacks for loading and viewing acceptance datasets."""

    @app.callback(
        Output("acceptance-carrier-dropdown", "options"),
        Output("acceptance-carrier-dropdown", "value"),
        Input("acceptance-dataset-path-store", "data"),
        Input("acceptance-selection-request-store", "data"),
        State("acceptance-carrier-dropdown", "value"),
    )
    def populate_carriers(
        dataset_config: Optional[object],
        selection_request: Optional[Dict[str, str]],
        current_value: Optional[str],
    ):
        dataset_config = dataset_config or _default_acceptance_config()
        if not dataset_config:
            return [], None
        dataset = load_acceptance_dataset(dataset_config)
        if "carrier_code" not in dataset.columns:
            return [], None
        options = options_from_series(dataset["carrier_code"])
        requested_carrier = None
        if selection_request:
            requested_carrier = selection_request.get("carrier")
        value = choose_dropdown_value(options, requested_carrier, current_value)
        return options, value

    @app.callback(
        Output("acceptance-flight-number-dropdown", "options"),
        Output("acceptance-flight-number-dropdown", "value"),
        Input("acceptance-carrier-dropdown", "value"),
        Input("acceptance-selection-request-store", "data"),
        State("acceptance-dataset-path-store", "data"),
        State("acceptance-flight-number-dropdown", "value"),
    )
    def populate_flights(
        carrier: Optional[str],
        selection_request: Optional[Dict[str, str]],
        dataset_config: Optional[object],
        current_value: Optional[str],
    ):
        dataset_config = dataset_config or _default_acceptance_config()
        if not dataset_config or not carrier:
            return [], None
        dataset = load_acceptance_dataset(dataset_config)
        if not {"carrier_code", "flight_number"}.issubset(dataset.columns):
            return [], None
        mask = dataset["carrier_code"] == carrier
        options = options_from_series(dataset.loc[mask, "flight_number"].astype(str))
        requested_flight = None
        if selection_request:
            requested_flight = selection_request.get("flight_number")
            if requested_flight is not None:
                requested_flight = str(requested_flight)
        current_value = str(current_value) if current_value is not None else None
        value = choose_dropdown_value(options, requested_flight, current_value)
        return options, value

    @app.callback(
        Output("acceptance-travel-date-dropdown", "options"),
        Output("acceptance-travel-date-dropdown", "value"),
        Input("acceptance-flight-number-dropdown", "value"),
        Input("acceptance-selection-request-store", "data"),
        State("acceptance-carrier-dropdown", "value"),
        State("acceptance-dataset-path-store", "data"),
        State("acceptance-travel-date-dropdown", "value"),
    )
    def populate_travel_dates(
        flight_number: Optional[str],
        selection_request: Optional[Dict[str, str]],
        carrier: Optional[str],
        dataset_config: Optional[object],
        current_value: Optional[str],
    ):
        dataset_config = dataset_config or _default_acceptance_config()
        if not dataset_config or not carrier or not flight_number:
            return [], None
        dataset = load_acceptance_dataset(dataset_config)
        if "travel_date" not in dataset.columns:
            return [], None
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
        )
        travel_dates = pd.to_datetime(dataset.loc[mask, "travel_date"], errors="coerce")
        options = options_from_series(
            pd.Series(
                [
                    date.strftime("%Y-%m-%d")
                    for date in travel_dates.dropna().drop_duplicates().sort_values()
                ]
            )
        )
        requested_travel_date = None
        if selection_request:
            requested_travel_date = selection_request.get("travel_date")
        value = choose_dropdown_value(options, requested_travel_date, current_value)
        return options, value

    @app.callback(
        Output("acceptance-upgrade-dropdown", "options"),
        Output("acceptance-upgrade-dropdown", "value"),
        Input("acceptance-travel-date-dropdown", "value"),
        Input("acceptance-selection-request-store", "data"),
        State("acceptance-carrier-dropdown", "value"),
        State("acceptance-flight-number-dropdown", "value"),
        State("acceptance-dataset-path-store", "data"),
        State("acceptance-upgrade-dropdown", "value"),
    )
    def populate_upgrades(
        travel_date: Optional[str],
        selection_request: Optional[Dict[str, str]],
        carrier: Optional[str],
        flight_number: Optional[str],
        dataset_config: Optional[object],
        current_value: Optional[str],
    ):
        dataset_config = dataset_config or _default_acceptance_config()
        if not dataset_config or not carrier or not flight_number or not travel_date:
            return [], None
        dataset = load_acceptance_dataset(dataset_config)
        if "upgrade_type" not in dataset.columns:
            return [], None
        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
        )
        options = options_from_series(dataset.loc[mask, "upgrade_type"])
        requested_upgrade = None
        if selection_request:
            requested_upgrade = selection_request.get("upgrade_type")
        value = choose_dropdown_value(options, requested_upgrade, current_value)
        return options, value

    @app.callback(
        Output("acceptance-snapshot-dropdown", "options"),
        Output("acceptance-snapshot-dropdown", "value"),
        Input("acceptance-upgrade-dropdown", "value"),
        State("acceptance-carrier-dropdown", "value"),
        State("acceptance-flight-number-dropdown", "value"),
        State("acceptance-travel-date-dropdown", "value"),
        State("acceptance-dataset-path-store", "data"),
    )
    def populate_snapshots(
        upgrade: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        dataset_config: Optional[object],
    ):
        dataset_config = dataset_config or _default_acceptance_config()
        if not dataset_config or not carrier or not flight_number or not travel_date or not upgrade:
            return [], None
        dataset = load_acceptance_dataset(dataset_config)
        if "snapshot_num" not in dataset.columns:
            return [], None
        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
            & (dataset["upgrade_type"] == upgrade)
        )
        snapshots = dataset.loc[mask, "snapshot_num"]
        options = build_snapshot_options(snapshots)
        return options, options[0]["value"] if options else None

    @app.callback(
        Output("acceptance-departure-timestamp", "children"),
        Output("acceptance-origination-code", "children"),
        Output("acceptance-destination-code", "children"),
        Input("acceptance-dataset-path-store", "data"),
        Input("acceptance-carrier-dropdown", "value"),
        Input("acceptance-flight-number-dropdown", "value"),
        Input("acceptance-travel-date-dropdown", "value"),
        Input("acceptance-upgrade-dropdown", "value"),
    )
    def populate_route_details(
        dataset_config: Optional[object],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade: Optional[str],
    ):
        dataset_config = dataset_config or _default_acceptance_config()
        if not dataset_config or not carrier or not flight_number:
            return "–", "–", "–"

        dataset = load_acceptance_dataset(dataset_config)
        if not {"carrier_code", "flight_number"}.issubset(dataset.columns):
            return "–", "–", "–"

        mask = (dataset["carrier_code"] == carrier) & (
            dataset["flight_number"].astype(str) == str(flight_number)
        )
        if travel_date and "travel_date" in dataset.columns:
            travel_date_dt = pd.to_datetime(travel_date).date()
            mask &= pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt
        if upgrade and "upgrade_type" in dataset.columns:
            mask &= dataset["upgrade_type"] == upgrade

        subset = dataset.loc[mask]
        if subset.empty:
            return "–", "–", "–"

        origin = subset.iloc[0].get("origination_code", "–")
        destination = subset.iloc[0].get("destination_code", "–")
        departure = subset.iloc[0].get("departure_timestamp", "–")
        if hasattr(departure, "isoformat"):
            departure = departure.isoformat()
        return departure or "–", origin or "–", destination or "–"

    @app.callback(
        Output("acceptance-flight-summary", "children"),
        Output("acceptance-prediction-graph", "figure"),
        Output("acceptance-warning", "children"),
        Output("acceptance-bid-table", "columns"),
        Output("acceptance-bid-table", "data"),
        Output("acceptance-bid-table", "style_data_conditional"),
        Output("acceptance-table-feedback", "children"),
        Input("acceptance-snapshot-dropdown", "value"),
        Input("acceptance-snapshot-frequency-dropdown", "value"),
        Input("acceptance-chart-style-radio", "value"),
        State("acceptance-carrier-dropdown", "value"),
        State("acceptance-flight-number-dropdown", "value"),
        State("acceptance-travel-date-dropdown", "value"),
        State("acceptance-upgrade-dropdown", "value"),
        State("acceptance-dataset-path-store", "data"),
    )
    def render_view(
        snapshot_value: Optional[str],
        snapshot_frequency: Optional[int],
        chart_style: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        dataset_config: Optional[object],
    ):
        summary = _build_summary(carrier, flight_number, travel_date, upgrade_type)
        dataset_config = dataset_config or _default_acceptance_config()
        if not dataset_config:
            return (
                summary,
                build_prediction_plot(pd.DataFrame(), chart_type=chart_style),
                "Load a dataset to begin.",
                no_update,
                [],
                [],
                "",
            )

        if not all([carrier, flight_number, travel_date, upgrade_type]):
            return (
                summary,
                build_prediction_plot(pd.DataFrame(), chart_type=chart_style),
                "",
                no_update,
                [],
                [],
                "Select a carrier, flight, travel date, and upgrade type.",
            )

        dataset = load_acceptance_dataset(dataset_config)
        required_columns = {
            "carrier_code",
            "flight_number",
            "travel_date",
            "upgrade_type",
        }
        if not required_columns.issubset(dataset.columns):
            empty_fig = build_prediction_plot(pd.DataFrame(), chart_type=chart_style)
            return (
                summary,
                empty_fig,
                "Dataset is missing required columns for this view.",
                no_update,
                [],
                [],
                "",
            )

        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
            & (dataset["upgrade_type"] == upgrade_type)
        )
        subset = dataset.loc[mask].copy()
        if subset.empty:
            empty_fig = build_prediction_plot(pd.DataFrame(), chart_type=chart_style)
            return summary, empty_fig, "No rows found for the selected flight.", no_update, [], [], ""

        subset["snapshot_num"] = subset.get("snapshot_num").astype(str)
        probability_series = _select_first_series(
            subset, ["Acceptance Probability", "acceptance_prob", "accept_prob"]
        )
        if probability_series is not None:
            subset["Acceptance Probability"] = probability_series
        if "Current Time" not in subset.columns:
            if "accept_prob_timestamp" in subset.columns:
                subset["Current Time"] = pd.to_datetime(
                    subset["accept_prob_timestamp"], errors="coerce"
                ).dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
            elif "current_timestamp" in subset.columns:
                subset["Current Time"] = pd.to_datetime(
                    subset["current_timestamp"], errors="coerce"
                ).dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
        graph_df = subset.copy()
        if "accept_prob_timestamp" in graph_df.columns and "current_timestamp" not in graph_df.columns:
            graph_df["current_timestamp"] = pd.to_datetime(
                graph_df["accept_prob_timestamp"], errors="coerce"
            )

        graph_df = filter_snapshots_by_frequency(
            graph_df, snapshot_frequency, priority_labels=[snapshot_value]
        )
        figure = build_prediction_plot(graph_df, chart_type=chart_style)
        warning = "" if snapshot_value else "Select a snapshot to view bid details."

        if not snapshot_value:
            return summary, figure, warning, no_update, [], [], ""

        snapshot_df = subset.loc[subset["snapshot_num"] == str(snapshot_value)].copy()
        snapshot_df = snapshot_df.reset_index(drop=True)
        if "Bid #" in snapshot_df.columns:
            snapshot_df = snapshot_df.sort_values("Bid #", kind="mergesort")
            snapshot_df = snapshot_df.reset_index(drop=True)
        if snapshot_df.empty:
            return summary, figure, "No rows found for the selected snapshot.", no_update, [], [], ""

        predictions = _build_predictions(snapshot_df)
        records = _prepare_records(snapshot_df)
        columns, data_rows, style_rules = _render_table(
            records, predictions, _acceptance_feature_config()
        )

        return summary, figure, warning, columns, data_rows, style_rules, ""

    register_selection_history_callbacks(
        app,
        dataset_store_id="acceptance-dataset-path-store",
        random_button_id="acceptance-random-selection-button",
        history_dropdown_id="acceptance-selection-history-dropdown",
        history_store_id="acceptance-selection-history-store",
        selection_request_store_id="acceptance-selection-request-store",
        carrier_dropdown_id="acceptance-carrier-dropdown",
        flight_dropdown_id="acceptance-flight-number-dropdown",
        travel_date_dropdown_id="acceptance-travel-date-dropdown",
        upgrade_dropdown_id="acceptance-upgrade-dropdown",
        loader=load_acceptance_dataset,
    )


__all__ = ["register_acceptance_callbacks", "load_acceptance_dataset"]
