"""Callbacks for performance history visualizations."""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dash import Dash, Input, Output, callback
from plotly import graph_objects as go

from .data import ALL_CARRIER_VALUE, HISTORY_DATE_COLUMN, load_performance_history
from .layout import COUNT_OPTIONS, METRIC_OPTIONS

COUNT_VALUE_TO_LABEL = {option["value"]: option["label"] for option in COUNT_OPTIONS}
METRIC_VALUE_TO_LABEL = {option["value"]: option["label"] for option in METRIC_OPTIONS}


def _resolve_performance_history_uri(history_uri: Optional[str]) -> str:
    if history_uri is not None:
        return history_uri
    return os.getenv("PERFORMANCE_HISTORY_S3_URI", "")


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        showarrow=False,
        xref="paper",
        yref="paper",
        font={"size": 14},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=380)
    return fig


def _apply_smoothing(
    history: pd.DataFrame, columns: Iterable[str], window: int
) -> pd.DataFrame:
    if history.empty or window <= 1:
        return history

    smoothed = history.copy()
    smoothed = smoothed.sort_values(HISTORY_DATE_COLUMN)
    for column in columns:
        if column not in smoothed.columns:
            continue
        smoothed[column] = smoothed[column].rolling(window=window, min_periods=1).mean()
    return smoothed


def _build_line_chart(
    history: pd.DataFrame,
    selected_columns: Iterable[str],
    label_map: Dict[str, str],
    y_title: str,
) -> go.Figure:
    selected = [col for col in selected_columns if col in history.columns]
    if not selected:
        return _empty_figure("Select at least one series to display.")

    fig = go.Figure()
    x_values = history[HISTORY_DATE_COLUMN]
    for column in selected:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=history[column],
                mode="lines+markers",
                name=label_map.get(column, column),
            )
        )

    fig.update_layout(
        margin={"l": 40, "r": 20, "t": 10, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(title_text="History date")
    fig.update_yaxes(title_text=y_title)
    return fig


def _extract_carrier_options(history: pd.DataFrame) -> List[Dict[str, str]]:
    options = [{"label": "All", "value": ALL_CARRIER_VALUE}]
    if history.empty or "carrier_code" not in history.columns:
        return options

    carriers = (
        history["carrier_code"].dropna().astype(str).sort_values().unique().tolist()
    )
    for carrier in carriers:
        if carrier == ALL_CARRIER_VALUE:
            continue
        options.append({"label": carrier, "value": carrier})
    return options


def register_performance_history_callbacks(
    app: Dash, history_uri: Optional[str] = None
) -> None:
    """Register callbacks powering the performance history tab."""
    resolved_history_uri = _resolve_performance_history_uri(history_uri)

    @callback(
        Output("performance-history-store", "data"),
        Output("performance-history-status", "children"),
        Input("acceptance-dataset-path-store", "data"),
    )
    def load_performance_history_data(
        _: Optional[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], str]:
        if not resolved_history_uri:
            return [], "PERFORMANCE_HISTORY_S3_URI is not configured."

        try:
            history = load_performance_history(resolved_history_uri)
        except Exception as exc:  # pragma: no cover - user feedback
            return [], f"Failed to load performance history: {exc}"

        if history.empty:
            return [], "No performance history data available yet."

        if HISTORY_DATE_COLUMN in history.columns:
            history[HISTORY_DATE_COLUMN] = pd.to_datetime(
                history[HISTORY_DATE_COLUMN], errors="coerce"
            )
            history[HISTORY_DATE_COLUMN] = history[HISTORY_DATE_COLUMN].dt.strftime(
                "%Y-%m-%d"
            )

        return history.to_dict("records"), ""

    @callback(
        Output("performance-history-carrier", "options"),
        Output("performance-history-carrier", "value"),
        Input("performance-history-store", "data"),
    )
    def populate_history_carriers(
        records: Optional[List[Dict[str, object]]],
    ) -> Tuple[List[Dict[str, str]], str]:
        if not records:
            return ([{"label": "All", "value": ALL_CARRIER_VALUE}], ALL_CARRIER_VALUE)

        history = pd.DataFrame(records)
        options = _extract_carrier_options(history)
        return options, ALL_CARRIER_VALUE

    @callback(
        Output("performance-history-counts-chart", "figure"),
        Output("performance-history-metrics-chart", "figure"),
        Input("performance-history-store", "data"),
        Input("performance-history-carrier", "value"),
        Input("performance-history-counts", "value"),
        Input("performance-history-metrics", "value"),
        Input("performance-history-smoothing", "value"),
    )
    def update_history_charts(
        records: Optional[List[Dict[str, object]]],
        carrier: str,
        count_values: Optional[List[str]],
        metric_values: Optional[List[str]],
        smoothing_window: Optional[int],
    ) -> Tuple[go.Figure, go.Figure]:
        if not records:
            empty = _empty_figure("Load acceptance data to view history.")
            return empty, empty

        history = pd.DataFrame(records)
        if HISTORY_DATE_COLUMN not in history.columns:
            empty = _empty_figure("History data is missing dates.")
            return empty, empty

        history[HISTORY_DATE_COLUMN] = pd.to_datetime(
            history[HISTORY_DATE_COLUMN], errors="coerce"
        )
        history = history.dropna(subset=[HISTORY_DATE_COLUMN])

        if history.empty:
            empty = _empty_figure("No valid dates found in history data.")
            return empty, empty

        if carrier == ALL_CARRIER_VALUE and "carrier_code" in history.columns:
            if ALL_CARRIER_VALUE in history["carrier_code"].astype(str).unique():
                history = history[history["carrier_code"].astype(str) == ALL_CARRIER_VALUE]
        elif carrier and "carrier_code" in history.columns:
            history = history[history["carrier_code"].astype(str) == str(carrier)]

        if history.empty:
            empty = _empty_figure("No history data for the selected carrier.")
            return empty, empty

        window = int(smoothing_window) if smoothing_window else 1
        selected_counts = count_values or []
        selected_metrics = metric_values or []

        counts_history = _apply_smoothing(history, selected_counts, window)
        metrics_history = _apply_smoothing(history, selected_metrics, window)

        count_labels = {value: COUNT_VALUE_TO_LABEL.get(value, value) for value in selected_counts}
        metric_labels = {value: METRIC_VALUE_TO_LABEL.get(value, value) for value in selected_metrics}

        counts_fig = _build_line_chart(
            counts_history,
            selected_counts,
            count_labels,
            y_title="Count",
        )
        metrics_fig = _build_line_chart(
            metrics_history,
            selected_metrics,
            metric_labels,
            y_title="Metric value",
        )
        return counts_fig, metrics_fig
