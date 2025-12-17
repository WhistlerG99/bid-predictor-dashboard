"""Callback logic for the performance tracker tab."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, callback
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from ..acceptance_explorer import load_acceptance_dataset


def _select_first_series(dataset: pd.DataFrame, columns: Iterable[str]) -> Optional[pd.Series]:
    for column in columns:
        if column in dataset.columns:
            return dataset[column]
    return None


def _compute_bin_metrics(
    df: pd.DataFrame, threshold: float, window: float, stride: float
) -> pd.DataFrame:
    working = df.copy()
    working = working[working["offer_status"].isin(["TICKETED", "EXPIRED"])]
    working["hours_before_departure"] = pd.to_numeric(
        working["hours_before_departure"], errors="coerce"
    )
    working["accept_prob"] = pd.to_numeric(working["accept_prob"], errors="coerce")
    working = working.dropna(subset=["hours_before_departure", "accept_prob"])
    if working.empty:
        return pd.DataFrame()

    working["actual_positive"] = working["offer_status"] == "TICKETED"
    working["predicted_positive"] = working["accept_prob"] >= threshold * 100

    min_hour = float(np.floor(working["hours_before_departure"].min()))
    max_hour = float(np.ceil(working["hours_before_departure"].max()))
    bin_starts = np.arange(min_hour, max_hour + stride, stride)

    records: List[Dict[str, object]] = []
    for start in bin_starts:
        end = start + window
        subset = working[
            (working["hours_before_departure"] >= start)
            & (working["hours_before_departure"] < end)
        ]
        if subset.empty:
            continue

        actual_pos = int(subset["actual_positive"].sum())
        actual_neg = int((~subset["actual_positive"]).sum())
        predicted_pos = int(subset["predicted_positive"].sum())
        predicted_neg = int((~subset["predicted_positive"]).sum())
        tp = int((subset["actual_positive"] & subset["predicted_positive"]).sum())
        tn = int((~subset["actual_positive"] & ~subset["predicted_positive"]).sum())
        fp = int((~subset["actual_positive"] & subset["predicted_positive"]).sum())
        fn = int((subset["actual_positive"] & ~subset["predicted_positive"]).sum())

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else None
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        negative_precision = tn / (tn + fn) if (tn + fn) else None
        negative_recall = tn / (tn + fp) if (tn + fp) else None

        records.append(
            {
                "bin_start": start,
                "bin_end": end,
                "bin_label": f"{start:.1f}–{end:.1f}h",
                "total": total,
                "actual_pos": actual_pos,
                "actual_neg": actual_neg,
                "predicted_pos": predicted_pos,
                "predicted_neg": predicted_neg,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "negative_precision": negative_precision,
                "negative_recall": negative_recall,
            }
        )

    return pd.DataFrame.from_records(records)


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
    fig.update_layout(height=320)
    return fig


def _dual_axis_chart(
    x: List[str],
    metric: List[Optional[float]],
    secondary: List[int],
    metric_name: str,
    secondary_name: str,
    color: str,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metric,
            mode="lines+markers",
            name=metric_name,
            line={"color": color},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=x, y=secondary, name=secondary_name, marker_color="#9fb1bc", opacity=0.6),
        secondary_y=True,
    )
    fig.update_layout(
        margin={"l": 40, "r": 40, "t": 10, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_yaxes(title_text=metric_name, secondary_y=False)
    fig.update_yaxes(title_text=secondary_name, secondary_y=True)
    return fig


def _actuals_chart(x: List[str], positives: List[int], negatives: List[int]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=positives, name="Actual positives", marker_color="#1b4965"))
    fig.add_trace(go.Bar(x=x, y=negatives, name="Actual negatives", marker_color="#ff6b6b"))
    fig.update_layout(
        barmode="group",
        margin={"l": 40, "r": 20, "t": 10, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(title_text="Hours before departure window")
    fig.update_yaxes(title_text="Count")
    return fig


def register_performance_callbacks(app: Dash) -> None:
    """Register callbacks powering the performance tracker tab."""

    @callback(
        Output("performance-status", "children"),
        Output("performance-actuals", "figure"),
        Output("performance-accuracy", "figure"),
        Output("performance-recall", "figure"),
        Output("performance-precision", "figure"),
        Output("performance-negative-recall", "figure"),
        Output("performance-negative-precision", "figure"),
        Input("acceptance-dataset-path-store", "data"),
        Input("performance-threshold", "value"),
        Input("performance-window", "value"),
        Input("performance-stride", "value"),
    )
    def update_performance_charts(
        dataset_config: Optional[Mapping[str, object]],
        threshold: Optional[float],
        window: Optional[float],
        stride: Optional[float],
    ) -> Tuple[object, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]:
        if not dataset_config:
            message = "Load a dataset in the acceptance explorer controls to view performance metrics."
            empty = _empty_figure(message)
            return (message, empty, empty, empty, empty, empty, empty)

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception as exc:  # pragma: no cover - user feedback
            message = f"Failed to load dataset: {exc}"
            empty = _empty_figure(message)
            return (message, empty, empty, empty, empty, empty, empty)

        if window in (None, 0) or stride in (None, 0):
            message = "Window and stride must be greater than zero."
            empty = _empty_figure(message)
            return (message, empty, empty, empty, empty, empty, empty)

        if "hours_before_departure" not in dataset.columns:
            message = "Dataset must include an 'hours_before_departure' column."
            empty = _empty_figure(message)
            return (message, empty, empty, empty, empty, empty, empty)

        prob_series = _select_first_series(dataset, ["accept_prob", "acceptance_prob", "Acceptance Probability"])
        if prob_series is None:
            message = "Dataset must include an 'accept_prob' column."
            empty = _empty_figure(message)
            return (message, empty, empty, empty, empty, empty, empty)

        dataset = dataset.copy()
        dataset["accept_prob"] = prob_series

        metrics_df = _compute_bin_metrics(dataset, float(threshold or 0.0), float(window), float(stride))
        if metrics_df.empty:
            message = "No rows matched the selected filters."
            empty = _empty_figure(message)
            return (message, empty, empty, empty, empty, empty, empty)

        x = metrics_df["bin_label"].tolist()
        actual_chart = _actuals_chart(x, metrics_df["actual_pos"].tolist(), metrics_df["actual_neg"].tolist())
        accuracy_chart = _dual_axis_chart(
            x,
            metrics_df["accuracy"].tolist(),
            metrics_df["total"].tolist(),
            "Accuracy",
            "Items in bin",
            "#1b4965",
        )
        recall_chart = _dual_axis_chart(
            x,
            metrics_df["recall"].tolist(),
            metrics_df["actual_pos"].tolist(),
            "Recall (TPR)",
            "Actual positives",
            "#0b7a75",
        )
        precision_chart = _dual_axis_chart(
            x,
            metrics_df["precision"].tolist(),
            metrics_df["predicted_pos"].tolist(),
            "Precision (PPV)",
            "Predicted positives",
            "#f4a261",
        )
        negative_recall_chart = _dual_axis_chart(
            x,
            metrics_df["negative_recall"].tolist(),
            metrics_df["actual_neg"].tolist(),
            "Negative recall (TNR)",
            "Actual negatives",
            "#457b9d",
        )
        negative_precision_chart = _dual_axis_chart(
            x,
            metrics_df["negative_precision"].tolist(),
            metrics_df["predicted_neg"].tolist(),
            "Negative precision (NPV)",
            "Predicted negatives",
            "#6d597a",
        )

        return (
            "",
            actual_chart,
            accuracy_chart,
            recall_chart,
            precision_chart,
            negative_recall_chart,
            negative_precision_chart,
        )


__all__ = ["register_performance_callbacks"]
