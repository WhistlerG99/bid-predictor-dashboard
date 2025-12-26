"""Callback logic for the performance tracker tab."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, callback, html
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from ..acceptance_explorer import load_acceptance_dataset


DEFAULT_THRESHOLD_POINTS = 200


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


def _format_metric_value(value: Optional[float], precision: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{precision}f}"


def _format_count_value(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{int(value):,}"


def _metric_tile(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                label,
                style={
                    "fontSize": "0.95rem",
                    "fontWeight": 700,
                    "color": "#1b4965",
                    "fontFamily": "Inter, 'Segoe UI', sans-serif",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "1.4rem",
                    "fontWeight": 700,
                    "color": "#16324f",
                    "fontFamily": "Inter, 'Segoe UI', sans-serif",
                },
            ),
        ],
        style={
            "padding": "0.75rem",
            "backgroundColor": "#f8fafc",
            "borderRadius": "10px",
            "display": "flex",
            "flexDirection": "column",
            "gap": "0.35rem",
        },
    )


def _metric_section(title: str, tiles: List[html.Div], columns: int) -> html.Div:
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "fontSize": "1rem",
                    "fontWeight": 700,
                    "color": "#1b4965",
                    "fontFamily": "Inter, 'Segoe UI', sans-serif",
                },
            ),
            html.Div(
                tiles,
                style={
                    "display": "grid",
                    "gridTemplateColumns": f"repeat({columns}, minmax(0, 1fr))",
                    "gap": "0.75rem",
                },
            ),
        ],
        style={"display": "flex", "flexDirection": "column", "gap": "0.5rem"},
    )


def _performance_overview_tiles(
    df: pd.DataFrame,
    threshold: float,
    carrier: Optional[str],
    hours_range: Optional[Iterable[float]],
) -> Tuple[List[html.Div], Optional[str]]:
    working = _filter_acceptance_rows(df, carrier, hours_range)
    working["accept_prob"] = pd.to_numeric(working.get("accept_prob"), errors="coerce")
    working = working.dropna(subset=["accept_prob", "offer_status"])
    if working.empty:
        return [], "No rows matched the selected filters."

    actual_positive = working["offer_status"] == "TICKETED"
    predicted_positive = working["accept_prob"] >= threshold * 100
    tp = int((actual_positive & predicted_positive).sum())
    tn = int((~actual_positive & ~predicted_positive).sum())
    fp = int((~actual_positive & predicted_positive).sum())
    fn = int((actual_positive & ~predicted_positive).sum())
    total = int(tp + tn + fp + fn)
    pos = tp + fn
    neg = tn + fp

    accuracy = (tp + tn) / total if total else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    negative_precision = tn / (tn + fn) if (tn + fn) else np.nan
    negative_recall = tn / (tn + fp) if (tn + fp) else np.nan
    fpr_val = fp / neg if neg else np.nan
    fnr_val = fn / pos if pos else np.nan

    count_summary_tiles = [
        _metric_tile("Total number of items", _format_count_value(total)),
        _metric_tile("Number of actual positives", _format_count_value(pos)),
        _metric_tile("Number of actual negatives", _format_count_value(neg)),
    ]
    count_detail_tiles = [
        _metric_tile("Number of true positives", _format_count_value(tp)),
        _metric_tile("Number of false positives", _format_count_value(fp)),
        _metric_tile("Number of true negatives", _format_count_value(tn)),
        _metric_tile("Number of false negatives", _format_count_value(fn)),
    ]
    positive_tiles = [
        _metric_tile("Accuracy", _format_metric_value(accuracy)),
        _metric_tile("Precision", _format_metric_value(precision)),
        _metric_tile("Recall", _format_metric_value(recall)),
    ]
    negative_tiles = [
        _metric_tile("Negative Precision", _format_metric_value(negative_precision)),
        _metric_tile("Negative Recall", _format_metric_value(negative_recall)),
    ]
    rate_tiles = [
        _metric_tile("False Positive Rate", _format_metric_value(fpr_val)),
        _metric_tile("False negative rate", _format_metric_value(fnr_val)),
    ]

    sections = [
        html.Div(
            [
                html.Div(
                    "Counts",
                    style={
                        "fontSize": "1rem",
                        "fontWeight": 700,
                        "color": "#1b4965",
                        "fontFamily": "Inter, 'Segoe UI', sans-serif",
                    },
                ),
                html.Div(
                    count_summary_tiles,
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                        "gap": "0.75rem",
                    },
                ),
                html.Div(
                    count_detail_tiles,
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                        "gap": "0.75rem",
                    },
                ),
            ],
            style={"display": "flex", "flexDirection": "column", "gap": "0.5rem"},
        ),
        _metric_section("Positive-class metrics", positive_tiles, 3),
        _metric_section("Negative-class metrics", negative_tiles, 2),
        _metric_section("Rates", rate_tiles, 2),
    ]
    return sections, None


def _accept_prob_distribution(
    df: pd.DataFrame,
    bin_count: int,
    yaxis_scale: str,
    carrier: Optional[str],
    hours_range: Optional[Iterable[float]],
) -> go.Figure:
    working = _filter_acceptance_rows(df, carrier, hours_range)
    working["accept_prob"] = pd.to_numeric(working["accept_prob"], errors="coerce")
    working = working.dropna(subset=["accept_prob"])
    if working.empty:
        return _empty_figure("No rows matched the selected filters.")

    fig = go.Figure()
    colors = {"TICKETED": "#1b4965", "EXPIRED": "#ff6b6b"}
    for status, color in colors.items():
        subset = working[working["offer_status"] == status]
        if subset.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=subset["accept_prob"],
                name=status,
                marker_color=color,
                opacity=0.65,
                nbinsx=max(int(bin_count), 1),
            )
        )

    if not fig.data:
        return _empty_figure("No rows matched the selected filters.")

    fig.update_layout(
        barmode="overlay",
        margin={"l": 40, "r": 20, "t": 10, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(title_text="Acceptance probability (%)", rangemode="tozero")
    fig.update_yaxes(title_text="Count", type=yaxis_scale, rangemode="tozero")
    return fig


def _build_carrier_options(dataset: pd.DataFrame) -> List[Dict[str, str]]:
    options = [{"label": "All", "value": "ALL"}]
    if dataset.empty or "carrier_code" not in dataset.columns:
        return options

    carriers = (
        dataset["carrier_code"].astype(str).dropna().drop_duplicates().sort_values()
    )
    options.extend({"label": str(code), "value": str(code)} for code in carriers)
    return options


def _generate_thresholds(scores: np.ndarray, requested_points: int) -> np.ndarray:
    valid_scores = scores[np.isfinite(scores)]
    if valid_scores.size == 0:
        return np.array([], dtype=float)

    min_score = float(np.min(valid_scores))
    max_score = float(np.max(valid_scores))
    if not np.isfinite(min_score) or not np.isfinite(max_score):
        return np.array([], dtype=float)

    target_points = max(int(requested_points), 11)
    quantiles = np.linspace(0, 1, num=target_points)
    sampled = np.quantile(np.unique(valid_scores), quantiles)

    padding = max(1e-6, (max_score - min_score) * 1e-3)
    thresholds = np.concatenate(
        ([max_score + padding], sampled[::-1], [min_score - padding])
    )

    deduped: List[float] = []
    for value in thresholds:
        if not deduped or not np.isclose(value, deduped[-1]):
            deduped.append(float(value))
    return np.array(deduped, dtype=float)


def _roc_pr_curves(
    df: pd.DataFrame,
    carrier: Optional[str],
    hours_range: Optional[Iterable[float]],
    threshold_points: int,
) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    working = _filter_acceptance_rows(df, carrier, hours_range)
    working["accept_prob"] = pd.to_numeric(working.get("accept_prob"), errors="coerce")
    working = working.dropna(subset=["accept_prob", "offer_status"])
    if working.empty:
        message = "No rows matched the selected filters."
        empty = _empty_figure(message)
        return empty, empty, empty, empty

    y_true = (working["offer_status"] == "TICKETED").astype(int)
    y_score = working["accept_prob"]

    if y_true.nunique() < 2:
        message = "Both TICKETED and EXPIRED rows are required to plot the curves."
        empty = _empty_figure(message)
        return empty, empty, empty, empty

    thresholds = _generate_thresholds(y_score.to_numpy(dtype=float), threshold_points)
    if thresholds.size == 0:
        empty = _empty_figure("No valid thresholds available to plot curves.")
        return empty, empty, empty, empty

    labels = y_true.to_numpy(dtype=int)
    scores = y_score.to_numpy(dtype=float)

    metrics: List[Tuple[float, float, float, float, float, float, float, float, float]] = []
    for threshold in thresholds:
        preds = scores >= threshold
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))

        pos = tp + fn
        neg = tn + fp

        tpr_val = tp / pos if pos else np.nan
        fpr_val = fp / neg if neg else np.nan
        precision_val = tp / (tp + fp) if (tp + fp) else np.nan
        recall_val = tpr_val
        tnr_val = tn / neg if neg else np.nan
        fnr_val = fn / pos if pos else np.nan
        neg_precision_val = tn / (tn + fn) if (tn + fn) else np.nan
        neg_recall_val = tnr_val

        metrics.append(
            (
                threshold,
                fpr_val,
                tpr_val,
                recall_val,
                precision_val,
                fnr_val,
                tnr_val,
                neg_recall_val,
                neg_precision_val,
            )
        )

    metrics_arr = np.array(metrics, dtype=float)

    threshold_vals = metrics_arr[:, 0]
    fpr_vals = metrics_arr[:, 1]
    tpr_vals = metrics_arr[:, 2]
    recall_vals = metrics_arr[:, 3]
    precision_vals = metrics_arr[:, 4]
    fnr_vals = metrics_arr[:, 5]
    tnr_vals = metrics_arr[:, 6]
    neg_recall_vals = metrics_arr[:, 7]
    neg_precision_vals = metrics_arr[:, 8]

    shared_customdata = np.column_stack(
        (
            threshold_vals,
            fpr_vals,
            tpr_vals,
            recall_vals,
            precision_vals,
            fnr_vals,
            tnr_vals,
            neg_recall_vals,
            neg_precision_vals,
        )
    )

    roc_fig = go.Figure()
    roc_fig.add_trace(
        go.Scatter(
            x=fpr_vals,
            y=tpr_vals,
            mode="lines",
            name="ROC",
            marker={"color": "#1b4965"},
            line={"color": "#1b4965"},
            customdata=shared_customdata,
            hovertemplate=(
                "Threshold: %{customdata[0]:.3f}<br>"
                "FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>"
                "PR Recall: %{customdata[3]:.3f}<br>PR Precision: %{customdata[4]:.3f}<br>"
                "Neg ROC FNR: %{customdata[5]:.3f}<br>Neg ROC TNR: %{customdata[6]:.3f}<br>"
                "Neg PR Recall: %{customdata[7]:.3f}<br>Neg PR Precision: %{customdata[8]:.3f}<extra></extra>"
            ),
        )
    )
    roc_fig.update_layout(
        margin={"l": 32, "r": 16, "t": 10, "b": 32},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    roc_fig.update_xaxes(
        title_text="False positive rate",
        title_standoff=6,
        tickfont={"size": 10},
        rangemode="tozero",
    )
    roc_fig.update_yaxes(
        title_text="True positive rate",
        title_standoff=6,
        tickfont={"size": 10},
        rangemode="tozero",
    )

    pr_fig = go.Figure()
    pr_fig.add_trace(
        go.Scatter(
            x=recall_vals,
            y=precision_vals,
            mode="lines",
            name="Precision-Recall",
            marker={"color": "#f4a261"},
            line={"color": "#f4a261"},
            customdata=shared_customdata,
            hovertemplate=(
                "Threshold: %{customdata[0]:.3f}<br>"
                "Recall: %{x:.3f}<br>Precision: %{y:.3f}<br>"
                "ROC FPR: %{customdata[1]:.3f}<br>ROC TPR: %{customdata[2]:.3f}<br>"
                "Neg ROC FNR: %{customdata[5]:.3f}<br>Neg ROC TNR: %{customdata[6]:.3f}<br>"
                "Neg PR Recall: %{customdata[7]:.3f}<br>Neg PR Precision: %{customdata[8]:.3f}<extra></extra>"
            ),
        )
    )
    pr_fig.update_layout(
        margin={"l": 32, "r": 16, "t": 10, "b": 32},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    pr_fig.update_xaxes(
        title_text="Recall",
        title_standoff=6,
        tickfont={"size": 10},
        rangemode="tozero",
    )
    pr_fig.update_yaxes(
        title_text="Precision",
        title_standoff=6,
        tickfont={"size": 10},
        rangemode="tozero",
    )

    neg_roc_fig = go.Figure()
    neg_roc_fig.add_trace(
        go.Scatter(
            x=fnr_vals,
            y=tnr_vals,
            mode="lines",
            name="Negative ROC",
            marker={"color": "#2a9d8f"},
            line={"color": "#2a9d8f"},
            customdata=shared_customdata,
            hovertemplate=(
                "Threshold: %{customdata[0]:.3f}<br>"
                "FNR: %{x:.3f}<br>TNR: %{y:.3f}<br>"
                "ROC FPR: %{customdata[1]:.3f}<br>ROC TPR: %{customdata[2]:.3f}<br>"
                "PR Recall: %{customdata[3]:.3f}<br>PR Precision: %{customdata[4]:.3f}<br>"
                "Neg PR Recall: %{customdata[7]:.3f}<br>Neg PR Precision: %{customdata[8]:.3f}<extra></extra>"
            ),
        )
    )
    neg_roc_fig.update_layout(
        margin={"l": 32, "r": 16, "t": 10, "b": 32},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    neg_roc_fig.update_xaxes(
        title_text="False negative rate",
        title_standoff=6,
        tickfont={"size": 10},
        rangemode="tozero",
    )
    neg_roc_fig.update_yaxes(
        title_text="True negative rate",
        title_standoff=6,
        tickfont={"size": 10},
        rangemode="tozero",
    )

    neg_pr_fig = go.Figure()
    neg_pr_fig.add_trace(
        go.Scatter(
            x=neg_recall_vals,
            y=neg_precision_vals,
            mode="lines",
            name="Negative Precision-Recall",
            marker={"color": "#6d597a"},
            line={"color": "#6d597a"},
            customdata=shared_customdata,
            hovertemplate=(
                "Threshold: %{customdata[0]:.3f}<br>"
                "Negative Recall: %{x:.3f}<br>Negative Precision: %{y:.3f}<br>"
                "ROC FPR: %{customdata[1]:.3f}<br>ROC TPR: %{customdata[2]:.3f}<br>"
                "PR Recall: %{customdata[3]:.3f}<br>PR Precision: %{customdata[4]:.3f}<br>"
                "Neg ROC FNR: %{customdata[5]:.3f}<br>Neg ROC TNR: %{customdata[6]:.3f}<extra></extra>"
            ),
        )
    )
    neg_pr_fig.update_layout(
        margin={"l": 32, "r": 16, "t": 10, "b": 32},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    neg_pr_fig.update_xaxes(
        title_text="Negative recall",
        title_standoff=6,
        tickfont={"size": 10},
        rangemode="tozero",
    )
    neg_pr_fig.update_yaxes(
        title_text="Negative precision",
        title_standoff=6,
        tickfont={"size": 10},
        rangemode="tozero",
    )

    return roc_fig, pr_fig, neg_roc_fig, neg_pr_fig


def _compute_hours_range(dataset: pd.DataFrame) -> Tuple[float, float, List[float], Dict[int, str]]:
    default_min, default_max = 0.0, 100.0
    default_marks = {int(default_min): "0h", int(default_max): "100h"}
    if dataset.empty or "hours_before_departure" not in dataset.columns:
        return default_min, default_max, [default_min, default_max], default_marks

    hours = pd.to_numeric(dataset["hours_before_departure"], errors="coerce").dropna()
    if hours.empty:
        return default_min, default_max, [default_min, default_max], default_marks

    min_hour = float(np.floor(hours.min()))
    max_hour = float(np.ceil(hours.max()))
    if min_hour == max_hour:
        max_hour = min_hour + 1.0

    marks = {int(min_hour): f"{min_hour:.0f}h", int(max_hour): f"{max_hour:.0f}h"}
    return min_hour, max_hour, [min_hour, max_hour], marks


def _filter_acceptance_rows(
    dataset: pd.DataFrame,
    carrier: Optional[str],
    hours_range: Optional[Iterable[float]],
) -> pd.DataFrame:
    working = dataset.copy()
    working = working[working["offer_status"].isin(["TICKETED", "EXPIRED"])]
    if carrier and carrier != "ALL" and "carrier_code" in working.columns:
        working = working[working["carrier_code"].astype(str) == str(carrier)]
    working["hours_before_departure"] = pd.to_numeric(
        working.get("hours_before_departure", pd.Series(dtype=float)), errors="coerce"
    )
    if hours_range is not None:
        try:
            lower, upper = float(hours_range[0]), float(hours_range[1])
            working = working[
                working["hours_before_departure"].between(lower, upper, inclusive="both")
            ]
        except (TypeError, ValueError, IndexError):
            pass
    return working


def register_performance_callbacks(app: Dash) -> None:
    """Register callbacks powering the performance tracker tab."""

    @callback(
        Output("accept-prob-carrier", "options"),
        Output("accept-prob-carrier", "value"),
        Input("acceptance-dataset-path-store", "data"),
    )
    def populate_accept_prob_carriers(
        dataset_config: Optional[Mapping[str, object]]
    ) -> Tuple[List[Dict[str, str]], str]:
        if not dataset_config:
            return ([{"label": "All", "value": "ALL"}], "ALL")

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception:  # pragma: no cover - user feedback path
            return ([{"label": "All", "value": "ALL"}], "ALL")

        options = _build_carrier_options(dataset)
        return (options, "ALL")

    @callback(
        Output("accept-prob-hours-range", "min"),
        Output("accept-prob-hours-range", "max"),
        Output("accept-prob-hours-range", "value"),
        Output("accept-prob-hours-range", "marks"),
        Input("acceptance-dataset-path-store", "data"),
    )
    def configure_accept_prob_hours_range(
        dataset_config: Optional[Mapping[str, object]]
    ) -> Tuple[float, float, List[float], Dict[int, str]]:
        if not dataset_config:
            return _compute_hours_range(pd.DataFrame())

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception:  # pragma: no cover - user feedback path
            return _compute_hours_range(pd.DataFrame())

        return _compute_hours_range(dataset)

    @callback(
        Output("roc-pr-carrier", "options"),
        Output("roc-pr-carrier", "value"),
        Input("acceptance-dataset-path-store", "data"),
    )
    def populate_roc_pr_carriers(
        dataset_config: Optional[Mapping[str, object]]
    ) -> Tuple[List[Dict[str, str]], str]:
        if not dataset_config:
            return ([{"label": "All", "value": "ALL"}], "ALL")

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception:  # pragma: no cover - user feedback path
            return ([{"label": "All", "value": "ALL"}], "ALL")

        options = _build_carrier_options(dataset)
        return (options, "ALL")

    @callback(
        Output("roc-pr-hours-range", "min"),
        Output("roc-pr-hours-range", "max"),
        Output("roc-pr-hours-range", "value"),
        Output("roc-pr-hours-range", "marks"),
        Input("acceptance-dataset-path-store", "data"),
    )
    def configure_roc_pr_hours_range(
        dataset_config: Optional[Mapping[str, object]]
    ) -> Tuple[float, float, List[float], Dict[int, str]]:
        if not dataset_config:
            return _compute_hours_range(pd.DataFrame())

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception:  # pragma: no cover - user feedback path
            return _compute_hours_range(pd.DataFrame())

        return _compute_hours_range(dataset)

    @callback(
        Output("performance-carrier", "options"),
        Output("performance-carrier", "value"),
        Input("acceptance-dataset-path-store", "data"),
    )
    def populate_performance_carriers(
        dataset_config: Optional[Mapping[str, object]]
    ) -> Tuple[List[Dict[str, str]], str]:
        if not dataset_config:
            return ([{"label": "All", "value": "ALL"}], "ALL")

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception:  # pragma: no cover - user feedback path
            return ([{"label": "All", "value": "ALL"}], "ALL")

        options = _build_carrier_options(dataset)
        return (options, "ALL")

    @callback(
        Output("performance-overview-carrier", "options"),
        Output("performance-overview-carrier", "value"),
        Input("acceptance-dataset-path-store", "data"),
    )
    def populate_performance_overview_carriers(
        dataset_config: Optional[Mapping[str, object]]
    ) -> Tuple[List[Dict[str, str]], str]:
        if not dataset_config:
            return ([{"label": "All", "value": "ALL"}], "ALL")

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception:  # pragma: no cover - user feedback path
            return ([{"label": "All", "value": "ALL"}], "ALL")

        options = _build_carrier_options(dataset)
        return (options, "ALL")

    @callback(
        Output("performance-overview-hours-range", "min"),
        Output("performance-overview-hours-range", "max"),
        Output("performance-overview-hours-range", "value"),
        Output("performance-overview-hours-range", "marks"),
        Input("acceptance-dataset-path-store", "data"),
    )
    def configure_performance_overview_hours_range(
        dataset_config: Optional[Mapping[str, object]]
    ) -> Tuple[float, float, List[float], Dict[int, str]]:
        if not dataset_config:
            return _compute_hours_range(pd.DataFrame())

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception:  # pragma: no cover - user feedback path
            return _compute_hours_range(pd.DataFrame())

        return _compute_hours_range(dataset)

    @callback(
        Output("performance-overview-status", "children"),
        Output("performance-overview-grid", "children"),
        Input("acceptance-dataset-path-store", "data"),
        Input("performance-overview-threshold", "value"),
        Input("performance-overview-carrier", "value"),
        Input("performance-overview-hours-range", "value"),
    )
    def update_performance_overview_table(
        dataset_config: Optional[Mapping[str, object]],
        threshold: Optional[float],
        carrier: Optional[str],
        hours_range: Optional[Iterable[float]],
    ) -> Tuple[str, List[html.Div]]:
        if not dataset_config:
            message = "Load a dataset in the acceptance explorer controls to view performance metrics."
            return message, []

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception as exc:  # pragma: no cover - user feedback
            message = f"Failed to load dataset: {exc}"
            return message, []

        prob_series = _select_first_series(
            dataset, ["accept_prob", "acceptance_prob", "Acceptance Probability"]
        )
        if prob_series is None:
            return "Dataset must include an 'accept_prob' column.", []

        dataset = dataset.copy()
        dataset["accept_prob"] = prob_series

        sections, error = _performance_overview_tiles(
            dataset, float(threshold or 0.0), carrier, hours_range
        )
        return (error or ""), sections

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
        Input("performance-carrier", "value"),
    )
    def update_performance_charts(
        dataset_config: Optional[Mapping[str, object]],
        threshold: Optional[float],
        window: Optional[float],
        stride: Optional[float],
        carrier: Optional[str],
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
        if carrier and carrier != "ALL" and "carrier_code" in dataset.columns:
            dataset = dataset[dataset["carrier_code"].astype(str) == str(carrier)]

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

    @callback(
        Output("accept-prob-distribution", "figure"),
        Input("acceptance-dataset-path-store", "data"),
        Input("accept-prob-bin-count", "value"),
        Input("accept-prob-scale", "value"),
        Input("accept-prob-carrier", "value"),
        Input("accept-prob-hours-range", "value"),
    )
    def update_accept_prob_distribution(
        dataset_config: Optional[Mapping[str, object]],
        bin_count: Optional[float],
        yaxis_scale: Optional[str],
        carrier: Optional[str],
        hours_range: Optional[Iterable[float]],
    ) -> go.Figure:
        if not dataset_config:
            return _empty_figure(
                "Load a dataset in the acceptance explorer controls to view the distribution."
            )

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception as exc:  # pragma: no cover - user feedback
            return _empty_figure(f"Failed to load dataset: {exc}")

        prob_series = _select_first_series(
            dataset, ["accept_prob", "acceptance_prob", "Acceptance Probability"]
        )
        if prob_series is None:
            return _empty_figure("Dataset must include an 'accept_prob' column.")

        dataset = dataset.copy()
        dataset["accept_prob"] = prob_series

        selected_scale = yaxis_scale if yaxis_scale in ("linear", "log") else "linear"
        try:
            parsed_bins = int(bin_count) if bin_count is not None else 30
        except (TypeError, ValueError):
            parsed_bins = 30
        selected_bins = parsed_bins if parsed_bins > 0 else 30
        return _accept_prob_distribution(
            dataset, selected_bins, selected_scale, carrier, hours_range
        )

    @callback(
        Output("roc-curve", "figure"),
        Output("precision-recall-curve", "figure"),
        Output("negative-roc-curve", "figure"),
        Output("negative-precision-recall-curve", "figure"),
        Input("acceptance-dataset-path-store", "data"),
        Input("roc-pr-carrier", "value"),
        Input("roc-pr-hours-range", "value"),
        Input("roc-pr-threshold-points", "value"),
    )
    def update_roc_pr_curves(
        dataset_config: Optional[Mapping[str, object]],
        carrier: Optional[str],
        hours_range: Optional[Iterable[float]],
        threshold_points: Optional[float],
    ) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
        if not dataset_config:
            message = "Load a dataset in the acceptance explorer controls to view the curves."
            empty = _empty_figure(message)
            return empty, empty, empty, empty

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception as exc:  # pragma: no cover - user feedback
            message = f"Failed to load dataset: {exc}"
            empty = _empty_figure(message)
            return empty, empty

        prob_series = _select_first_series(
            dataset, ["accept_prob", "acceptance_prob", "Acceptance Probability"]
        )
        if prob_series is None:
            empty = _empty_figure("Dataset must include an 'accept_prob' column.")
            return empty, empty, empty, empty

        dataset = dataset.copy()
        dataset["accept_prob"] = prob_series

        try:
            requested_points = (
                int(threshold_points)
                if threshold_points is not None
                else DEFAULT_THRESHOLD_POINTS
            )
        except (TypeError, ValueError):
            requested_points = DEFAULT_THRESHOLD_POINTS

        if requested_points <= 10:
            requested_points = DEFAULT_THRESHOLD_POINTS

        return _roc_pr_curves(dataset, carrier, hours_range, requested_points)


__all__ = ["register_performance_callbacks"]
