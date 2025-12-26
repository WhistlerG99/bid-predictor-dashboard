"""Layout for the performance history tab."""
from __future__ import annotations

from dash import dcc, html

COUNT_OPTIONS = [
    {"label": "Total Number of Items", "value": "total"},
    {"label": "Number of Actual Positive", "value": "actual_pos"},
    {"label": "Number of Actual Negatives", "value": "actual_neg"},
    {"label": "Number of True Positive", "value": "tp"},
    {"label": "Number of False Positive", "value": "fp"},
    {"label": "Number of True Negatives", "value": "tn"},
    {"label": "Number of False Negatives", "value": "fn"},
]

METRIC_OPTIONS = [
    {"label": "Accuracy", "value": "accuracy"},
    {"label": "Balanced Accuracy", "value": "balanced_accuracy"},
    {"label": "Prevalence", "value": "prevalence"},
    {"label": "F-Score", "value": "f_score"},
    {"label": "FM Index", "value": "fm_index"},
    {"label": "Negative F-Score", "value": "negative_f_score"},
    {"label": "Negative FM Index", "value": "negative_fm_index"},
    {"label": "Precision", "value": "precision"},
    {"label": "Recall", "value": "recall"},
    {"label": "False Negative Rate", "value": "false_negative_rate"},
    {"label": "Negative Precision", "value": "negative_precision"},
    {"label": "Negative Recall", "value": "negative_recall"},
    {"label": "False Positive Rate", "value": "false_positive_rate"},
]

CONTROL_CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "12px",
    "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
    "padding": "1rem",
    "marginBottom": "1rem",
}


def _build_controls() -> html.Div:
    return html.Div(
        [
            html.H4("Display settings", style={"margin": "0 0 0.75rem 0", "color": "#1b4965"}),
            html.Div(
                [
                    html.Label("Carrier", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="performance-history-carrier",
                        placeholder="All carriers",
                        options=[],
                        value="ALL",
                        clearable=False,
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            html.Div(
                [
                    html.Label("Counts", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="performance-history-counts",
                        options=COUNT_OPTIONS,
                        value=["total", "actual_pos", "actual_neg"],
                        multi=True,
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            html.Div(
                [
                    html.Label("Metrics", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="performance-history-metrics",
                        options=METRIC_OPTIONS,
                        value=["accuracy", "precision", "recall"],
                        multi=True,
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            html.Div(
                [
                    html.Label("Smoothing window (days)", style={"fontWeight": "600"}),
                    dcc.Slider(
                        id="performance-history-smoothing",
                        min=1,
                        max=14,
                        step=1,
                        value=1,
                        marks={1: "1", 7: "7", 14: "14"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
            ),
        ],
        style=CONTROL_CARD_STYLE,
    )


def _build_chart_card(title: str, graph_id: str) -> html.Div:
    return html.Div(
        [
            html.H4(title, style={"margin": "0 0 0.5rem 0", "color": "#1b4965"}),
            dcc.Graph(id=graph_id, figure={}, style={"height": "380px"}),
        ],
        style={
            "backgroundColor": "#ffffff",
            "borderRadius": "12px",
            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
            "padding": "1rem",
        },
    )


def build_performance_history_tab() -> html.Div:
    return html.Div(
        [
            html.H3("Performance History", style={"margin": "0 0 0.75rem 0", "color": "#1b4965"}),
            html.Div(
                id="performance-history-status",
                className="status-message",
                style={"marginBottom": "0.75rem", "color": "#c1121f", "fontWeight": 600},
            ),
            html.Div(
                [
                    html.Div(_build_controls(), className="side-panel", style={"flex": "0 0 320px"}),
                    html.Div(
                        [
                            _build_chart_card("Counts over time", "performance-history-counts-chart"),
                            _build_chart_card("Metrics over time", "performance-history-metrics-chart"),
                        ],
                        className="chart-grid",
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                            "gap": "1rem",
                            "flex": "1",
                        },
                    ),
                ],
                className="split-panel",
                style={
                    "display": "flex",
                    "gap": "1rem",
                    "alignItems": "stretch",
                    "flexWrap": "wrap",
                },
            ),
            dcc.Store(id="performance-history-store"),
        ],
        style={"marginBottom": "1.5rem"},
    )
