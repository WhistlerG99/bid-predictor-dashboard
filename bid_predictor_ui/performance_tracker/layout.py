"""Layout for the performance tracker tab."""
from __future__ import annotations

from dash import dcc, html


CONTROL_CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "12px",
    "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
    "padding": "1rem",
    "marginBottom": "1rem",
}


def _build_control_panel() -> html.Div:
    """Compose the controls for performance calculations."""

    return html.Div(
        [
            html.Div(
                [
                    html.Label("Threshold", style={"fontWeight": "600"}),
                    dcc.Slider(
                        id="performance-threshold",
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.5,
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            html.Div(
                [
                    html.Label("Carrier", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="performance-carrier",
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
                    html.Div(
                        [
                            html.Label("Window (hours)", style={"fontWeight": "600"}),
                            dcc.Input(
                                id="performance-window",
                                type="number",
                                min=0.1,
                                step=0.1,
                                value=1.0,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Label("Stride (hours)", style={"fontWeight": "600"}),
                            dcc.Input(
                                id="performance-stride",
                                type="number",
                                min=0.1,
                                step=0.1,
                                value=1.0,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "1rem"},
            ),
        ],
        style=CONTROL_CARD_STYLE,
    )


def _build_chart_grid() -> html.Div:
    """Create the 2x3 grid of performance charts."""

    chart_ids = [
        ("performance-actuals", "Actual positives vs negatives"),
        ("performance-accuracy", "Accuracy"),
        ("performance-recall", "Recall (True Positive Rate)"),
        ("performance-precision", "Precision (Positive Predictive Value)"),
        ("performance-negative-recall", "Negative Recall (True Negative Rate)"),
        ("performance-negative-precision", "Negative Precision (Negative Predictive Value)"),
    ]

    cells = []
    for chart_id, title in chart_ids:
        cells.append(
            html.Div(
                [
                    html.H4(title, style={"margin": "0 0 0.5rem 0", "color": "#1b4965"}),
                    dcc.Graph(id=chart_id, figure={}, style={"height": "320px"}),
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "borderRadius": "12px",
                    "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
                    "padding": "1rem",
                },
            )
        )

    return html.Div(
        cells,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
            "gap": "1rem",
        },
    )


def _build_distribution_section() -> html.Div:
    """Create the acceptance probability distribution controls and chart."""

    control_panel = html.Div(
        [
            html.H4(
                "Display settings", style={"margin": "0 0 0.75rem 0", "color": "#1b4965"}
            ),
            html.Div(
                [
                    html.Label("Y-axis scale", style={"fontWeight": "600"}),
                    dcc.RadioItems(
                        id="accept-prob-scale",
                        options=[
                            {"label": "Linear", "value": "linear"},
                            {"label": "Log", "value": "log"},
                        ],
                        value="linear",
                        labelStyle={"display": "inline-block", "marginRight": "1rem"},
                        style={"marginBottom": "1rem"},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Number of bins", style={"fontWeight": "600"}),
                    dcc.Input(
                        id="accept-prob-bin-count",
                        type="number",
                        min=1,
                        step=1,
                        debounce=False,
                        value=30,
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            html.Div(
                [
                    html.Label("Carrier", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="accept-prob-carrier",
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
                    html.Label("Hours before departure", style={"fontWeight": "600"}),
                    dcc.RangeSlider(
                        id="accept-prob-hours-range",
                        min=0,
                        max=100,
                        step=1,
                        value=[0, 100],
                        allowCross=False,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ]
            ),
        ],
        style=CONTROL_CARD_STYLE,
    )

    return html.Div(
        [
            html.H3(
                "Acceptance Probability Distribution",
                style={"margin": "0 0 0.75rem 0", "color": "#1b4965"},
            ),
            html.Div(
                [
                    html.Div(control_panel, style={"flex": "0 0 320px"}),
                    html.Div(
                        [
                            html.H4(
                                "Histogram by offer status",
                                style={"margin": "0 0 0.5rem 0", "color": "#1b4965"},
                            ),
                            dcc.Graph(
                                id="accept-prob-distribution",
                                figure={},
                                style={"height": "420px"},
                            ),
                        ],
                        style={
                            "backgroundColor": "#ffffff",
                            "borderRadius": "12px",
                            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
                            "padding": "1rem",
                            "flex": "1",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "1rem", "alignItems": "stretch"},
            ),
        ],
        style={"marginTop": "1.5rem"},
    )


def _build_roc_pr_section() -> html.Div:
    """Create the ROC and precision-recall controls and charts."""

    control_panel = html.Div(
        [
            html.H4(
                "Filter settings", style={"margin": "0 0 0.75rem 0", "color": "#1b4965"}
            ),
            html.Div(
                [
                    html.Label("Carrier", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="roc-pr-carrier",
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
                    html.Label("Hours before departure", style={"fontWeight": "600"}),
                    dcc.RangeSlider(
                        id="roc-pr-hours-range",
                        min=0,
                        max=100,
                        step=1,
                        value=[0, 100],
                        allowCross=False,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ]
            ),
        ],
        style=CONTROL_CARD_STYLE,
    )

    charts = html.Div(
        [
            html.H4(
                "ROC and Precision-Recall Curves",
                style={"margin": "0 0 0.75rem 0", "color": "#1b4965"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H5(
                                "ROC curve",
                                style={"margin": "0 0 0.5rem 0", "color": "#1b4965"},
                            ),
                            dcc.Graph(id="roc-curve", figure={}, style={"height": "360px"}),
                        ],
                        style={
                            "backgroundColor": "#ffffff",
                            "borderRadius": "12px",
                            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
                            "padding": "1rem",
                            "flex": "1",
                        },
                    ),
                    html.Div(
                        [
                            html.H5(
                                "Precision-Recall curve",
                                style={"margin": "0 0 0.5rem 0", "color": "#1b4965"},
                            ),
                            dcc.Graph(
                                id="precision-recall-curve",
                                figure={},
                                style={"height": "360px"},
                            ),
                        ],
                        style={
                            "backgroundColor": "#ffffff",
                            "borderRadius": "12px",
                            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
                            "padding": "1rem",
                            "flex": "1",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                    "gap": "1rem",
                },
            ),
        ],
        style={"flex": "1"},
    )

    return html.Div(
        [
            html.H3(
                "ROC and Precision-Recall Curves",
                style={"margin": "1.5rem 0 0.75rem 0", "color": "#1b4965"},
            ),
            html.Div(
                [
                    html.Div(control_panel, style={"flex": "0 0 320px"}),
                    charts,
                ],
                style={"display": "flex", "gap": "1rem", "alignItems": "stretch"},
            ),
        ]
    )


def build_performance_tab() -> dcc.Tab:
    """Compose the performance tracker tab."""

    return dcc.Tab(
        label="Performance tracker",
        value="performance",
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(
                                "Model performance vs. hours before departure",
                                style={"margin": "0", "color": "#1b4965"},
                            ),
                            html.P(
                                "Configure the threshold, window, and stride to see how classification metrics change as departure approaches.",
                                style={"margin": "0", "color": "#16324f"},
                            ),
                        ],
                        style={"marginBottom": "1rem"},
                    ),
                    _build_control_panel(),
                    html.Div(
                        id="performance-status",
                        style={"marginBottom": "1rem", "color": "#c1121f", "fontWeight": 600},
                    ),
                    _build_chart_grid(),
                    _build_distribution_section(),
                    _build_roc_pr_section(),
                ],
                style={"padding": "1rem"},
            )
        ],
    )
