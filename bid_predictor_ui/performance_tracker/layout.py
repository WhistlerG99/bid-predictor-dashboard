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
                ],
                style={"padding": "1rem"},
            )
        ],
    )
