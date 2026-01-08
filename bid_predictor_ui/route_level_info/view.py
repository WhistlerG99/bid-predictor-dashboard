from dash import html, dcc
import dash_table
import os

ACCEPT_PROB_THRESHOLD = float(os.environ.get("ACCEPT_PROB_THRESHOLD", 0.2))

BASE_COLUMNS = [
    {"name": "Route", "id": "route"},
    {"name": "Offers ($)", "id": "offers_usd", "type": "numeric"},
    {"name": "Upgrades ($)", "id": "upgrades_usd", "type": "numeric"},
    {"name": "Acceptance Rate (%)", "id": "acceptance_rate", "type": "numeric"},
]

HORIZON_COLUMNS = {
    72: [
        {"name": "Offer Count 72h", "id": "offer_count_72h", "type": "numeric"},
        {"name": "Accepted Count 72h", "id": "num_actual_ticketed_72h", "type": "numeric"},
        {"name": "Expired Count 72h", "id": "num_actual_expired_72h", "type": "numeric"},
        {"name": "BSP Expired 72h", "id": "expiry_72h", "type": "numeric"},
        {"name": "False -ve 72h", "id": "num_wrongly_expired_72h", "type": "numeric"},
        {"name": "Accuracy 72h", "id": "negative_precision_72h", "type": "numeric"},
        {"name": "True +ve 72h", "id": "negative_recall_72h", "type": "numeric"},
    ],
    48: [
        {"name": "Offer Count 48h", "id": "offer_count_48h", "type": "numeric"},
        {"name": "Accepted Count 48h", "id": "num_actual_ticketed_48h", "type": "numeric"},
        {"name": "Expired Count 48h", "id": "num_actual_expired_48h", "type": "numeric"},
        {"name": "BSP Expired 48h", "id": "expiry_48h", "type": "numeric"},
        {"name": "False -ve 48h", "id": "num_wrongly_expired_48h", "type": "numeric"},
        {"name": "Accuracy 48h", "id": "negative_precision_48h", "type": "numeric"},
        {"name": "True +ve 48h", "id": "negative_recall_48h", "type": "numeric"},
    ],
    24: [
        {"name": "Offer Count 24h", "id": "offer_count_24h", "type": "numeric"},
        {"name": "Accepted Count 24h", "id": "num_actual_ticketed_24h", "type": "numeric"},
        {"name": "Expired Count 24h", "id": "num_actual_expired_24h", "type": "numeric"},
        {"name": "BSP Expired 24h", "id": "expiry_24h", "type": "numeric"},
        {"name": "False -ve 24h", "id": "num_wrongly_expired_24h", "type": "numeric"},
        {"name": "Accuracy 24h", "id": "negative_precision_24h", "type": "numeric"},
        {"name": "True +ve 24h", "id": "negative_recall_24h", "type": "numeric"},
    ],
}

def build_route_level_info_tab():
    return dcc.Tab(
        label="Audit Data",
        value="audit",
        children=[
            dcc.Store(id="audit-data-store"),

            dcc.Interval(
                id="audit-loader-once",
                interval=500,
                n_intervals=0,
                max_intervals=1,
            ),

            html.Div(
                id="audit-tab-content",
                style={
                    "display": "flex",
                    "gap": "1rem",
                    "padding": "1rem",
                    "minHeight": "700px",
                },
                children=[
                    html.Div(
                        style={
                            "width": "130px",
                            "borderRight": "1px solid #ddd",
                            "paddingRight": "1rem",
                            "flexShrink": 0,
                        },
                        children=[
                            html.H4("Filters"),
                            html.Label("Carrier"),
                            dcc.Dropdown(
                                id="carrier-dropdown",
                                placeholder="Select carrier",
                                clearable=False,
                            ),
                            html.Label("Hours before departure"),
                            dcc.Dropdown(
                                id="horizon-dropdown",
                                options=[
                                    {"label": "72 Hours", "value": 72},
                                    {"label": "48 Hours", "value": 48},
                                    {"label": "24 Hours", "value": 24},
                                ],
                                value=72,  # default
                                clearable=False,
                            ),
                            html.Div(
                                f"Acceptance Probability Threshold: {ACCEPT_PROB_THRESHOLD:.2f}",
                                style={
                                    "marginTop": "12px",
                                    "fontSize": "13px",
                                    "fontWeight": "500",
                                    "color": "#000000",
                                    # "backgroundColor": "#f0f6ff",
                                    "padding": "6px 10px",
                                    # "borderRadius": "4px",
                                    "display": "inline-block",
                                    # "border": "5px solid #d0e3ff",
                                },
                            ),
                            html.Div(
                                children=[
                                    html.Hr(style={"margin": "10px 0"}),  # subtle separator
                                    html.H4(
                                        "📊 Last 7 Days of data",
                                        style={
                                            "textAlign": "center",
                                            "color": "#2E86AB",
                                            "marginBottom": "5px",
                                            "marginTop": "10px",
                                            "fontWeight": "500",
                                            "fontSize": "16px",
                                        },
                                    ),
                                    html.P(
                                        "Only includes flights that have already departed.",
                                        style={
                                            "textAlign": "center",
                                            "color": "#555",
                                            "fontSize": "14px",
                                            "marginTop": "0px",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                ]
                            )
                        ],
                    ),

                    # -------------------------
                    # TABLE CONTAINER
                    # -------------------------
                    html.Div(
                        style={
                            "flex": 1,
                            "minWidth": 0,           
                            "overflowX": "auto",   
                        },
                        children=[
                            html.H3("Routes by Carrier"),

                            # Wrap table and status in Loading
                            dcc.Loading(
                                id="audit-loading",
                                type="circle",
                                children=[
                                    html.Div(id="audit-status"),
                                    dash_table.DataTable(
                                        id="routes-table",

                                        columns=BASE_COLUMNS + HORIZON_COLUMNS[72],

                                        fixed_columns={"headers": True, "data": 1},

                                        style_table={
                                            "width": "max-content", 
                                            "minWidth": "100%",
                                            "overflowX": "auto",
                                        },

                                        style_cell={
                                            "whiteSpace": "nowrap",
                                            "textAlign": "left",
                                            "padding": "6px 10px",
                                            "fontSize": "11.5px",
                                            # "maxWidth": "120px",
                                        },

                                        style_header={
                                            "whiteSpace": "normal",
                                            "backgroundColor": "#f7f7f7",
                                            "fontWeight": "600",
                                            "borderBottom": "1px solid #ccc",
                                            "backgroundColor": "#f7f7f7",
                                            "height": "auto",
                                            "lineHeight": "1.2",
                                            "textAlign": "center",
                                        },

                                        style_data={
                                            "borderBottom": "1px solid #eee",
                                        },

                                        style_cell_conditional=[
                                            {"if": {"column_id": "route"}, "minWidth": "85px"},
                                            {"if": {"column_id": "offers_usd"}, "minWidth": "85px"},
                                            {"if": {"column_id": "upgrades_usd"}, "minWidth": "80px"},
                                            {"if": {"column_id": "acceptance_rate"}, "minWidth": "90px"},
                                            # {"if": {"column_id": "expiry_72h"}, "maxWidth": "70px"},
                                        ],
                                        page_action="none"
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )
