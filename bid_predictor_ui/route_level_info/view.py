from dash import html, dcc
import dash_table
import os

money = dash_table.FormatTemplate.money(2)
percentage = dash_table.FormatTemplate.percentage(2).nully('NaN')

# CSS for sorting indicators
SORTING_STYLES = """
<style>
/* Make headers look clickable */
.react-grid-HeaderCell {
    cursor: pointer;
    transition: background-color 0.2s ease;
}

.react-grid-HeaderCell:hover {
    background-color: #e8e8e8;
}

/* Sorting indicator styles */
.react-grid-HeaderCell.sortable::after {
    content: '⇅';
    opacity: 0.3;
    margin-left: 4px;
    font-size: 12px;
}

.react-grid-HeaderCell.sort-ascending::after {
    content: '↑';
    opacity: 1;
    color: #2E86AB;
    font-weight: bold;
}

.react-grid-HeaderCell.sort-descending::after {
    content: '↓';
    opacity: 1;
    color: #2E86AB;
    font-weight: bold;
}

/* Sorted column header highlight */
.react-grid-HeaderCell.sorted {
    background-color: #D9E8F7 !important;
    font-weight: 600;
}
</style>
"""

BASE_COLUMNS = [
    {"name": "Route", "id": "route"},
    {"name": "Submitted Offers", "id": "total_submitted_offers", "type": "numeric"},
    {"name": "Offers ($)", "id": "offers_usd", "type": "numeric", "format": money},
    {"name": "Upgraded Offers", "id": "total_upgraded_offers", "type": "numeric"},
    {"name": "Upgrades ($)", "id": "upgrades_usd", "type": "numeric", "format": money},
    {"name": "Acceptance Rate (%)", "id": "acceptance_rate", "type": "numeric", "format": percentage},
]

HORIZON_COLUMNS = {
    72: [
        # {"name": "Offer Count 72h", "id": "offer_count_72h", "type": "numeric"},
        # {"name": "Accepted Count 72h", "id": "num_actual_ticketed_72h", "type": "numeric"},
        {"name": "Expired Count 72h", "id": "num_actual_expired_72h", "type": "numeric"},
        {"name": "BSP Expired 72h", "id": "expiry_72h", "type": "numeric"},
        {"name": "False -ve 72h", "id": "num_wrongly_expired_72h", "type": "numeric"},
        {"name": "Precision 72h", "id": "negative_precision_72h", "type": "numeric", "format": percentage},
        {"name": "True +ve 72h", "id": "negative_recall_72h", "type": "numeric", "format": percentage},
    ],
    48: [
        # {"name": "Offer Count 48h", "id": "offer_count_48h", "type": "numeric"},
        # {"name": "Accepted Count 48h", "id": "num_actual_ticketed_48h", "type": "numeric"},
        {"name": "Expired Count 48h", "id": "num_actual_expired_48h", "type": "numeric"},
        {"name": "BSP Expired 48h", "id": "expiry_48h", "type": "numeric"},
        {"name": "False -ve 48h", "id": "num_wrongly_expired_48h", "type": "numeric"},
        {"name": "Precision 48h", "id": "negative_precision_48h", "type": "numeric", "format": percentage},
        {"name": "True +ve 48h", "id": "negative_recall_48h", "type": "numeric", "format": percentage},
    ],
    24: [
        # {"name": "Offer Count 24h", "id": "offer_count_24h", "type": "numeric"},
        # {"name": "Accepted Count 24h", "id": "num_actual_ticketed_24h", "type": "numeric"},
        {"name": "Expired Count 24h", "id": "num_actual_expired_24h", "type": "numeric"},
        {"name": "BSP Expired 24h", "id": "expiry_24h", "type": "numeric"},
        {"name": "False -ve 24h", "id": "num_wrongly_expired_24h", "type": "numeric"},
        {"name": "Precision 24h", "id": "negative_precision_24h", "type": "numeric", "format": percentage},
        {"name": "True +ve 24h", "id": "negative_recall_24h", "type": "numeric", "format": percentage},
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
                            "width": "220px",
                            "minWidth": "220px",
                            "borderRight": "1px solid #ddd",
                            "paddingRight": "1rem",
                            "flexShrink": 0,
                        },
                        children=[
                            html.H4("Filters"),
                            html.Label("Carrier"),
                            dcc.Dropdown(
                                id="carrier-dropdown",
                                style={"marginTop": "2px", "display": "block"}, 
                                placeholder="Select carrier",
                                clearable=False,
                            ),
                            html.Label("Hours before departure", style={"marginTop": "6px", "display": "block"}),
                            dcc.Dropdown(
                                id="horizon-dropdown",
                                style={"marginTop": "2px"}, 
                                options=[
                                    {"label": "72 Hours", "value": 72},
                                    {"label": "48 Hours", "value": 48},
                                    {"label": "24 Hours", "value": 24},
                                ],
                                value=72,  # default
                                clearable=False,
                            ),
                            html.Div(
                                id="threshold-display",
                                style={
                                    "marginTop": "12px",
                                    "fontSize": "13px",
                                    "fontWeight": "500",
                                    "color": "#000000",
                                    "padding": "6px 10px",
                                    "display": "inline-block",
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
                            ),
                            html.Div(
                                style={
                                    "marginTop": "14px",
                                    "padding": "10px",
                                    "border": "1px solid #ddd",
                                    "borderRadius": "6px",
                                    "backgroundColor": "#fafafa",
                                    "fontSize": "12px",
                                    "lineHeight": "1.4",
                                },
                                children=[
                                    html.H4(
                                        "📘 Column Legend",
                                        style={
                                            "marginBottom": "6px",
                                            "fontSize": "14px",
                                            "fontWeight": "600",
                                            "textAlign": "center",
                                        },
                                    ),

                                    # ---- Base columns ----
                                    # html.B("Base Metrics"),
                                    html.Ul(
                                        style={"paddingLeft": "16px", "marginTop": "4px"},
                                        children=[
                                            html.Li("Route — Flight route (Origin → Destination)"),
                                            html.Li("Submitted Offers — Total number of offers submitted"),
                                            html.Li("Offers ($) — Total value of submitted offers"),
                                            html.Li("Upgraded Offers — Total number of accepted offers"),
                                            html.Li("Upgrades ($) — Total value of accepted offers"),
                                            html.Li("Acceptance Rate (%) — Percentage of offers that were accepted"),
                                        ],
                                    ),

                                    html.Hr(style={"margin": "8px 0"}),

                                    # ---- Horizon columns ----
                                    # html.B("Horizon-Specific Metrics"),
                                    html.P(
                                        "Shown for the selected hours before departure (72h / 48h / 24h).",
                                        style={"marginBottom": "6px"},
                                    ),

                                    html.Ul(
                                        style={"paddingLeft": "16px"},
                                        children=[
                                            html.Li("Expired Count — Number of offers that actually expired"),
                                            html.Li("BSP Expired — Number of offers the model predicts will expire based on the probability threshold"),
                                            html.Li("False -ve — Offers predicted to expire by the model that were actually ticketed"),
                                            html.Li("Precision — Percentage of model expired offers that were truly expired"),
                                            html.Li("True +ve — Proportion of actually expired offers that the model successfully identifies as expired"),
                                        ],
                                    ),
                                ],
                            ),

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
                                    html.Div(id="audit-status", style={"display": "none"}),
                                    dash_table.DataTable(
                                        id="routes-table",

                                        columns=BASE_COLUMNS + HORIZON_COLUMNS[72],

                                        # Enable native sorting - users can click column headers
                                        sort_action="native",

                                        # fixed_rows={'headers': True},
                                        fixed_columns={"headers": True, "data": 1},

                                        style_table={
                                            "width": "max-content", 
                                            # "minWidth": "100%",
                                            # "overflowX": "scroll",
                                            # 'overflowY': 'scroll',
                                            "minWidth": "100%",
                                            "overflowX": "auto",
                                            "overflowY": "auto",
                                        },

                                        style_cell={
                                            "whiteSpace": "nowrap",
                                            "textAlign": "left",
                                            "padding": "6px 10px",
                                            "fontSize": "11.5px",
                                            "minWidth": "90px",
                                            "maxWidth": "120px",
                                            "verticalAlign": "middle",
                                            # "width": "90px",
                                            "height": "34px",
                                            "lineHeight": "1.2",
                                        },

                                        style_header={
                                            "whiteSpace": "normal",
                                            "backgroundColor": "#f7f7f7",
                                            "fontWeight": "600",
                                            "borderBottom": "1px solid #ccc",
                                            "backgroundColor": "#f7f7f7",
                                            "height": "48px",
                                            "lineHeight": "1.5",
                                            "textAlign": "center",
                                            "verticalAlign": "middle",
                                            "cursor": "pointer",
                                        },

                                        style_header_conditional=[
                                            {
                                                "if": {
                                                    "column_id": [
                                                        "total_submitted_offers",
                                                        "offers_usd",
                                                        "total_upgraded_offers",
                                                        "upgrades_usd",
                                                        "acceptance_rate",
                                                    ]
                                                },
                                                "backgroundColor": "#D9E8F7",
                                            }
                                        ],

                                        style_data={
                                            "borderBottom": "1px solid #eee",
                                        },

                                        style_cell_conditional=[
                                            {"if": {"column_id": "route"},"textAlign": "left", "minWidth": "90px"},
                                            {"if": {"column_id": "offers_usd"}, "minWidth": "85px"},
                                            {"if": {"column_id": "upgrades_usd"}, "minWidth": "80px"},
                                            {"if": {"column_id": "acceptance_rate"}, "minWidth": "70px"},
                                            # {"if": {"column_id": "expiry_72h"}, "maxWidth": "70px"},
                                            
                                            {
                                                'if': {'row_index': 'odd'}, # 'odd' or 'even' also works for simple cases
                                                'backgroundColor': 'rgb(248, 248, 248)' # A light grey
                                            },
                                            {
                                                'if': {'row_index': 'even'},
                                                'backgroundColor': 'white' # Or another color for even rows
                                            },
                                            # {
                                            #     "if": {
                                            #         "column_id": [
                                            #             "total_submitted_offers",
                                            #             "offers_usd",
                                            #             "total_upgraded_offers",
                                            #             "upgrades_usd",
                                            #             "acceptance_rate",
                                            #         ]
                                            #     },
                                            #     "backgroundColor": "#F6F9FC",
                                            #     "fontWeight": "500",
                                            # },                                      
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
