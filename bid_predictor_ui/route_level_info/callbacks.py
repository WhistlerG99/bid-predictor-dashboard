from datetime import datetime
from dash import Input, Output, State, html, dcc, no_update
import pandas as pd
import numpy as np
import os
import json

from .data_loader import load_audit_data_cached
from .redshift_loader import load_offer_statuses_cached
from .metrics import compute_bucket_metrics
from .route_offer_revenue_cache import get_cached_route_offer_revenue
from ..utils.redis_client import get_redis_client

from .view import BASE_COLUMNS, HORIZON_COLUMNS

from .route_metrics_cache import (
    get_cached_route_metrics,
    set_cached_route_metrics,
)

# Final results cache TTL (24 hours)
FINAL_RESULTS_CACHE_TTL = 24 * 3600
SUMMARY_STATS_CACHE_TTL = 24 * 3600  # Summary stats cache (24 hours)

# Carrier-specific thresholds
CARRIER_THRESHOLDS = {
    "EY": float(os.environ.get("ACCEPT_PROB_THRESHOLD_EY", 0.2)),
    "SV": float(os.environ.get("ACCEPT_PROB_THRESHOLD_SV", 0.1)),
}


def get_threshold_for_carrier(carrier: str) -> float:
    """Get the acceptance probability threshold for a specific carrier."""
    return CARRIER_THRESHOLDS.get(carrier, 0.2)  # default to 0.2 if carrier not found


def register_route_level_info_callbacks(app):

    @app.callback(
        Output("audit-data-store", "data"),
        Output("audit-status", "children"),
        Input("audit-loader-once", "n_intervals"),
    )
    def load_audit_once(_):
        df = load_audit_data_cached()
        if df.empty:
            return {"status": "empty", "carriers": []}, "No audit data found"

        carriers = sorted(df["carrier_code"].dropna().unique())
        return (
            {"status": "loaded", "carriers": carriers},
            f"Loaded {len(df):,} audit rows",
        )

    @app.callback(
        Output("carrier-dropdown", "options"),
        Input("audit-data-store", "data"),
    )
    def populate_carriers(data):
        if not data:
            return []
        return [{"label": c, "value": c} for c in data["carriers"]]
    
    @app.callback(
        Output("threshold-display", "children"),
        Input("carrier-dropdown", "value"),
    )
    def update_threshold_display(carrier):
        if not carrier:
            return "Select a carrier to see threshold"
        threshold = get_threshold_for_carrier(carrier)
        print(f"[THRESHOLD DEBUG] Carrier: {carrier}, Threshold: {threshold:.2f}")
        return f"Acceptance Probability Threshold: {threshold:.2f}"
    
    @app.callback(
        Output("routes-table", "columns"),
        Input("horizon-dropdown", "value"),
    )
    def update_table_columns(selected_horizon):
        return BASE_COLUMNS + HORIZON_COLUMNS[selected_horizon]

    @app.callback(
        Output("routes-table", "style_header_conditional"),
        Input("routes-table", "sort_by"),
    )
    def update_header_style(sort_by):
        """Highlight and add arrows to the sorted column header"""
        style_header_conditional = [
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
        ]
        
        # Add highlighting for sorted columns
        if sort_by and len(sort_by) > 0:
            for sort_item in sort_by:
                col_id = sort_item.get("column_id")
                direction = sort_item.get("direction", "asc")
                
                if col_id:
                    arrow = "↑ " if direction == "asc" else "↓ "
                    style_header_conditional.append({
                        "if": {"column_id": col_id},
                        "backgroundColor": "#2E86AB",
                        "color": "white",
                        "fontWeight": "700",
                    })
        
        return style_header_conditional

    @app.callback(
        Output("period-info-title", "children"),
        Input("period-dropdown", "value"),
    )
    def update_period_info(period_days):
        """Update the period info title when period dropdown changes"""
        period_labels = {
            7: "📊 Last 7 Days of data",
            14: "📊 Last 14 Days of data",
            21: "📊 Last 21 Days of data",
            30: "📊 Last 30 Days of data",
        }
        return period_labels.get(period_days, "📊 Last 7 Days of data")

    @app.callback(
        Output("routes-table", "data"),
        Input("carrier-dropdown", "value"),
        Input("period-dropdown", "value"),
        State("audit-data-store", "data"),
    )
    def update_routes_table(carrier, period_days, data):
        if not carrier:
            return []

        # Get carrier-specific threshold
        threshold = get_threshold_for_carrier(carrier)
        print(f"\n[ROUTES TABLE UPDATE] Carrier: {carrier}, Using threshold: {threshold:.2f}, Period: {period_days} days")

        # ✅ STEP 1: CHECK FINAL RESULTS CACHE FIRST (fastest!)
        redis_client = get_redis_client()
        final_cache_key = f"routes_table_final:{carrier}:threshold={threshold}:period={period_days}d"
        
        if redis_client:
            try:
                cached_final_result = redis_client.get(final_cache_key)
                if cached_final_result:
                    print(f"[ROUTES TABLE] 🚀 Final results cache HIT for {carrier} - instant result!")
                    return json.loads(cached_final_result)
            except Exception as e:
                print(f"[ROUTES TABLE] Warning checking final cache: {str(e)}")

        # 1️⃣ Route metrics cache (carrier + threshold + window)
        cached_rows = get_cached_route_metrics(carrier, threshold, period_days)
        if cached_rows is not None:
            print(f"[ROUTES TABLE] Route metrics cache HIT for {carrier} with threshold {threshold:.2f} and period {period_days} days")
            # Cache the final result for next time
            if redis_client:
                try:
                    redis_client.setex(
                        final_cache_key,
                        FINAL_RESULTS_CACHE_TTL,
                        json.dumps(cached_rows)
                    )
                    print(f"[ROUTES TABLE] Stored final results in cache for {carrier}")
                except Exception as e:
                    print(f"[ROUTES TABLE] Warning caching final results: {str(e)}")
            return cached_rows

        print(f"[ROUTES TABLE] Cache MISS for {carrier}, computing metrics with threshold {threshold:.2f}")

        # 2️⃣ Load raw audit data (server-side)
        # Convert period_days (7, 14, 21, 30) to S3 fetch days parameter
        # period_days = lookback_days + 1, so days = period_days - 1
        s3_days_param = period_days - 1
        df = load_audit_data_cached(days=s3_days_param)
        df = df[df["carrier_code"] == carrier]

        if df.empty:
            return []

        # Defensive datetime parsing (metrics no longer use timestamps,
        # but these may still be needed elsewhere)
        if "departure_timestamp" in df.columns:
            df["departure_timestamp"] = pd.to_datetime(df["departure_timestamp"])
        if "accept_prob_timestamp" in df.columns:
            df["accept_prob_timestamp"] = pd.to_datetime(df["accept_prob_timestamp"])

        # Print raw data range before filtering
        if "departure_timestamp" in df.columns:
            min_departure_raw = df["departure_timestamp"].min()
            max_departure_raw = df["departure_timestamp"].max()
            print(f"[DATA BEFORE FILTER] Carrier: {carrier}, Min departure: {min_departure_raw}, Max departure: {max_departure_raw}, Total rows: {len(df)}")

        # Apply dynamic departure_timestamp filter based on selected period
        if "departure_timestamp" in df.columns:
            from datetime import timedelta, time as dt_time
            today = datetime.utcnow().date()
            end_date = today - timedelta(days=1)  # Yesterday
            # period_days is user-selected (7, 14, 21, 30), convert to lookback days
            lookback_days = period_days - 1
            start_date = end_date - timedelta(days=lookback_days)
            
            end_ts = datetime.combine(end_date, dt_time.max)  # 23:59:59
            start_ts = datetime.combine(start_date, dt_time.min)  # 00:00:00
            
            print(f"[DEPARTURE FILTER APPLIED] Carrier: {carrier}, Period: {period_days} days, Filter range: {start_ts} to {end_ts}")
            
            rows_before = len(df)
            df = df[(df["departure_timestamp"] >= start_ts) & (df["departure_timestamp"] <= end_ts)]
            rows_after = len(df)
            rows_dropped = rows_before - rows_after
            
            print(f"[DEPARTURE FILTER RESULT] Rows before: {rows_before}, Rows after: {rows_after}, Rows dropped: {rows_dropped} ({100*rows_dropped/rows_before:.1f}%)" if rows_before > 0 else f"[DEPARTURE FILTER RESULT] No rows to filter")

        if df.empty:
            print(f"[WARNING] Carrier {carrier} has no data after departure_timestamp filter")
            return []

        # Print filtered data range
        if "departure_timestamp" in df.columns:
            min_departure_filtered = df["departure_timestamp"].min()
            max_departure_filtered = df["departure_timestamp"].max()
            print(f"[DATA AFTER FILTER] Carrier: {carrier}, Min departure: {min_departure_filtered}, Max departure: {max_departure_filtered}, Total rows: {len(df)}")
            print(f"[PERIOD VERIFICATION] Carrier: {carrier}, Working with {period_days}-day window data - from {min_departure_filtered.date()} to {max_departure_filtered.date()}")

        # Route key
        df["route"] = df["origination_code"] + "-" + df["destination_code"]

        # 3️⃣ Load offer status (cached / Redshift)
        print(f"[LOADING OFFER STATUS] Carrier: {carrier}, Fetching status for {len(df['offer_id'].dropna().unique())} unique offer IDs")
        offer_ids = df["offer_id"].dropna().unique().tolist()
        status_df = load_offer_statuses_cached(offer_ids)
        print(f"[OFFER STATUS LOADED] Carrier: {carrier}, Got status for {len(status_df)} offers")

        df = df.merge(status_df, on="offer_id", how="left")
        print(f"[MERGE COMPLETE] Carrier: {carrier}, Merged dataframe has {len(df)} rows")

        rows = []
        print(f"[STARTING ROUTE GROUPING] Carrier: {carrier}, Grouping {len(df)} rows by route")

        route_revenue_map = get_cached_route_offer_revenue(
            carrier=carrier,
            start_date=start_date,
            end_date=end_date,
            period_days=period_days,
        )

        for route, route_df in df.groupby("route"):
            route_df_non_cancelled = route_df[
                route_df["offer_status"] != "CANCELLED"
            ]

            final_state = (
                route_df_non_cancelled.sort_values("accept_prob_timestamp")
                        .groupby("offer_id", as_index=False)
                        .last()
            )

            valid_final = final_state[
                final_state["offer_status"].isin(
                    ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY", "EXPIRED"]
                )
            ]

            accepted_mask = valid_final["offer_status"].isin(
                ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
            )

            # offers_usd = (final_state["usd_base_amount"]*final_state["item_count"]).sum()
            # upgrades_usd = (valid_final["usd_base_amount"]*valid_final["item_count"]).loc[accepted_mask].sum()
            
            # num_offers = len(final_state)
            # num_upgrades = len(valid_final.loc[accepted_mask])

            origination, destination = route.split("-")
            route_key = (origination, destination)
            route_revenue = route_revenue_map.get(route_key)

            if route_revenue:
                offers_usd = route_revenue["offers_usd"]
                upgrades_usd = route_revenue["upgrades_usd"]
                num_offers = route_revenue["offer_count"]
                num_upgrades = route_revenue["num_actual_ticketed"]
            else:
                offers_usd = 0.0
                upgrades_usd = 0.0
                num_offers = 0
                num_upgrades = 0

            acceptance_rate = (
                upgrades_usd / offers_usd if offers_usd else np.nan
            )

            horizon = compute_bucket_metrics(route_df, threshold)

            rows.append({
                "route": route,

                # Business metrics (final-state)
                "offers_usd": round(offers_usd, 2),
                "upgrades_usd": round(upgrades_usd, 2),
                "acceptance_rate": round(acceptance_rate, 4),

                "total_submitted_offers": int(num_offers),
                "total_upgraded_offers": int(num_upgrades),

                # "offers_usd_72h": horizon["72h"]["offers_usd"],
                # "offers_usd_48h": horizon["48h"]["offers_usd"],
                # "offers_usd_24h": horizon["24h"]["offers_usd"],

                # "upgrades_usd_72h": horizon["72h"]["upgrades_usd"],
                # "upgrades_usd_48h": horizon["48h"]["upgrades_usd"],
                # "upgrades_usd_24h": horizon["24h"]["upgrades_usd"],

                # "acceptance_rate_72h": horizon["72h"]["acceptance_rate"],
                # "acceptance_rate_48h": horizon["48h"]["acceptance_rate"],
                # "acceptance_rate_24h": horizon["24h"]["acceptance_rate"],

                # Offer counts are PER HORIZON (notebook semantics)
                "offer_count_72h": horizon["72h"]["offer_count"],
                "offer_count_48h": horizon["48h"]["offer_count"],
                "offer_count_24h": horizon["24h"]["offer_count"],

                "num_actual_ticketed_72h": horizon["72h"]["num_actual_ticketed"],
                "num_actual_ticketed_48h": horizon["48h"]["num_actual_ticketed"],
                "num_actual_ticketed_24h": horizon["24h"]["num_actual_ticketed"],

                'num_actual_expired_72h': horizon["72h"]["num_actual_expired"],
                'num_actual_expired_48h': horizon["48h"]["num_actual_expired"],
                'num_actual_expired_24h': horizon["24h"]["num_actual_expired"],

                # Predicted expired per horizon
                "expiry_72h": horizon["72h"]["num_predicted_expired"],
                "expiry_48h": horizon["48h"]["num_predicted_expired"],
                "expiry_24h": horizon["24h"]["num_predicted_expired"],

                # Wrongly expired + precision / recall
                "num_wrongly_expired_72h": horizon["72h"]["num_wrongly_expired"],
                "negative_precision_72h": horizon["72h"]["negative_precision"],
                "negative_recall_72h": horizon["72h"]["negative_recall"],

                "num_wrongly_expired_48h": horizon["48h"]["num_wrongly_expired"],
                "negative_precision_48h": horizon["48h"]["negative_precision"],
                "negative_recall_48h": horizon["48h"]["negative_recall"],

                "num_wrongly_expired_24h": horizon["24h"]["num_wrongly_expired"],
                "negative_precision_24h": horizon["24h"]["negative_precision"],
                "negative_recall_24h": horizon["24h"]["negative_recall"],
            })

        print(f"[ROUTES LOOP COMPLETE] Carrier: {carrier}, Built {len(rows)} route rows")
        
        rows_df = pd.DataFrame(rows)
        print(f"[DATAFRAME CREATED] Carrier: {carrier}, DataFrame shape: {rows_df.shape}")

        # No automatic sorting - users can sort by clicking column headers
        print(f"[DATA READY FOR DISPLAY] Carrier: {carrier}, Total routes: {len(rows_df)}, sorted by insertion order")

        rows = rows_df.to_dict("records")
        print(f"[CONVERTED TO RECORDS] Carrier: {carrier}, Ready to cache {len(rows)} records")

        # 4️⃣ Cache computed route metrics (in insertion order)
        set_cached_route_metrics(carrier, threshold, period_days, rows)
        print(f"[CACHE STORED] Carrier: {carrier}, Period: {period_days}d, Metrics cached for {len(rows)} routes")

        # ✅ STEP 5: CACHE THE FINAL RESULTS (for instant future loads)
        if redis_client:
            try:
                redis_client.setex(
                    final_cache_key,
                    FINAL_RESULTS_CACHE_TTL,
                    json.dumps(rows)
                )
                print(f"[ROUTES TABLE] Stored final results in cache for {carrier} - {len(rows)} routes")
            except Exception as e:
                print(f"[ROUTES TABLE] Warning caching final results: {str(e)}")

        print(f"[RETURNING RESULTS] Carrier: {carrier}, Returning {len(rows)} rows to display - users can sort by clicking column headers")
        return rows

    def cache_summary_stats(carrier, period_days, selected_horizon, summary_data):
        """Helper function to cache summary statistics"""
        redis_client = get_redis_client()
        if not redis_client:
            return
        
        try:
            summary_cache_key = f"summary_stats:{carrier}:period={period_days}d:horizon={selected_horizon}h"
            redis_client.setex(
                summary_cache_key,
                SUMMARY_STATS_CACHE_TTL,
                json.dumps(summary_data)
            )
            print(f"[SUMMARY STATS] Cached summary for {carrier}, period {period_days}d, horizon {selected_horizon}h")
        except Exception as e:
            print(f"[SUMMARY STATS] Warning caching summary: {str(e)}")

    @app.callback(
        Output("summary-stats-content", "children"),
        Output("summary-stats-store", "data"),
        Input("routes-table", "data"),
        Input("horizon-dropdown", "value"),
        State("carrier-dropdown", "value"),
        State("period-dropdown", "value"),
    )
    def update_summary_stats(table_data, selected_horizon, carrier, period_days):
        """Calculate and display summary statistics from the routes table with caching"""
        if not table_data:
            return html.P("No data available", style={"textAlign": "center", "color": "#999"}), None

        try:
            df = pd.DataFrame(table_data)
            
            # Calculate totals for base metrics
            total_submitted = df["total_submitted_offers"].sum()
            total_offers_usd = df["offers_usd"].sum()
            total_upgraded = df["total_upgraded_offers"].sum()
            total_upgrades_usd = df["upgrades_usd"].sum()

            # Calculate totals for selected horizon
            horizon_columns = {
                72: {"expired": "num_actual_expired_72h", "bsp": "expiry_72h", "false_neg": "num_wrongly_expired_72h"},
                48: {"expired": "num_actual_expired_48h", "bsp": "expiry_48h", "false_neg": "num_wrongly_expired_48h"},
                24: {"expired": "num_actual_expired_24h", "bsp": "expiry_24h", "false_neg": "num_wrongly_expired_24h"},
            }
            
            horizon_cols = horizon_columns.get(selected_horizon, horizon_columns[72])
            total_expired = df[horizon_cols["expired"]].sum() if horizon_cols["expired"] in df.columns else 0
            total_bsp_expired = df[horizon_cols["bsp"]].sum() if horizon_cols["bsp"] in df.columns else 0
            total_false_neg = df[horizon_cols["false_neg"]].sum() if horizon_cols["false_neg"] in df.columns else 0

            # Format currency values
            total_offers_formatted = f"${total_offers_usd:,.2f}"
            total_upgrades_formatted = f"${total_upgrades_usd:,.2f}"

            # Calculate Precision: 100% * (1 - (Total False -ve)/(Total BSP Expired))
            precision = None
            if total_bsp_expired > 0:
                precision = 100 * (1 - (total_false_neg / total_bsp_expired))
            
            # Calculate True +ve: 100% * ((Total BSP Expired) - (Total False -ve))/(Total Expired Count)
            true_positive = None
            if total_expired > 0:
                true_positive = 100 * ((total_bsp_expired - total_false_neg) / total_expired)

            # Store summary data for caching
            summary_data = {
                "total_submitted": int(total_submitted),
                "total_offers_usd": float(total_offers_usd),
                "total_offers_formatted": total_offers_formatted,
                "total_upgraded": int(total_upgraded),
                "total_upgrades_usd": float(total_upgrades_usd),
                "total_upgrades_formatted": total_upgrades_formatted,
                "total_expired": int(total_expired),
                "total_bsp_expired": int(total_bsp_expired),
                "total_false_neg": int(total_false_neg),
                "precision": precision,
                "true_positive": true_positive,
                "selected_horizon": selected_horizon,
            }

            # Cache the summary stats
            cache_summary_stats(carrier, period_days, selected_horizon, summary_data)

            content = [
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "paddingBottom": "4px", "borderBottom": "1px solid #B0BEC5"},
                    children=[
                        html.Span("Total Submitted Offers:", style={"fontWeight": "500"}),
                        html.Span(f"{int(total_submitted)}", style={"fontWeight": "600", "color": "#1565C0"}),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "paddingBottom": "4px", "borderBottom": "1px solid #B0BEC5"},
                    children=[
                        html.Span("Total Offers ($):", style={"fontWeight": "500"}),
                        html.Span(total_offers_formatted, style={"fontWeight": "600", "color": "#1565C0"}),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "paddingBottom": "4px", "borderBottom": "1px solid #B0BEC5"},
                    children=[
                        html.Span("Total Upgraded Offers:", style={"fontWeight": "500"}),
                        html.Span(f"{int(total_upgraded)}", style={"fontWeight": "600", "color": "#1565C0"}),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "paddingBottom": "4px", "borderBottom": "1px solid #B0BEC5"},
                    children=[
                        html.Span("Total Upgrades ($):", style={"fontWeight": "500"}),
                        html.Span(total_upgrades_formatted, style={"fontWeight": "600", "color": "#1565C0"}),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "paddingBottom": "4px", "borderBottom": "1px solid #B0BEC5"},
                    children=[
                        html.Span(f"Total Expired Count ({selected_horizon}h):", style={"fontWeight": "500"}),
                        html.Span(f"{int(total_expired)}", style={"fontWeight": "600", "color": "#1565C0"}),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "paddingBottom": "4px", "borderBottom": "1px solid #B0BEC5"},
                    children=[
                        html.Span(f"Total BSP Expired ({selected_horizon}h):", style={"fontWeight": "500"}),
                        html.Span(f"{int(total_bsp_expired)}", style={"fontWeight": "600", "color": "#1565C0"}),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "paddingBottom": "4px", "borderBottom": "1px solid #B0BEC5"},
                    children=[
                        html.Span(f"Total False -ve ({selected_horizon}h):", style={"fontWeight": "500"}),
                        html.Span(f"{int(total_false_neg)}", style={"fontWeight": "600", "color": "#D32F2F"}),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "paddingBottom": "4px", "borderBottom": "1px solid #B0BEC5"},
                    children=[
                        html.Span(f"Precision ({selected_horizon}h):", style={"fontWeight": "500"}),
                        html.Span(
                            f"{precision:.2f}%" if precision is not None else "N/A",
                            style={"fontWeight": "600", "color": "#1565C0"}
                        ),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "paddingTop": "4px"},
                    children=[
                        html.Span(f"True +ve ({selected_horizon}h):", style={"fontWeight": "500"}),
                        html.Span(
                            f"{true_positive:.2f}%" if true_positive is not None else "N/A",
                            style={"fontWeight": "600", "color": "#1565C0"}
                        ),
                    ],
                ),
            ]

            return content, summary_data
        except Exception as e:
            print(f"[SUMMARY STATS ERROR] {str(e)}")
            return html.P(f"Error calculating summary: {str(e)}", style={"textAlign": "center", "color": "red"}), None

    @app.callback(
        Output("routes-table-download", "data"),
        Input("routes-table-download-button", "n_clicks"),
        State("routes-table", "data"),
        State("routes-table", "columns"),
        State("carrier-dropdown", "value"),
        State("period-dropdown", "value"),
        State("horizon-dropdown", "value"),
        prevent_initial_call=True,
    )
    def download_routes_table(n_clicks, table_data, columns, carrier, period_days, horizon_hours):
        if not n_clicks or not table_data:
            return no_update

        df = pd.DataFrame(table_data)
        if columns:
            column_ids = [col["id"] for col in columns if "id" in col]
            column_names = {col["id"]: col.get("name", col["id"]) for col in columns if "id" in col}
            df = df.reindex(columns=column_ids)
            df = df.rename(columns=column_names)

        safe_carrier = (carrier or "all").lower()
        safe_period = f"{period_days}d" if period_days else "period"
        safe_horizon = f"{horizon_hours}h" if horizon_hours else "hours"
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"bsp-route-metrics-{safe_carrier}-{safe_period}-{safe_horizon}-{timestamp}.csv"
        return dcc.send_data_frame(df.to_csv, filename, index=False)
