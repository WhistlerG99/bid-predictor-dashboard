from datetime import datetime
from dash import Input, Output, State
import pandas as pd
import numpy as np
import os

from .data_loader import load_audit_data_cached
from .redshift_loader import load_offer_statuses_cached
from .metrics import compute_bucket_metrics

from .view import BASE_COLUMNS, HORIZON_COLUMNS

from .route_metrics_cache import (
    get_cached_route_metrics,
    set_cached_route_metrics,
)

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
        Output("routes-table", "data"),
        Input("carrier-dropdown", "value"),
        State("audit-data-store", "data"),
    )
    def update_routes_table(carrier, data):
        if not carrier:
            return []

        # Get carrier-specific threshold
        threshold = get_threshold_for_carrier(carrier)
        print(f"\n[ROUTES TABLE UPDATE] Carrier: {carrier}, Using threshold: {threshold:.2f}")

        # 1️⃣ Route metrics cache (carrier + threshold + window)
        cached_rows = get_cached_route_metrics(carrier, threshold)
        if cached_rows is not None:
            print(f"[ROUTES TABLE] Cache HIT for {carrier} with threshold {threshold:.2f}")
            return cached_rows

        print(f"[ROUTES TABLE] Cache MISS for {carrier}, computing metrics with threshold {threshold:.2f}")

        # 2️⃣ Load raw audit data (server-side)
        df = load_audit_data_cached()
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

        # Apply 7-day departure_timestamp filter: yesterday back to 7 days ago
        if "departure_timestamp" in df.columns:
            from datetime import timedelta, time as dt_time
            today = datetime.utcnow().date()
            end_date = today - timedelta(days=1)  # Yesterday
            start_date = end_date - timedelta(days=6)  # 7 days total
            
            end_ts = datetime.combine(end_date, dt_time.max)  # 23:59:59
            start_ts = datetime.combine(start_date, dt_time.min)  # 00:00:00
            
            print(f"[DEPARTURE FILTER APPLIED] Carrier: {carrier}, Filter range: {start_ts} to {end_ts}")
            
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
            print(f"[7-DAY VERIFICATION] Carrier: {carrier}, Working with 7-day window data ONLY - from {min_departure_filtered.date()} to {max_departure_filtered.date()}")

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

        for route, route_df in df.groupby("route"):
            # print(f"[PROCESSING ROUTE] Carrier: {carrier}, Route: {route}, Rows: {len(route_df)}")

            route_df_non_cancelled = route_df[
                route_df["offer_status"] != "CANCELLED"
            ]
            # print(f"[ROUTE FILTERED] Carrier: {carrier}, Route: {route}, After removing CANCELLED: {len(route_df_non_cancelled)} rows")


            # --- Business / revenue metrics ---
            # These intentionally use final state per offer
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

            offers_usd = (final_state["usd_base_amount"]*final_state["item_count"]).sum()
            upgrades_usd = (valid_final["usd_base_amount"]*valid_final["item_count"]).loc[accepted_mask].sum()
            
            num_offers = len(final_state)
            num_upgrades = len(valid_final.loc[accepted_mask])
            acceptance_rate = (
                upgrades_usd / offers_usd if offers_usd else np.nan
            )

            # --- NOTEBOOK-ALIGNED SNAPSHOT METRICS ---
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
                "negative_precision_72h": horizon["72h"]["negative_precision"] if horizon["72h"]["negative_precision"] else "NaN",
                "negative_recall_72h": horizon["72h"]["negative_recall"],

                "num_wrongly_expired_48h": horizon["48h"]["num_wrongly_expired"],
                "negative_precision_48h": horizon["48h"]["negative_precision"] if horizon["48h"]["negative_precision"] else "NaN",
                "negative_recall_48h": horizon["48h"]["negative_recall"],

                "num_wrongly_expired_24h": horizon["24h"]["num_wrongly_expired"],
                "negative_precision_24h": horizon["24h"]["negative_precision"] if horizon["24h"]["negative_precision"] else "NaN",
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
        set_cached_route_metrics(carrier, threshold, rows)
        print(f"[CACHE STORED] Carrier: {carrier}, Metrics cached for {len(rows)} routes")

        print(f"[RETURNING RESULTS] Carrier: {carrier}, Returning {len(rows)} rows to display - users can sort by clicking column headers")
        return rows