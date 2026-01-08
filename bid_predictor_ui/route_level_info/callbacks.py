from dash import Input, Output, State
import pandas as pd
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

        # Route key
        df["route"] = df["origination_code"] + "-" + df["destination_code"]

        # 3️⃣ Load offer status (cached / Redshift)
        offer_ids = df["offer_id"].dropna().unique().tolist()
        status_df = load_offer_statuses_cached(offer_ids)

        df = df.merge(status_df, on="offer_id", how="left")

        rows = []

        for route, route_df in df.groupby("route"):

            # --- Business / revenue metrics ---
            # These intentionally use final state per offer
            final_state = (
                route_df.sort_values("accept_prob_timestamp")
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

            offers_usd = valid_final["usd_base_amount"].sum()
            upgrades_usd = valid_final.loc[accepted_mask, "usd_base_amount"].sum()
            acceptance_rate = (
                upgrades_usd / offers_usd * 100 if offers_usd else 0.0
            )

            # --- NOTEBOOK-ALIGNED SNAPSHOT METRICS ---
            horizon = compute_bucket_metrics(route_df, threshold)

            rows.append({
                "route": route,

                # Business metrics (final-state)
                "offers_usd": round(offers_usd, 2),
                "upgrades_usd": round(upgrades_usd, 2),
                "acceptance_rate": round(acceptance_rate, 2),

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

                "num_unique_offers_72h": horizon["72h"]["num_unique_offers"],
                "num_unique_offers_48h": horizon["48h"]["num_unique_offers"],
                "num_unique_offers_24h": horizon["24h"]["num_unique_offers"],
            })

        rows_df = pd.DataFrame(rows)

        # Sort routes by 72h offer volume (closest to decision point)
        rows_df = rows_df.sort_values(
            "offer_count_72h", ascending=False
        )

        rows = rows_df.to_dict("records")

        # 4️⃣ Cache computed route metrics
        set_cached_route_metrics(carrier, threshold, rows)

        return rows
