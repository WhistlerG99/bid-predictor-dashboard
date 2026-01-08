from dash import Input, Output, State
import pandas as pd
import os

from .data_loader import load_audit_data_cached
from .redshift_loader import load_offer_statuses_cached
from .route_metrics_cache import (
    get_cached_route_metrics,
    set_cached_route_metrics,
)

ACCEPT_PROB_THRESHOLD = float(os.environ.get("ACCEPT_PROB_THRESHOLD"))


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
        return {"status": "loaded", "carriers": carriers}, f"Loaded {len(df):,} audit rows"

    @app.callback(
        Output("carrier-dropdown", "options"),
        Input("audit-data-store", "data"),
    )
    def populate_carriers(data):
        if not data:
            return []
        return [{"label": c, "value": c} for c in data["carriers"]]

    @app.callback(
        Output("routes-table", "data"),
        Input("carrier-dropdown", "value"),
        State("audit-data-store", "data"),
    )
    def update_routes_table(carrier, data):
        if not carrier:
            return []

        # Check cached metrics first
        cached_rows = get_cached_route_metrics(carrier, ACCEPT_PROB_THRESHOLD)
        if cached_rows is not None:
            return cached_rows

        df = load_audit_data_cached()
        df = df[df["carrier_code"] == carrier]
        if df.empty:
            return []

        df["departure_timestamp"] = pd.to_datetime(df["departure_timestamp"])
        df["accept_prob_timestamp"] = pd.to_datetime(df["accept_prob_timestamp"])
        df["route"] = df["origination_code"] + "-" + df["destination_code"]

        # Load offer status from cache/redshift
        offer_ids = df["offer_id"].unique().tolist()
        status_df = load_offer_statuses_cached(offer_ids)
        df = df.merge(status_df, on="offer_id", how="left")

        rows = []
        for route, route_df in df.groupby("route"):
            # Deduplicate: keep latest accept_prob_timestamp per offer
            offers_status = (
                route_df.sort_values("accept_prob_timestamp")
                        .groupby("offer_id", as_index=False)
                        .last()
            )

            # Filter only relevant statuses
            filtered_offers = offers_status[
                offers_status["offer_status"].isin(
                    ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY", "EXPIRED"]
                )
            ]

            offer_count = len(filtered_offers)
            accepted_mask = filtered_offers["offer_status"].isin(
                ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
            )
            expiry_mask = filtered_offers["offer_status"] == "EXPIRED"

            accepted_count = accepted_mask.sum()
            expiry_count = expiry_mask.sum()

            # USD calculations
            upgrades_usd = filtered_offers.loc[accepted_mask, "usd_base_amount"].sum()
            offers_usd = filtered_offers["usd_base_amount"].sum()
            acceptance_rate = (upgrades_usd / offers_usd * 100) if offers_usd else 0.0

            # Model predictions
            filtered_offers["predicted_expired"] = filtered_offers["accept_prob"] < ACCEPT_PROB_THRESHOLD

            num_predicted_expired = int(filtered_offers["predicted_expired"].sum())
            num_wrongly_expired = int((filtered_offers["predicted_expired"] & accepted_mask).sum())

            negative_precision = (
                1 - (num_wrongly_expired / num_predicted_expired)
                if num_predicted_expired > 0 else 0.0
            )
            negative_recall = (
                int((filtered_offers["predicted_expired"] & expiry_mask).sum()) / expiry_count
                if expiry_count > 0 else 0.0
            )

            rows.append({
                "route": route,
                "offers_usd": round(offers_usd, 2),
                "upgrades_usd": round(upgrades_usd, 2),
                "offer_count": int(offer_count),
                "acceptance_rate": round(acceptance_rate, 2),
                "accepted": int(accepted_count),
                "expiry": int(expiry_count),
                "num_wrongly_expired": int(num_wrongly_expired),
                "negative_precision": round(negative_precision, 3),
                "negative_recall": round(negative_recall, 3),
                "num_predicted_expired": int(num_predicted_expired),
            })

        rows_df = pd.DataFrame(rows)
        rows_df = rows_df.sort_values("offer_count", ascending=False)
        rows = rows_df.to_dict("records")

        # Cache the computed metrics
        set_cached_route_metrics(carrier, ACCEPT_PROB_THRESHOLD, rows)

        return rows
