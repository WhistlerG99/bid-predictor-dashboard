from dash import Input, Output, State
import pandas as pd
import os

from .data_loader import load_audit_data_cached
from .redshift_loader import load_offer_statuses_cached
from .metrics import compute_bucket_metrics

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

        # Only store list of carriers (small) instead of full 450K rows
        carriers = sorted(df["carrier_code"].dropna().unique())
        return {"status": "loaded", "carriers": carriers}, f"Loaded {len(df):,} audit rows"


    @app.callback(
        Output("carrier-dropdown", "options"),
        Input("audit-data-store", "data"),
    )
    def populate_carriers(data):
        if not data:
            return []
        # df = pd.DataFrame(data)
        # carriers = sorted(df["carrier_code"].dropna().unique())
        # return [{"label": c, "value": c} for c in carriers]
        return [{"label": c, "value": c} for c in data["carriers"]]

    @app.callback(
        Output("routes-table", "data"),
        Input("carrier-dropdown", "value"),
        State("audit-data-store", "data"),
    )
    def update_routes_table(carrier, data):
        if not carrier:
            return []

        # 1️⃣ Try precomputed route metrics cache first
        cached_rows = get_cached_route_metrics(carrier, ACCEPT_PROB_THRESHOLD)
        if cached_rows is not None:
            return cached_rows

        # 2️⃣ Load raw data server-side (internal), NOT from frontend
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
            offers_status = (
                route_df.sort_values("accept_prob_timestamp")
                .groupby("offer_id", as_index=False)
                .last()
            )

            filtered_offers = offers_status[
                offers_status["offer_status"].isin(
                    ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY", "EXPIRED"]
                )
            ]

            accepted_mask = offers_status["offer_status"].isin(
                ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
            )

            offer_count = len(filtered_offers)
            accepted_count = accepted_mask.sum()
            # expiry_count = offer_count - accepted_count
            expiry_mask = offers_status["offer_status"] == "EXPIRED"
            expiry_count = expiry_mask.sum()

            upgrades_usd = offers_status.loc[accepted_mask, "usd_base_amount"].sum()
            offers_usd = offers_status["usd_base_amount"].sum()
            acceptance_rate = (upgrades_usd / offers_usd * 100) if offers_usd else 0.0

            horizon = compute_bucket_metrics(route_df, ACCEPT_PROB_THRESHOLD)

            rows.append({
                "route": route,
                "offers_usd": round(offers_usd, 2),
                "upgrades_usd": round(upgrades_usd, 2),
                "offer_count": int(offer_count),
                "acceptance_rate": round(acceptance_rate, 2),
                "accepted": int(accepted_count),
                "expiry": int(expiry_count),

                # "expiry_72h": horizon["72h"]["expiry_horizon"],
                "num_wrongly_expired_72h": horizon["72h"]["num_wrongly_expired"],
                # "percent_wrongly_expired_72h": horizon["72h"]["percent_wrongly_expired"],
                "negative_precision_72h": horizon["72h"]["negative_precision"],
                "negative_recall_72h": horizon["72h"]["negative_recall"],

                # "expiry_48h": horizon["48h"]["expiry_horizon"],
                "num_wrongly_expired_48h": horizon["48h"]["num_wrongly_expired"],
                # "percent_wrongly_expired_48h": horizon["48h"]["percent_wrongly_expired"],
                "negative_precision_48h": horizon["48h"]["negative_precision"],
                "negative_recall_48h": horizon["48h"]["negative_recall"],

                # "expiry_24h": horizon["24h"]["expiry_horizon"],
                "expiry_72h": horizon["72h"]["num_predicted_expired"],
                "expiry_48h": horizon["48h"]["num_predicted_expired"],
                "expiry_24h": horizon["24h"]["num_predicted_expired"],

                "num_wrongly_expired_24h": horizon["24h"]["num_wrongly_expired"],
                # "percent_wrongly_expired_24h": horizon["24h"]["percent_wrongly_expired"],
                "negative_precision_24h": horizon["24h"]["negative_precision"],
                "negative_recall_24h": horizon["24h"]["negative_recall"],
            })

        rows_df = pd.DataFrame(rows)
        rows_df = rows_df.sort_values("offer_count", ascending=False)
        rows = rows_df.to_dict("records")

        # 3️⃣ Cache computed metrics for this carrier
        set_cached_route_metrics(carrier, ACCEPT_PROB_THRESHOLD, rows)

        return rows

