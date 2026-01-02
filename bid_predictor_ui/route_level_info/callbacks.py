from dash import Input, Output, State
import pandas as pd
import os

from .data_loader import load_audit_data_cached
from .redshift_loader import load_offer_statuses_cached
from .metrics import compute_horizon_metrics

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
            return None, "No audit data found"
        return df.to_dict("records"), f"Loaded {len(df):,} audit rows"

    @app.callback(
        Output("carrier-dropdown", "options"),
        Input("audit-data-store", "data"),
    )
    def populate_carriers(data):
        if not data:
            return []
        df = pd.DataFrame(data)
        carriers = sorted(df["carrier_code"].dropna().unique())
        return [{"label": c, "value": c} for c in carriers]

    @app.callback(
        Output("routes-table", "data"),
        Input("carrier-dropdown", "value"),
        State("audit-data-store", "data"),
    )
    def update_routes_table(carrier, data):
        if not carrier or not data:
            return []

        df = pd.DataFrame(data)
        df = df[df["carrier_code"] == carrier]

        if df.empty:
            return []

        df["departure_timestamp"] = pd.to_datetime(df["departure_timestamp"])
        df["accept_prob_timestamp"] = pd.to_datetime(df["accept_prob_timestamp"])
        df["route"] = df["origination_code"] + "-" + df["destination_code"]

        # Load status for all offers
        offer_ids = df["offer_id"].unique().tolist()
        status_df = load_offer_statuses_cached(offer_ids)

        # Ensure offer_status exists
        # if "offer_status" not in status_df.columns:
        #     status_df["offer_status"] = pd.NA

        # Merge status into df for per-route metrics
        df = df.merge(status_df, on="offer_id", how="left")

        rows = []

        for route, route_df in df.groupby("route"):
            # Deduplicate latest per offer for status-based metrics
            offers_status = (
                route_df.sort_values("accept_prob_timestamp")
                .groupby("offer_id", as_index=False)
                .last()
            )

            # Ensure offer_status exists to prevent KeyError
            # if "offer_status" not in offers_status.columns:
            #     offers_status["offer_status"] = pd.NA

            offer_count = len(offers_status)

            # Status-based metrics
            accepted_mask = offers_status["offer_status"].isin(
                ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
            )
            accepted_count = accepted_mask.sum()
            expiry_count = offer_count - accepted_count
            upgrades_usd = offers_status.loc[accepted_mask, "usd_base_amount"].sum()
            offers_usd = offers_status["usd_base_amount"].sum()
            acceptance_rate = (upgrades_usd / offers_usd * 100) if offers_usd > 0 else 0.0

            # Horizon metrics (False Negatives, Accuracy, Expiry @Xh)
            horizon_metrics = compute_horizon_metrics(
                route_df,
                ACCEPT_PROB_THRESHOLD,
            )


            rows.append(
                {
                    "route": route,
                    "offers_usd": round(offers_usd, 2),
                    "upgrades_usd": round(upgrades_usd, 2),
                    "offer_count": int(offer_count),
                    "acceptance_rate": round(acceptance_rate, 2),
                    "accepted": int(accepted_count),
                    "expiry": int(expiry_count),

                    "false_negatives_72h": horizon_metrics["72h"]["false_negatives"],
                    "false_negatives_48h": horizon_metrics["48h"]["false_negatives"],
                    "false_negatives_24h": horizon_metrics["24h"]["false_negatives"],

                    "accuracy_rate_72h": horizon_metrics["72h"]["accuracy"],
                    "accuracy_rate_48h": horizon_metrics["48h"]["accuracy"],
                    "accuracy_rate_24h": horizon_metrics["24h"]["accuracy"],

                    "expiry_48h": horizon_metrics["48h"]["expiry_horizon"],
                    "expiry_24h": horizon_metrics["24h"]["expiry_horizon"],
                }
            )

        return rows
