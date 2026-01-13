# metrics.py
import pandas as pd
import numpy as np


HORIZONS = {
    "72h": 72,
    "48h": 48,
    "24h": 24,
}

TOLERANCE_HOURS = 1

AUCTION_COLS = [
    "carrier_code",
    "flight_number",
    "travel_date",
    "upgrade_type",
]

def _select_nearest_snapshot(df: pd.DataFrame, target_hours: int) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # --- Notebook-aligned hours before departure
    df["hrs_before_dep"] = df["days_before_departure"].astype(float) * 24.0
    df = df[df["hrs_before_dep"] >= 0]

    # Drop rows with missing keys
    df = df.dropna(subset=AUCTION_COLS + ["hrs_before_dep", "current_timestamp"])

    # --- Snapshot candidates
    snapshots = (
        df[AUCTION_COLS + ["current_timestamp", "hrs_before_dep"]]
        .drop_duplicates(AUCTION_COLS + ["current_timestamp"])
    )
    auctions = snapshots[AUCTION_COLS].drop_duplicates()

    # Add target_hours_before_dep column
    target_df = auctions.merge(
        pd.DataFrame({"target_hours_before_dep": [float(target_hours)]}),
        how="cross"
    )

    target_df["hrs_before_dep"] = target_df["target_hours_before_dep"].astype(float)

    sort_cols = ["hrs_before_dep"] + AUCTION_COLS
    snapshots = snapshots.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    target_df = target_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # --- Find nearest snapshot
    chosen = pd.merge_asof(
        target_df,
        snapshots,
        by=AUCTION_COLS,
        on="hrs_before_dep",
        direction="nearest",
    )

    result = df.merge(
        chosen[AUCTION_COLS + ["target_hours_before_dep", "current_timestamp"]],
        on=AUCTION_COLS + ["current_timestamp"],
        how="inner"
    )

    result["abs_error_hours"] = (result["hrs_before_dep"] - result["target_hours_before_dep"]).abs()
    result = result[result["abs_error_hours"] <= TOLERANCE_HOURS]

    return result


def compute_bucket_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """
    Compute metrics per horizon, including total unique offer IDs.
    """
    results = {}

    df = df[df["offer_status"].isin(["TICKETED", "EXPIRED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"])]

    if df.empty:
        for label in HORIZONS:
            results[label] = {
                "offers_usd": 0.0,
                "upgrades_usd": 0.0,
                "acceptance_rate": 0.0,                
                "offer_count": 0,
                "num_actual_ticketed": 0,
                "num_actual_expired": 0,
                "num_predicted_expired": 0,
                "num_wrongly_expired": 0,
                "negative_precision": 0.0,
                "negative_recall": 0.0,
                "num_unique_offers": 0,
            }
        return results

    for label, target_hours in HORIZONS.items():
        snap_df = _select_nearest_snapshot(df, target_hours)

        if snap_df.empty:
            results[label] = {
                "offers_usd": 0.0,
                "upgrades_usd": 0.0,
                "acceptance_rate": 0.0,
                "offer_count": 0,
                "num_actual_ticketed": 0,
                "num_actual_expired": 0,
                "num_predicted_expired": 0,
                "num_wrongly_expired": 0,
                "negative_precision": 0.0,
                "negative_recall": 0.0,
                "num_unique_offers": 0,  # NEW
            }
            continue

        snap_df = (
            snap_df.sort_values("current_timestamp")
            .groupby("offer_id", as_index=False)
            .last()
        )

        snap_df["actual_ticketed"] = snap_df["offer_status"].isin(
            ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
        )
        snap_df["actual_expired"] = snap_df["offer_status"] == "EXPIRED"
        snap_df["predicted_expired"] = snap_df["accept_prob"] < threshold

        offers_usd = (snap_df["usd_base_amount"]*snap_df["item_count"]).sum()
        upgrades_usd = (snap_df[snap_df.actual_ticketed]["usd_base_amount"]*snap_df[snap_df.actual_ticketed]["item_count"]).sum()

        offer_count = len(snap_df)
        num_actual_ticketed = int(snap_df["actual_ticketed"].sum())

        acceptance_rate = (
            upgrades_usd / offers_usd if offers_usd else np.nan
        )

        num_actual_expired = int(snap_df["actual_expired"].sum())
        num_model_expired = int(snap_df["predicted_expired"].sum())
        num_wrongly_expired = int(
            (snap_df["predicted_expired"] & snap_df["actual_ticketed"]).sum()
        )
        negative_precision = (
            1 - (num_wrongly_expired / num_model_expired) if num_model_expired > 0 else np.nan
        )
        negative_recall = (
            int((snap_df["predicted_expired"] & snap_df["actual_expired"]).sum()) / num_actual_expired
            if num_actual_expired > 0 else 1.0
        )

        results[label] = {
            "offers_usd": round(offers_usd,2),
            "upgrades_usd": round(upgrades_usd,2),
            "acceptance_rate": round(acceptance_rate,4),
            "offer_count": offer_count,
            "num_actual_ticketed": num_actual_ticketed,
            "num_actual_expired": num_actual_expired,
            "num_predicted_expired": num_model_expired,
            "num_wrongly_expired": num_wrongly_expired,
            "negative_precision": round(negative_precision, 4),
            "negative_recall": round(negative_recall, 4),
        }

    return results
