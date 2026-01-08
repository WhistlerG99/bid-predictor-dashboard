# metrics.py
import pandas as pd

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
    """
    CHANGES MADE:
    1. Added 'target_hours_before_dep' column creation (line 33-35)
    2. Added 'hrs_before_dep' assignment from target (line 36)
    3. Added 'abs_error_hours' calculation (line 53)
    4. Changed tolerance filter to use 'abs_error_hours' (line 56)
    5. Added 'target_hours_before_dep' to the final merge (line 60)
    
    This now matches the notebook's get_snapshots() logic exactly.
    """
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
    
    # CHANGE 1: Add target_hours_before_dep column (like notebook)
    target_df = auctions.merge(
        pd.DataFrame({"target_hours_before_dep": [float(target_hours)]}),
        how="cross"
    )
    # CHANGE 2: Set hrs_before_dep from target (like notebook)
    target_df["hrs_before_dep"] = target_df["target_hours_before_dep"].astype(float)
    
    # CRITICAL: For merge_asof, sort by 'on' key FIRST, then 'by' keys
    # This is different from the notebook but required by pandas merge_asof
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
    
    # CHANGE 3: Calculate absolute error (like notebook)
    chosen["abs_error_hours"] = (chosen["hrs_before_dep"] - chosen["target_hours_before_dep"]).abs()
    
    # CHANGE 4: Filter by tolerance using abs_error_hours (like notebook)
    chosen = chosen[chosen["abs_error_hours"] <= TOLERANCE_HOURS]
    
    # CHANGE 5: Include target_hours_before_dep in merge (like notebook)
    return df.merge(
        chosen[AUCTION_COLS + ["target_hours_before_dep", "current_timestamp"]],
        on=AUCTION_COLS + ["current_timestamp"],
        how="inner"
    )


def compute_bucket_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """
    Notebook-aligned snapshot metrics (PER HORIZON):
    - offer_count
    - num_actual_ticketed
    - num_actual_expired
    - predicted expired
    - wrongly expired
    - negative precision / recall
    
    CHANGES MADE:
    1. Added offer_status filter at the start (matches notebook's get_metrics)
    2. Rest of logic remains the same
    """
    results = {}
    
    # CHANGE: Filter by valid offer statuses BEFORE computing metrics (like notebook)
    df = df[df["offer_status"].isin(["TICKETED", "EXPIRED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"])]
    
    if df.empty:
        for label in HORIZONS:
            results[label] = {
                "offer_count": 0,
                "num_actual_ticketed": 0,
                "num_actual_expired": 0,
                "num_predicted_expired": 0,
                "num_wrongly_expired": 0,
                "negative_precision": 0.0,
                "negative_recall": 0.0,
            }
        return results
    
    for label, target_hours in HORIZONS.items():
        snap_df = _select_nearest_snapshot(df, target_hours)
        
        if snap_df.empty:
            results[label] = {
                "offer_count": 0,
                "num_actual_ticketed": 0,
                "num_actual_expired": 0,
                "num_predicted_expired": 0,
                "num_wrongly_expired": 0,
                "negative_precision": 0.0,
                "negative_recall": 0.0,
            }
            continue
        
        # Ground truth (snapshot-based)
        snap_df["actual_ticketed"] = snap_df["offer_status"].isin(
            ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
        )
        snap_df["actual_expired"] = snap_df["offer_status"] == "EXPIRED"
        
        # Model decision
        snap_df["predicted_expired"] = snap_df["accept_prob"] < threshold
        
        offer_count = len(snap_df)
        num_actual_ticketed = int(snap_df["actual_ticketed"].sum())
        num_actual_expired = int(snap_df["actual_expired"].sum())
        num_model_expired = int(snap_df["predicted_expired"].sum())
        num_wrongly_expired = int(
            (snap_df["predicted_expired"] & snap_df["actual_ticketed"]).sum()
        )
        
        negative_precision = (
            1 - (num_wrongly_expired / num_model_expired)
            if num_model_expired > 0
            else 0.0
        )
        
        negative_recall = (
            int((snap_df["predicted_expired"] & snap_df["actual_expired"]).sum())
            / num_actual_expired
            if num_actual_expired > 0
            else 0.0
        )
        
        results[label] = {
            "offer_count": offer_count,
            "num_actual_ticketed": num_actual_ticketed,
            "num_actual_expired": num_actual_expired,
            "num_predicted_expired": num_model_expired,
            "num_wrongly_expired": num_wrongly_expired,
            "negative_precision": round(negative_precision, 3),
            "negative_recall": round(negative_recall, 3),
        }
    
    return results
