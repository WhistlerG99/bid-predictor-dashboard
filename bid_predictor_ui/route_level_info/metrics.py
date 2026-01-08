# import pandas as pd

# HORIZONS = {
#     "72h": 72,
#     "48h": 48,
#     "24h": 24,
# }

# BUCKETS = {
#     "72h": (48, 72),   # 72 <= hours_before < 48
#     "48h": (24, 48),   # 48 <= hours_before < 24
#     "24h": (0, 24),    # 24 <= hours_before < 0
# }

# def _select_predictions_for_bucket(df: pd.DataFrame, lower_hours: int, upper_hours: int) -> pd.DataFrame:
#     """
#     Select predictions in the bucket: lower_hours <= hours_before < upper_hours
#     """
#     lower = df["departure_timestamp"] - pd.Timedelta(hours=upper_hours)
#     upper = df["departure_timestamp"] - pd.Timedelta(hours=lower_hours)
#     mask = (df["accept_prob_timestamp"] >= lower) & (df["accept_prob_timestamp"] <= upper)
#     return df.loc[mask].copy()



# def compute_bucket_metrics(df: pd.DataFrame, threshold: float) -> dict:
#     if "offer_status" not in df.columns:
#         raise RuntimeError("offer_status missing in compute_bucket_metrics")

#     results = {}

#     for label, (lower, upper) in BUCKETS.items():
#         bucket_df = _select_predictions_for_bucket(df, lower, upper)
#         if bucket_df.empty:
#             results[label] = {
#                 "num_wrongly_expired": 0,
#                 "expiry_horizon": 0,
#                 "percent_wrongly_expired": 0.0,
#                 "negative_precision": 0.0,
#                 "negative_recall": 0.0,
#                 "score": 0,
#             }
#             continue

#         # Deduplicate per bucket (latest accept_prob_timestamp per offer)
#         bucket_df = (
#             bucket_df.sort_values("accept_prob_timestamp")
#                      .groupby("offer_id", as_index=False)
#                      .last()
#         )

#         bucket_df["predicted_expired"] = bucket_df["accept_prob"] < threshold
#         bucket_df["actual_ticketed"] = bucket_df["offer_status"].isin(
#             ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
#         )
#         bucket_df["actual_expired"] = bucket_df["offer_status"] == "EXPIRED"

#         num_model_expired = int(bucket_df["predicted_expired"].sum())
#         num_wrongly_expired = (bucket_df["predicted_expired"] & bucket_df["actual_ticketed"]).sum()

#         percent_wrongly_expired = (
#             round(num_wrongly_expired / num_model_expired * 100, 2)
#             if num_model_expired > 0 else 0.0
#         )

#         negative_precision = 1 - (num_wrongly_expired / num_model_expired) if num_model_expired > 0 else 0.0

#         model_and_actual_expired = bucket_df["predicted_expired"] & bucket_df["actual_expired"]
#         negative_recall = (
#             model_and_actual_expired.sum() / bucket_df["actual_expired"].sum()
#             if bucket_df["actual_expired"].sum() > 0 else 0.0
#         )

#         score = num_model_expired - num_wrongly_expired - (num_wrongly_expired*10)

#         results[label] = {
#             "expiry_horizon": int(num_model_expired),
#             "num_wrongly_expired": int(num_wrongly_expired),
#             "percent_wrongly_expired": percent_wrongly_expired,
#             "negative_precision": round(negative_precision, 3),
#             "negative_recall": round(negative_recall, 3),
#             "score": int(score),
#         }

#     return results

import pandas as pd

# Exact horizons (same concept as notebook)
HORIZONS = {
    "72h": 72,
    "48h": 48,
    "24h": 24,
}

# Tolerance around the exact horizon (±1 hour, notebook equivalent)
TOLERANCE_HOURS = 1


def _select_nearest_snapshot(
    df: pd.DataFrame,
    target_hours: int,
) -> pd.DataFrame:
    """
    For each offer_id, select the prediction snapshot whose timestamp
    is closest to `target_hours` before departure, within ±TOLERANCE_HOURS.

    This matches the notebook's merge_asof + abs(hour_error) <= 1 logic.
    """

    if df.empty:
        return df

    required_cols = {
        "offer_id",
        "departure_timestamp",
        "accept_prob_timestamp",
        "accept_prob",
        "offer_status",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in metrics: {missing}")

    df = df.copy()

    # Hours before departure for each prediction
    df["hrs_before_dep"] = (
        (df["departure_timestamp"] - df["accept_prob_timestamp"])
        .dt.total_seconds() / 3600
    )

    # Absolute error from target horizon
    df["abs_err"] = (df["hrs_before_dep"] - target_hours).abs()

    # Keep only snapshots within tolerance
    df = df[df["abs_err"] <= TOLERANCE_HOURS]
    if df.empty:
        return df

    # Nearest snapshot per offer_id
    # (same offer can appear once per horizon, but can appear again for other horizons)
    nearest = (
        df.sort_values("abs_err")
          .groupby("offer_id", as_index=False)
          .first()
    )

    return nearest


def compute_bucket_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """
    Snapshot-based horizon metrics (not bucket-based).

    For each horizon (72h / 48h / 24h):
    - Select nearest snapshot per offer within ±1h
    - Count snapshots (num_of_bids)
    - Compute wrongly expired, negative precision, negative recall

    Offer status semantics are intentionally kept EXACTLY
    as they are in the current app.
    """

    results = {}

    if df.empty:
        for label in HORIZONS:
            results[label] = {
                "num_of_bids": 0,
                "num_wrongly_expired": 0,
                "negative_precision": 0.0,
                "negative_recall": 0.0,
                "num_predicted_expired": 0,
            }
        return results

    # Defensive: ensure timestamps are datetime
    df = df.copy()
    df["departure_timestamp"] = pd.to_datetime(df["departure_timestamp"])
    df["accept_prob_timestamp"] = pd.to_datetime(df["accept_prob_timestamp"])

    for label, target_hours in HORIZONS.items():
        snap_df = _select_nearest_snapshot(df, target_hours)

        if snap_df.empty:
            results[label] = {
                "num_of_bids": 0,
                "num_wrongly_expired": 0,
                "negative_precision": 0.0,
                "negative_recall": 0.0,
                "num_predicted_expired": 0,
            }
            continue

        # Model decision
        snap_df["predicted_expired"] = snap_df["accept_prob"] < threshold

        # Ground truth (KEEPING APP SEMANTICS)
        snap_df["actual_ticketed"] = snap_df["offer_status"].isin(
            ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
        )
        snap_df["actual_expired"] = snap_df["offer_status"] == "EXPIRED"

        num_of_bids = len(snap_df)
        num_model_expired = int(snap_df["predicted_expired"].sum())

        num_wrongly_expired = int(
            (snap_df["predicted_expired"] & snap_df["actual_ticketed"]).sum()
        )

        # Precision on negative class (same as notebook)
        negative_precision = (
            1 - (num_wrongly_expired / num_model_expired)
            if num_model_expired > 0 else 0.0
        )

        # Recall on negative class
        actual_expired_count = int(snap_df["actual_expired"].sum())
        negative_recall = (
            int((snap_df["predicted_expired"] & snap_df["actual_expired"]).sum())
            / actual_expired_count
            if actual_expired_count > 0 else 0.0
        )

        results[label] = {
            "num_of_bids": int(num_of_bids),
            "num_wrongly_expired": int(num_wrongly_expired),
            "negative_precision": round(negative_precision, 3),
            "negative_recall": round(negative_recall, 3),
            "num_predicted_expired": int(num_model_expired),
        }

    return results
