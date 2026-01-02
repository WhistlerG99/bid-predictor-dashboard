import pandas as pd

HORIZONS = {
    "72h": 72,
    "48h": 48,
    "24h": 24,
}


def _select_prediction_for_horizon(df: pd.DataFrame, hours_before: int) -> pd.DataFrame:
    """
    Select predictions around departure time minus hours_before ±1h
    """
    lower = df["departure_timestamp"] - pd.Timedelta(hours=hours_before + 1)
    upper = df["departure_timestamp"] - pd.Timedelta(hours=hours_before - 1)
    mask = (df["accept_prob_timestamp"] >= lower) & (df["accept_prob_timestamp"] <= upper)
    return df.loc[mask].copy()


def compute_horizon_metrics(df: pd.DataFrame, threshold: float) -> dict:
    if "offer_status" not in df.columns:
        raise RuntimeError("offer_status missing in compute_horizon_metrics")

    results = {}

    for label, hours in HORIZONS.items():
        hdf = _select_prediction_for_horizon(df, hours)
        if hdf.empty:
            results[label] = {
                "false_negatives": 0,
                "accuracy": 0.0,
                "expiry_horizon": 0,
            }
            continue

        hdf["predicted_expired"] = hdf["accept_prob"] < threshold
        hdf["actual_ticketed"] = hdf["offer_status"].isin(
            ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
        )
        hdf["actual_expired"] = hdf["offer_status"] == "EXPIRED"

        hdf = (
            hdf.sort_values("accept_prob_timestamp")
               .groupby("offer_id", as_index=False)
               .last()
        )

        false_negatives = (hdf["predicted_expired"] & hdf["actual_ticketed"]).sum()

        TP = (~hdf["predicted_expired"] & hdf["actual_ticketed"]).sum()
        TN = (hdf["predicted_expired"] & hdf["actual_expired"]).sum()
        total = len(hdf)
        accuracy = ((TP + TN) / total * 100) if total else 0.0

        results[label] = {
            "false_negatives": int(false_negatives),
            "accuracy": round(accuracy, 2),
            "expiry_horizon": int(hdf["predicted_expired"].sum()),
        }

    return results



