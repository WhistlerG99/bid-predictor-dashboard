import os
import pickle
from datetime import datetime

import psycopg2
import pandas as pd

from ..utils.redis_client import get_redis_client
from .data_loader import _compute_window

CACHE_TTL_SECONDS = 24 * 3600

REDSHIFT_CONN = {
    "host": os.environ["REDSHIFT_HOST"],
    "port": 5439,
    "dbname": os.environ["REDSHIFT_DATABASE"],
    "user": os.environ["REDSHIFT_USER"],
    "password": os.environ["REDSHIFT_PASSWORD"],
}

VALID_TICKETED_STATUSES = (
    "TICKETED",
    "CC_AUTH_DECLINED",
    "CC_AUTH_RETRY",
)


def _offer_status_cache_key(start: datetime, end: datetime) -> str:
    return f"audit_offer_status:{start:%Y-%m-%d}:{end:%Y-%m-%d}"


def _load_offer_statuses_from_redshift(
    offer_ids: list[int],
) -> pd.DataFrame:
    if not offer_ids:
        return pd.DataFrame(columns=["offer_id", "offer_status"])

    placeholders = ",".join(["%s"] * len(offer_ids))

    query = f"""
        SELECT
            id AS offer_id,
            offer_status
        FROM prd_offers_rds.offers
        WHERE id IN ({placeholders})
    """

    with psycopg2.connect(**REDSHIFT_CONN) as conn:
        df = pd.read_sql(query, conn, params=offer_ids)

    return df


def load_offer_statuses_cached(offer_ids: list[int]) -> pd.DataFrame:
    if not offer_ids:
        return pd.DataFrame({"offer_id": [], "offer_status": []})

    redis_client = get_redis_client()
    start_ts, end_ts = _compute_window()
    cache_key = _offer_status_cache_key(start_ts, end_ts)

    cached_df = pd.DataFrame(columns=["offer_id", "offer_status"])
    if redis_client is not None:
        cached = redis_client.get(cache_key)
        if cached:
            cached_df = pickle.loads(cached)

    cached_ids = set(cached_df["offer_id"].tolist())
    missing_ids = [oid for oid in offer_ids if oid not in cached_ids]

    new_df = pd.DataFrame(columns=["offer_id", "offer_status"])
    if missing_ids:
        new_df = _load_offer_statuses_from_redshift(missing_ids)

    df = pd.concat([cached_df, new_df], ignore_index=True)
    print("Columnssssssssssssss----------------------------------------")
    print(df.columns)

    # Ensure the column always exists for the app
    # if "offer_status" not in df.columns:
    #     df["offer_status"] = pd.NA

    # Update cache
    if redis_client is not None and not df.empty:
        redis_client.setex(
            cache_key,
            CACHE_TTL_SECONDS,
            pickle.dumps(df),
        )

    # Debug log for missing offers
    missing_offers = set(offer_ids) - set(df["offer_id"].tolist())
    if missing_offers:
        print(f"[DEBUG] offer_status missing for offer_ids: {missing_offers}")

    return df


