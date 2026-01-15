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
        print(f"[REDSHIFT] No offer IDs to fetch")
        return pd.DataFrame(columns=["offer_id", "offer_status"])

    print(f"[REDSHIFT START] Fetching status for {len(offer_ids)} offer IDs from Redshift")
    placeholders = ",".join(["%s"] * len(offer_ids))

    query = f"""
        SELECT
            id AS offer_id,
            offer_status
        FROM prd_offers_rds.offers
        WHERE id IN ({placeholders})
    """

    try:
        print(f"[REDSHIFT CONNECT] Connecting to Redshift...")
        with psycopg2.connect(**REDSHIFT_CONN) as conn:
            print(f"[REDSHIFT QUERY] Executing query for {len(offer_ids)} offers...")
            df = pd.read_sql(query, conn, params=offer_ids)
            print(f"[REDSHIFT SUCCESS] Got {len(df)} rows from Redshift")
            return df
    except Exception as e:
        print(f"[REDSHIFT ERROR] Failed to fetch from Redshift: {str(e)}")
        return pd.DataFrame(columns=["offer_id", "offer_status"])


def load_offer_statuses_cached(offer_ids: list[int]) -> pd.DataFrame:
    print(f"[OFFER STATUS CACHE START] Total offer_ids to load: {len(offer_ids)}")
    
    if not offer_ids:
        print(f"[OFFER STATUS CACHE] No offer IDs provided")
        return pd.DataFrame({"offer_id": [], "offer_status": []})

    redis_client = get_redis_client()
    print(f"[OFFER STATUS CACHE] Redis client: {'Connected' if redis_client is not None else 'None'}")
    
    start_ts, end_ts = _compute_window()
    cache_key = _offer_status_cache_key(start_ts, end_ts)
    print(f"[OFFER STATUS CACHE] Cache key: {cache_key}")

    cached_df = pd.DataFrame(columns=["offer_id", "offer_status"])
    if redis_client is not None:
        print(f"[OFFER STATUS CACHE] Checking Redis for cached data...")
        cached = redis_client.get(cache_key)
        if cached:
            cached_df = pickle.loads(cached)
            print(f"[OFFER STATUS CACHE] Redis HIT: Got {len(cached_df)} cached offers")
        else:
            print(f"[OFFER STATUS CACHE] Redis MISS: No cached data found")

    cached_ids = set(cached_df["offer_id"].tolist())
    missing_ids = [oid for oid in offer_ids if oid not in cached_ids]
    print(f"[OFFER STATUS CACHE] Cached IDs: {len(cached_ids)}, Missing IDs: {len(missing_ids)}")

    new_df = pd.DataFrame(columns=["offer_id", "offer_status"])
    if missing_ids:
        print(f"[OFFER STATUS CACHE] Fetching {len(missing_ids)} missing offers from Redshift...")
        new_df = _load_offer_statuses_from_redshift(missing_ids)
        print(f"[OFFER STATUS CACHE] Got {len(new_df)} new offers from Redshift")

    df = pd.concat([cached_df, new_df], ignore_index=True)
    print(f"[OFFER STATUS CACHE] Combined dataframe: {len(df)} rows")
    print("Columnssssssssssssss----------------------------------------")
    print(df.columns)

    # Ensure the column always exists for the app
    # if "offer_status" not in df.columns:
    #     df["offer_status"] = pd.NA

    # Update cache
    if redis_client is not None and not df.empty:
        print(f"[OFFER STATUS CACHE] Updating Redis cache...")
        redis_client.setex(
            cache_key,
            CACHE_TTL_SECONDS,
            pickle.dumps(df),
        )
        print(f"[OFFER STATUS CACHE] Redis cache updated")

    # Debug log for missing offers
    missing_offers = set(offer_ids) - set(df["offer_id"].tolist())
    if missing_offers:
        print(f"[OFFER STATUS CACHE WARNING] Missing statuses for {len(missing_offers)} offers: {missing_offers}")
    
    print(f"[OFFER STATUS CACHE END] Returning {len(df)} rows with offer statuses")
    return df


