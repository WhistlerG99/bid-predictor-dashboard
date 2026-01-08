import pickle
import time
import pandas as pd

from .cache_keys import route_metrics_cache_key
from .data_loader import _compute_window
from ..utils.redis_client import get_redis_client

CACHE_TTL_SECONDS = 24 * 3600


def get_cached_route_metrics(carrier: str, threshold: float):
    redis = get_redis_client()
    if not redis:
        return None

    start_ts, end_ts = _compute_window()
    key = route_metrics_cache_key(start_ts, end_ts, carrier, threshold)

    cached = redis.get(key)
    if cached:
        data = pickle.loads(cached)
        print(f"[ROUTE CACHE HIT] {carrier}")
        return data

    print(f"[ROUTE CACHE MISS] {carrier}")
    return None


def set_cached_route_metrics(carrier: str, threshold: float, rows: list[dict]):
    redis = get_redis_client()
    if not redis:
        return

    start_ts, end_ts = _compute_window()
    key = route_metrics_cache_key(start_ts, end_ts, carrier, threshold)

    redis.setex(
        key,
        CACHE_TTL_SECONDS,
        pickle.dumps(rows),
    )

    print(f"[ROUTE CACHE SET] {carrier} rows={len(rows)}")
