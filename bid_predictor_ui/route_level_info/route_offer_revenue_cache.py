from __future__ import annotations

import json
from datetime import date
from typing import Dict, Tuple

from .redshift_offers_usd_loader import load_offers_with_usd_amount
from ..utils.redis_client import get_redis_client

CACHE_TTL_SECONDS = 24 * 3600


def _cache_key(carrier: str, start_date: date, end_date: date, period_days: int = 7) -> str:
    return f"route_offer_revenue:{carrier}:{start_date}:{end_date}:period={period_days}d"


def get_cached_route_offer_revenue(
    *,
    carrier: str,
    start_date: date,
    end_date: date,
    period_days: int = 7,
) -> Dict[Tuple[str, str], dict]:
    """
    {
      (origination, destination): {
        offers_usd,
        upgrades_usd,
        offer_count,
        num_actual_ticketed
      }
    }
    """

    redis = get_redis_client()
    key = _cache_key(carrier, start_date, end_date, period_days)

    cached = redis.get(key)
    if cached:
        raw = json.loads(cached)
        print(f"[ROUTE REVENUE CACHE HIT] {carrier} period={period_days}d")
        return {(k.split("|")[0], k.split("|")[1]): v for k, v in raw.items()}

    print(f"[ROUTE REVENUE CACHE MISS] {carrier} period={period_days}d, fetching from Redshift")
    df = load_offers_with_usd_amount(
        start_date=start_date,
        end_date=end_date,
        carriers=[carrier],
    )

    if df.empty:
        print(f"[ROUTE REVENUE CACHE] No data from Redshift for {carrier} period={period_days}d")
        redis.setex(key, CACHE_TTL_SECONDS, json.dumps({}))
        return {}

    result = {}

    for (orig, dest), g in df.groupby(["origination", "destination"]):
        offers_usd = g["offer_value_usd"].sum()

        upgrades_mask = g["offer_status"].isin(
            ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
        )

        result[f"{orig}|{dest}"] = {
            "offers_usd": float(offers_usd),
            "upgrades_usd": float(g.loc[upgrades_mask, "offer_value_usd"].sum()),
            "offer_count": int(len(g)),
            "num_actual_ticketed": int(
                g["offer_status"].isin(
                    ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
                ).sum()
            ),
        }

    print(f"[ROUTE REVENUE CACHE SET] {carrier} period={period_days}d routes={len(result)}")
    redis.setex(key, CACHE_TTL_SECONDS, json.dumps(result))
    return {(k.split("|")[0], k.split("|")[1]): v for k, v in result.items()}
