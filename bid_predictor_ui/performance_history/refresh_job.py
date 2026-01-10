"""Background refresh job for performance history snapshots."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

from bid_predictor_ui.performance_history.data import (
    update_performance_history_from_source,
)

logger = logging.getLogger(__name__)


def _get_redis_client(redis_url: Optional[str]) -> Optional[object]:
    if not redis_url:
        return None
    try:
        import redis  # type: ignore
    except ImportError:
        logger.warning("Redis client unavailable; skipping cache integration.")
        return None

    try:
        return redis.Redis.from_url(redis_url)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to connect to Redis.", extra={"error": str(exc)})
        return None


def run_performance_history_refresh(
    history_uri: str,
    source_uri: str,
    refresh_days: int,
    redis_url: Optional[str] = None,
) -> bool:
    if not history_uri or not source_uri:
        logger.info("Performance history refresh skipped: missing configuration.")
        return False

    cache_client = _get_redis_client(redis_url)
    update_performance_history_from_source(
        history_uri,
        source_uri,
        refresh_days=refresh_days,
        cache_client=cache_client,
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh performance history parquet data from source datasets."
    )
    parser.add_argument(
        "--history-uri",
        default=os.getenv("PERFORMANCE_HISTORY_S3_URI", ""),
        help="Destination URI for performance history parquet data.",
    )
    parser.add_argument(
        "--source-uri",
        default=os.getenv("S3_DATASET_LISTING_URI", ""),
        help="Source dataset listing URI (local path or S3).",
    )
    parser.add_argument(
        "--refresh-days",
        default=int(os.getenv("PERFORMANCE_HISTORY_REFRESH_DAYS", "5")),
        type=int,
        help="Number of trailing days to refresh.",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", ""),
        help="Redis URL for offer-status caching (optional).",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    parser = build_parser()
    args = parser.parse_args()
    success = run_performance_history_refresh(
        args.history_uri,
        args.source_uri,
        args.refresh_days,
        redis_url=args.redis_url or None,
    )
    if not success:
        logger.error("Performance history refresh failed: missing required settings.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
