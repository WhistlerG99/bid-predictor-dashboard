import os
import redis
from typing import Optional


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_redis_client() -> Optional[redis.Redis]:
    try:
        return redis.Redis.from_url(REDIS_URL)
    except Exception:
        return None
