from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from pyarrow import fs as pyfs

import pickle
from ..utils.redis_client import get_redis_client

CACHE_TTL_SECONDS = 24 * 3600

S3_DATASET_LISTING_URI = os.environ.get("S3_DATASET_LISTING_URI")

BASE_S3_URI = (
    "s3://amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8/"
    "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/audit_bid_predictor_csv/"
)

FILENAME_PATTERN = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-audit_bid_predictor\.csv"
)
def _parse_timestamp_from_name(name: str) -> datetime | None:
    match = FILENAME_PATTERN.search(name)
    if not match:
        return None
    return datetime.strptime(match.group("ts"), "%Y-%m-%dT%H-%M-%S")


def _compute_window() -> tuple[datetime, datetime]:
    """
    today - 5 days = anchor
    window = [anchor - 7 days, anchor]
    """
    now = datetime.utcnow()
    anchor = now - timedelta(days=5)
    start = anchor - timedelta(days=7)
    return start, anchor


def _audit_combined_cache_key(start: datetime, end: datetime) -> str:
    prefix = S3_DATASET_LISTING_URI or ""
    return (
        f"audit_dataset_combined:{prefix}:"
        f"{start:%Y-%m-%d}:{end:%Y-%m-%d}"
    )

def _load_audit_data_from_s3(start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    filesystem = pyfs.S3FileSystem()

    selector = pyfs.FileSelector(
        BASE_S3_URI.replace("s3://", ""),
        recursive=True,
    )

    files = filesystem.get_file_info(selector)
    frames: List[pd.DataFrame] = []

    for info in files:
        if info.type != pyfs.FileType.File:
            continue

        filename = info.path.split("/")[-1]
        file_ts = _parse_timestamp_from_name(filename)
        if not file_ts:
            continue

        if not (start_ts <= file_ts <= end_ts):
            continue

        try:
            with filesystem.open_input_file(info.path) as f:
                df = pd.read_csv(f)
                df["__source_file"] = filename
                df["__file_timestamp"] = file_ts
                frames.append(df)
        except Exception as exc:
            print(f"[Audit loader] Failed {info.path}: {exc}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def load_audit_data_cached() -> pd.DataFrame:
    redis_client = get_redis_client()
    start_ts, end_ts = _compute_window()

    cache_key = _audit_combined_cache_key(start_ts, end_ts)

    if redis_client is not None:
        cached = redis_client.get(cache_key)
        if cached:
            print(f"[Audit cache] Hit {cache_key}")
            return pickle.loads(cached)

    print(
        f"[Audit cache] Miss {cache_key}, loading from S3 "
        f"({start_ts:%Y-%m-%d} → {end_ts:%Y-%m-%d})"
    )

    df = _load_audit_data_from_s3(start_ts, end_ts)

    if df.empty:
        print("[Audit loader] No audit data found")
        return df

    print(f"[Audit loader] Loaded {len(df):,} rows")

    if redis_client is not None:
        redis_client.setex(
            cache_key,
            CACHE_TTL_SECONDS,
            pickle.dumps(df),
        )
        print(
            f"[Audit cache] Stored combined dataset "
            f"{cache_key} ({len(df):,} rows)"
        )

    return df
