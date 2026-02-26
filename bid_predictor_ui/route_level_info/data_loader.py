from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, time as dt_time
from typing import List
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .cache_keys import audit_raw_cache_key


import pandas as pd
from pyarrow import fs as pyfs

from ..utils.redis_client import get_redis_client

CACHE_TTL_SECONDS = 24 * 3600

S3_DATASET_LISTING_URI = os.environ.get("S3_DATASET_LISTING_URI")
DAYS = int(os.environ.get("DAYS"))

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


def _compute_window(days: int = DAYS) -> tuple[datetime, datetime]:
    """
    Compute a time window for fetching S3 files.
    
    Default behavior (days=DAYS, which is 6):
    today - 5 days = anchor
    window = [anchor - DAYS, anchor] = [today - 11, today - 5] = 7 days total
    
    For custom periods:
    - 14 days: days=13 → window = [today - 18, today - 5] = 14 days
    - 21 days: days=20 → window = [today - 25, today - 5] = 21 days
    - 30 days: days=29 → window = [today - 34, today - 5] = 30 days
    
    This window is used for fetching S3 files based on their filenames
    """
    today = datetime.utcnow().date()
    anchor_day = today - timedelta(days=5)
    start_day = anchor_day - timedelta(days=days)
    start_ts = datetime.combine(start_day, dt_time.min)
    end_ts = datetime.combine(anchor_day, dt_time.max)
    print(f'[DATA LOADER S3 FETCH WINDOW] Start date: {start_ts}, End date: {end_ts}, Days: {days}')
    return start_ts, end_ts


# def _audit_combined_cache_key(start: datetime, end: datetime) -> str:
#     prefix = S3_DATASET_LISTING_URI or ""
#     return f"audit_dataset_combined:{prefix}:{start:%Y-%m-%d}:{end:%Y-%m-%d}"


def _load_single_file(filesystem, path, file_ts) -> pd.DataFrame:
    """Load a single CSV from S3 and add metadata."""
    try:
        with filesystem.open_input_file(path) as f:
            df = pd.read_csv(f)
            df["__source_file"] = path.split("/")[-1]
            df["__file_timestamp"] = file_ts
            return df
    except Exception as e:
        print(f"[Audit loader] Failed {path}: {e}")
        return pd.DataFrame()


def _load_audit_data_from_s3(start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    filesystem = pyfs.S3FileSystem()
    selector = pyfs.FileSelector(
        BASE_S3_URI.replace("s3://", ""),
        recursive=True,
    )

    files = filesystem.get_file_info(selector)
    valid_files = []

    for info in files:
        if info.type != pyfs.FileType.File:
            continue
        filename = info.path.split("/")[-1]
        file_ts = _parse_timestamp_from_name(filename)
        if file_ts and (start_ts <= file_ts <= end_ts):
            valid_files.append((info.path, file_ts))

    print(f"[Audit loader] {len(valid_files)} files to load from S3")

    frames: List[pd.DataFrame] = []

    # Load files in parallel using threads
    with ThreadPoolExecutor(max_workers=12) as executor:  # adjust max_workers as needed
        future_to_path = {
            executor.submit(_load_single_file, filesystem, path, ts): path
            for path, ts in valid_files
        }

        for future in as_completed(future_to_path):
            df = future.result()
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    start_concat = time.time()
    combined_df = pd.concat(frames, ignore_index=True)
    print(f"[Audit loader] Concatenated {len(frames)} frames into {len(combined_df)} rows in {time.time()-start_concat:.2f}s")
    return combined_df


def load_audit_data_cached(days: int = DAYS) -> pd.DataFrame:
    redis_client = get_redis_client()
    start_ts, end_ts = _compute_window(days=days)
    cache_key = audit_raw_cache_key(start_ts, end_ts)

    start = time.time()

    # 1. Check Redis cache first
    if redis_client is not None:
        cached = redis_client.get(cache_key)
        if cached:
            df = pickle.loads(cached)
            print(f"[RAW CACHE HIT] {len(df)} rows in {time.time()-start:.2f}s")
            return df

    print(f"[Audit cache] MISS {cache_key}, loading from S3")

    # 2. Load from S3
    start_time = time.time()
    df = _load_audit_data_from_s3(start_ts, end_ts)
    elapsed = time.time() - start_time
    print(f"[Audit loader] Loaded {len(df)} rows from S3 in {elapsed:.2f}s")

    if df.empty:
        print("[Audit loader] No audit data found")
        return df

    # --- Snapshot timestamp (notebook: current_timestamp)
    if "accept_prob_timestamp" not in df.columns:
        raise RuntimeError("accept_prob_timestamp missing from audit data")

    df["current_timestamp"] = pd.to_datetime(df["accept_prob_timestamp"])

    # --- Travel date (notebook: travel_date)
    if "departure_timestamp" not in df.columns:
        raise RuntimeError("departure_timestamp missing from audit data")

    df["travel_date"] = (
        pd.to_datetime(df["departure_timestamp"])
        .dt.date
    )

    # Optional safety checks (cheap, but very helpful)
    if df["current_timestamp"].isna().any():
        raise RuntimeError("Null current_timestamp after normalization")

    if df["travel_date"].isna().any():
        raise RuntimeError("Null travel_date after normalization")

    # 3. Cache normalized DF in Redis
    if redis_client is not None:
        start_cache = time.time()
        redis_client.setex(
            cache_key,
            CACHE_TTL_SECONDS,
            pickle.dumps(df),
        )
        cache_elapsed = time.time() - start_cache
        print(f"[Audit cache] Stored {len(df)} rows in Redis in {cache_elapsed:.2f}s")

    return df

