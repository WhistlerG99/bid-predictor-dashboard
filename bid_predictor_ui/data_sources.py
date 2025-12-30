"""Shared loaders for path and Redshift-backed datasets.

The helpers here are intentionally UI-focused: they support the flexible file
and Redshift sources that individual tabs previously reimplemented on their
own.  Caching is opt-in with a reload flag so dashboards can refresh data when
files change without restarting the app.
"""
from __future__ import annotations

import json
import os
import re
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import pandas as pd
import psycopg2
from pyarrow import fs as pyfs

DEFAULT_ACCEPTANCE_TABLE = "model_prediction_testing.audit_bid_predictor"
DEFAULT_OFFERS_TABLE = os.getenv("REDSHIFT_OFFERS_TABLE", "prd_offers_rds.offers")
DEFAULT_OFFERS_TABLE = os.getenv("REDSHIFT_OFFERS_TABLE", "prd_offers_rds.offers")

_TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")
_DATA_CACHE: MutableMapping[Tuple[str, Optional[str], Optional[int], Optional[str]], pd.DataFrame] = {}


def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def _list_remote_files(filesystem: pyfs.FileSystem, uri: str) -> List[str]:
    """List parquet/CSV files under an S3 URI, returning bucket/key paths."""

    relative_path = uri.replace("s3://", "", 1)
    info = filesystem.get_file_info([relative_path])[0]

    if info.type == pyfs.FileType.NotFound:
        raise FileNotFoundError(f"Dataset path does not exist: {uri}")

    exts = {".parquet", ".pq", ".csv"}

    if info.type == pyfs.FileType.File:
        if PurePosixPath(relative_path).suffix.lower() in exts:
            return [relative_path]
        raise ValueError("Provided file is not a parquet or CSV file.")

    if info.type == pyfs.FileType.Directory:
        selector = pyfs.FileSelector(relative_path, recursive=True)
        entries = filesystem.get_file_info(selector)
        files = [
            entry.path
            for entry in entries
            if entry.type == pyfs.FileType.File
            and PurePosixPath(entry.path).suffix.lower() in exts
        ]
        if not files:
            raise ValueError(
                "No parquet or CSV files found in the provided directory or its subdirectories."
            )
        return sorted(files)

    raise ValueError(f"Unsupported S3 path type for {uri}")


def _extract_timestamp_from_name(name: str) -> Optional[pd.Timestamp]:
    match = _TIMESTAMP_PATTERN.search(name)
    if not match:
        return None
    try:
        return pd.to_datetime(match.group(1), format="%Y-%m-%dT%H-%M-%S")
    except (TypeError, ValueError):
        return None


def _filter_files_by_recent_hours(files: Sequence[str], hours: Optional[int]) -> List[str]:
    if hours is None:
        return sorted(files)

    dated_files: List[Tuple[str, Optional[pd.Timestamp]]] = []
    for file_path in files:
        timestamp = _extract_timestamp_from_name(PurePosixPath(file_path).name)
        dated_files.append((file_path, timestamp))

    timestamps = [ts for _, ts in dated_files if ts is not None and not pd.isna(ts)]
    if not timestamps:
        return sorted(files)

    latest_timestamp = max(timestamps)
    threshold = latest_timestamp - pd.Timedelta(hours=hours)
    filtered = [path for path, ts in dated_files if ts is not None and ts >= threshold]
    return sorted(filtered or [path for path, _ in dated_files])


def parse_recent_hours(hours_value: object) -> Optional[int]:
    if hours_value in (None, ""):
        return None
    try:
        hours = int(hours_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Recent hours must be an integer.") from exc
    if hours <= 0:
        raise ValueError("Recent hours must be positive.")
    return hours


def _validate_table_name(table_name: str) -> str:
    if not table_name or not re.match(r"^[A-Za-z0-9_.]+$", table_name):
        raise ValueError(
            "Table name must include only letters, numbers, underscores, or periods."
        )
    return table_name


def _redshift_credentials() -> Dict[str, str]:
    required = [
        "REDSHIFT_HOST",
        "REDSHIFT_DATABASE",
        "REDSHIFT_USER",
        "REDSHIFT_PASSWORD",
        "REDSHIFT_PORT",
    ]
    values = {name: os.getenv(name) for name in required}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise EnvironmentError(
            "Missing environment variables for Redshift connection: "
            + ", ".join(missing)
        )
    return {name: str(values[name]) for name in required}


def _has_redshift_credentials() -> bool:
    return all(
        os.getenv(key)
        for key in (
            "REDSHIFT_HOST",
            "REDSHIFT_DATABASE",
            "REDSHIFT_USER",
            "REDSHIFT_PASSWORD",
            "REDSHIFT_PORT",
        )
    )


def get_redshift_connection() -> Optional["psycopg2.extensions.connection"]:
    """Get Redshift connection if configured."""
    if not _has_redshift_credentials():
        return None
    try:
        credentials = _redshift_credentials()
        return psycopg2.connect(
            host=credentials["REDSHIFT_HOST"],
            dbname=credentials["REDSHIFT_DATABASE"],
            user=credentials["REDSHIFT_USER"],
            password=credentials["REDSHIFT_PASSWORD"],
            port=int(credentials["REDSHIFT_PORT"]),
        )
    except Exception:
        return None


def _has_redshift_credentials() -> bool:
    return all(
        os.getenv(key)
        for key in (
            "REDSHIFT_HOST",
            "REDSHIFT_DATABASE",
            "REDSHIFT_USER",
            "REDSHIFT_PASSWORD",
            "REDSHIFT_PORT",
        )
    )


def get_redshift_connection() -> Optional["psycopg2.extensions.connection"]:
    """Get Redshift connection if configured."""
    if not _has_redshift_credentials():
        return None
    try:
        credentials = _redshift_credentials()
        return psycopg2.connect(
            host=credentials["REDSHIFT_HOST"],
            dbname=credentials["REDSHIFT_DATABASE"],
            user=credentials["REDSHIFT_USER"],
            password=credentials["REDSHIFT_PASSWORD"],
            port=int(credentials["REDSHIFT_PORT"]),
        )
    except Exception:
        return None


def _load_from_path(path: str, hours: Optional[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if _is_s3_path(path):
        filesystem = pyfs.S3FileSystem()
        files = _list_remote_files(filesystem, path)
        filtered_files = _filter_files_by_recent_hours(files, hours)
        for remote_path in filtered_files:
            suffix = PurePosixPath(remote_path).suffix.lower()
            with filesystem.open_input_file(remote_path) as handle:
                if suffix in {".parquet", ".pq"}:
                    frames.append(pd.read_parquet(handle))
                elif suffix == ".csv":
                    frames.append(pd.read_csv(handle))
                else:  # pragma: no cover - guarded by suffix checks
                    raise ValueError(
                        f"Unsupported file extension for s3://{remote_path}"
                    )
    else:
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {resolved}")

        if resolved.is_dir():
            parquet_files = sorted(resolved.glob("*.parquet")) + sorted(
                resolved.glob("*.pq")
            )
            csv_files = sorted(resolved.glob("*.csv"))
            files = parquet_files + csv_files
            if not files:
                raise ValueError(
                    "No parquet or CSV files found in the provided directory."
                )
            filtered_files = _filter_files_by_recent_hours(
                [str(path) for path in files], hours
            )
            files = [Path(file_path) for file_path in filtered_files]
        else:
            files = [resolved]

        for file_path in files:
            suffix = file_path.suffix.lower()
            if suffix in {".parquet", ".pq"}:
                frames.append(pd.read_parquet(file_path))
            elif suffix == ".csv":
                frames.append(pd.read_csv(file_path))
            else:
                raise ValueError(f"Unsupported file extension for {file_path}")

    return pd.concat(frames, ignore_index=True, sort=False)


def _load_from_redshift(table_name: str, hours: Optional[int]) -> pd.DataFrame:
    validated_table = _validate_table_name(table_name)
    credentials = _redshift_credentials()
    query = f"SELECT * FROM {validated_table}"
    params: Tuple[object, ...] = ()
    if hours is not None:
        query += " WHERE accept_prob_timestamp >= DATEADD(hour, -%s, GETDATE())"
        params = (hours,)

    connection = psycopg2.connect(
        host=credentials["REDSHIFT_HOST"],
        dbname=credentials["REDSHIFT_DATABASE"],
        user=credentials["REDSHIFT_USER"],
        password=credentials["REDSHIFT_PASSWORD"],
        port=int(credentials["REDSHIFT_PORT"]),
    )
    with closing(connection) as conn:
        return pd.read_sql_query(query, conn, params=params or None)


def _normalize_config(config: str | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(config, str):
        return {"source": "path", "path": config, "hours": None}
    source = str(config.get("source") or "path").lower()
    normalized: Dict[str, object] = {"source": source}
    if source == "redshift":
        normalized["table"] = str(config.get("table") or DEFAULT_ACCEPTANCE_TABLE)
    else:
        normalized["path"] = str(config.get("path") or "")

    normalized["hours"] = parse_recent_hours(config.get("hours"))
    return normalized


def _cache_key(config: Mapping[str, object], normalizer: Optional[Callable[[pd.DataFrame], pd.DataFrame]]) -> Tuple[str, Optional[str], Optional[int], Optional[str]]:
    source = str(config.get("source") or "path")
    identifier = (
        str(config.get("table"))
        if source == "redshift"
        else str(config.get("path"))
    )
    hours = config.get("hours") if isinstance(config.get("hours"), int) else None
    normalizer_key = getattr(normalizer, "__name__", None)
    return (source, identifier, hours, normalizer_key)


def load_dataset_from_source(
    config: str | Mapping[str, object],
    *,
    normalizer: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    reload: bool = False,
) -> pd.DataFrame:
    """Load tabular data from either a filesystem path or Redshift table.

    The function caches results keyed by source, identifier, and recent-hour
    window.  Pass ``reload=True`` to bypass and refresh the cache for the given
    configuration.
    """

    normalized_config = _normalize_config(config)
    key = _cache_key(normalized_config, normalizer)
    if not reload and key in _DATA_CACHE:
        return _DATA_CACHE[key]

    source = normalized_config["source"]
    hours = normalized_config.get("hours")
    if source == "redshift":
        table = str(normalized_config.get("table") or DEFAULT_ACCEPTANCE_TABLE)
        data = _load_from_redshift(table, hours)  # type: ignore[arg-type]
    else:
        path = str(normalized_config.get("path") or "")
        if not path:
            raise ValueError("Please provide a dataset path.")
        data = _load_from_path(path, hours)  # type: ignore[arg-type]

    if normalizer:
        data = normalizer(data)

    _DATA_CACHE[key] = data
    return data


def offer_status_cache_key(prefix: str, hour_timestamp: pd.Timestamp) -> str:
    """Cache key for offer_status mapping for a specific hour."""
    hour_str = hour_timestamp.strftime("%Y-%m-%dT%H")
    return f"offer_status_hour:{prefix}:{hour_str}"


def get_offer_statuses_from_cache(
    cache_client: Optional[object],
    cache_prefix: str,
    hour_timestamps: list[pd.Timestamp],
) -> dict[str, str]:
    """Get offer_status mappings from cache for given hours."""
    if cache_client is None:
        return {}

    combined_statuses: dict[str, str] = {}
    for hour_ts in hour_timestamps:
        cache_key = offer_status_cache_key(cache_prefix, hour_ts)
        cached = cache_client.get(cache_key)
        if cached:
            try:
                statuses = json.loads(cached.decode("utf-8"))
                combined_statuses.update(statuses)
            except Exception:
                continue

    return combined_statuses


def cache_offer_statuses(
    cache_client: Optional[object],
    cache_prefix: str,
    hour_timestamp: pd.Timestamp,
    statuses: Mapping[str, str],
) -> None:
    """Cache offer_status mappings for a specific hour."""
    if cache_client is None:
        return

    try:
        cache_key = offer_status_cache_key(cache_prefix, hour_timestamp)
        cache_client.setex(cache_key, 7200, json.dumps(dict(statuses)))
    except Exception:
        return


def fetch_offer_statuses(
    offer_ids: Iterable[str],
    *,
    offers_table: str = DEFAULT_OFFERS_TABLE,
    connection: Optional["psycopg2.extensions.connection"] = None,
) -> dict[str, str]:
    """Fetch offer_status for given offer_ids from Redshift."""
    offer_ids = [str(offer_id) for offer_id in offer_ids if offer_id]
    if not offer_ids:
        return {}

    conn = connection or get_redshift_connection()
    if conn is None:
        return {}

    try:
        placeholders = ",".join(["%s"] * len(offer_ids))
        query = f"""
            SELECT id, offer_status
            FROM {offers_table}
            WHERE id IN ({placeholders})
        """
        with closing(conn) as active:
            with active.cursor() as cursor:
                cursor.execute(query, offer_ids)
                results = cursor.fetchall()
                return {
                    str(row[0]): str(row[1]) for row in results if row[0] and row[1]
                }
    except Exception:
        return {}


def enrich_with_offer_status(
    dataset: pd.DataFrame,
    *,
    cache_client: Optional[object] = None,
    cache_prefix: str = "",
    hour_timestamps: Optional[list[pd.Timestamp]] = None,
    offers_table: str = DEFAULT_OFFERS_TABLE,
) -> pd.DataFrame:
    """Enrich dataset with offer_status from cache or Redshift."""
    if dataset.empty or "offer_id" not in dataset.columns:
        return dataset

    working = dataset.copy()
    if hour_timestamps is None and "accept_prob_timestamp" in working.columns:
        timestamps = pd.to_datetime(working["accept_prob_timestamp"], errors="coerce")
        hour_timestamps = [
            ts.replace(minute=0, second=0, microsecond=0)
            for ts in timestamps.dropna().unique()
        ]
    hour_timestamps = hour_timestamps or []
    offer_statuses = get_offer_statuses_from_cache(
        cache_client, cache_prefix, hour_timestamps
    )
    working["offer_id_str"] = working["offer_id"].astype(str)
    dataset_offer_ids = working["offer_id_str"].unique()
    missing_offer_ids = [oid for oid in dataset_offer_ids if oid not in offer_statuses]

    if missing_offer_ids:
        fetched_statuses = fetch_offer_statuses(
            missing_offer_ids, offers_table=offers_table
        )
        offer_statuses.update(fetched_statuses)

        if fetched_statuses and hour_timestamps:
            most_recent_hour = max(hour_timestamps)
            cache_offer_statuses(cache_client, cache_prefix, most_recent_hour, fetched_statuses)

    working["offer_status"] = working["offer_id_str"].map(offer_statuses)
    working["offer_status"] = working["offer_status"].fillna("pending")
    working = working.drop(columns=["offer_id_str"], errors="ignore")
    return working


def clear_data_cache() -> None:
    _DATA_CACHE.clear()


__all__ = [
    "DEFAULT_ACCEPTANCE_TABLE",
    "DEFAULT_OFFERS_TABLE",
    "clear_data_cache",
    "enrich_with_offer_status",
    "fetch_offer_statuses",
    "get_offer_statuses_from_cache",
    "get_redshift_connection",
    "load_dataset_from_source",
    "offer_status_cache_key",
    "parse_recent_hours",
]
