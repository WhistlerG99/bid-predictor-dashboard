"""Shared loaders for path and Redshift-backed datasets.

The helpers here are intentionally UI-focused: they support the flexible file
and Redshift sources that individual tabs previously reimplemented on their
own.  Caching is opt-in with a reload flag so dashboards can refresh data when
files change without restarting the app.
"""
from __future__ import annotations

import os
import re
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

import pandas as pd
import psycopg2
from pyarrow import fs as pyfs

DEFAULT_ACCEPTANCE_TABLE = "model_prediction_testing.audit_bid_predictor"

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


def clear_data_cache() -> None:
    _DATA_CACHE.clear()


__all__ = [
    "DEFAULT_ACCEPTANCE_TABLE",
    "clear_data_cache",
    "load_dataset_from_source",
    "parse_recent_hours",
]
