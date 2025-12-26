"""Helpers for performance history metrics and storage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs as pyfs

DEFAULT_THRESHOLD = 0.5
HISTORY_DATE_COLUMN = "history_date"
ALL_CARRIER_VALUE = "ALL"


@dataclass(frozen=True)
class PerformanceHistoryConfig:
    history_uri: str
    threshold: float = DEFAULT_THRESHOLD


def _is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def _split_s3_uri(uri: str) -> str:
    return uri.replace("s3://", "", 1)


def _file_exists(uri: str) -> bool:
    if _is_s3_uri(uri):
        filesystem = pyfs.S3FileSystem()
        info = filesystem.get_file_info([_split_s3_uri(uri)])[0]
        return info.type == pyfs.FileType.File

    return Path(uri).expanduser().exists()


def _write_parquet(uri: str, frame: pd.DataFrame) -> None:
    if _is_s3_uri(uri):
        filesystem = pyfs.S3FileSystem()
        with filesystem.open_output_stream(_split_s3_uri(uri)) as handle:
            table = pa.Table.from_pandas(frame)
            pq.write_table(table, handle)
        return

    resolved = Path(uri).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(resolved, index=False)


def _read_parquet(uri: str) -> pd.DataFrame:
    if _is_s3_uri(uri):
        filesystem = pyfs.S3FileSystem()
        with filesystem.open_input_file(_split_s3_uri(uri)) as handle:
            return pd.read_parquet(handle)

    resolved = Path(uri).expanduser()
    return pd.read_parquet(resolved)


def _normalize_accept_probabilities(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    non_na = numeric.dropna()
    if not non_na.empty and non_na.max() <= 1:
        numeric = numeric * 100.0
    return numeric


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator.replace({0: np.nan})
    return ratio.where(np.isfinite(ratio))


def _summarize_groups(
    working: pd.DataFrame, group_keys: Iterable[str]
) -> pd.DataFrame:
    grouped = working.groupby(list(group_keys), dropna=False)
    summary = grouped.agg(
        total=("offer_status", "size"),
        actual_pos=("actual_positive", "sum"),
        predicted_pos=("predicted_positive", "sum"),
        tp=("tp", "sum"),
        tn=("tn", "sum"),
        fp=("fp", "sum"),
        fn=("fn", "sum"),
    )

    summary["actual_neg"] = summary["total"] - summary["actual_pos"]
    summary["predicted_neg"] = summary["total"] - summary["predicted_pos"]

    pos = summary["tp"] + summary["fn"]
    neg = summary["tn"] + summary["fp"]

    summary["accuracy"] = _safe_ratio(summary["tp"] + summary["tn"], summary["total"])
    summary["precision"] = _safe_ratio(summary["tp"], summary["tp"] + summary["fp"])
    summary["recall"] = _safe_ratio(summary["tp"], summary["tp"] + summary["fn"])
    summary["negative_precision"] = _safe_ratio(summary["tn"], summary["tn"] + summary["fn"])
    summary["negative_recall"] = _safe_ratio(summary["tn"], summary["tn"] + summary["fp"])
    summary["false_positive_rate"] = _safe_ratio(summary["fp"], neg)
    summary["false_negative_rate"] = _safe_ratio(summary["fn"], pos)
    summary["balanced_accuracy"] = _safe_ratio(
        summary["recall"] + summary["negative_recall"],
        pd.Series([2] * len(summary), index=summary.index),
    )
    summary["prevalence"] = _safe_ratio(pos, summary["total"])
    summary["f_score"] = _safe_ratio(
        2 * summary["precision"] * summary["recall"],
        summary["precision"] + summary["recall"],
    )
    summary["negative_f_score"] = _safe_ratio(
        2 * summary["negative_precision"] * summary["negative_recall"],
        summary["negative_precision"] + summary["negative_recall"],
    )
    summary["fm_index"] = np.sqrt(summary["precision"] * summary["recall"])
    summary["negative_fm_index"] = np.sqrt(
        summary["negative_precision"] * summary["negative_recall"]
    )

    summary = summary.reset_index()
    count_columns = [
        "total",
        "actual_pos",
        "actual_neg",
        "predicted_pos",
        "predicted_neg",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    summary[count_columns] = summary[count_columns].fillna(0).astype(int)
    return summary


def compute_daily_performance_history(
    dataset: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()

    if "accept_prob_timestamp" not in dataset.columns:
        return pd.DataFrame()

    working = dataset.copy()
    working = working[working["offer_status"].isin(["TICKETED", "EXPIRED"])]
    working["accept_prob"] = _normalize_accept_probabilities(working["accept_prob"])
    working["accept_prob_timestamp"] = pd.to_datetime(
        working["accept_prob_timestamp"], errors="coerce"
    )
    working[HISTORY_DATE_COLUMN] = working["accept_prob_timestamp"].dt.normalize()
    working = working.dropna(subset=[HISTORY_DATE_COLUMN, "accept_prob", "offer_status"])

    if working.empty:
        return pd.DataFrame()

    if "carrier_code" not in working.columns:
        working["carrier_code"] = ALL_CARRIER_VALUE
    else:
        working["carrier_code"] = working["carrier_code"].astype(str).fillna("Unknown")

    working["actual_positive"] = working["offer_status"] == "TICKETED"
    working["predicted_positive"] = working["accept_prob"] >= threshold * 100
    working["tp"] = working["actual_positive"] & working["predicted_positive"]
    working["tn"] = ~working["actual_positive"] & ~working["predicted_positive"]
    working["fp"] = ~working["actual_positive"] & working["predicted_positive"]
    working["fn"] = working["actual_positive"] & ~working["predicted_positive"]

    carrier_summary = _summarize_groups(working, [HISTORY_DATE_COLUMN, "carrier_code"])

    if "carrier_code" in dataset.columns:
        overall = working.copy()
        overall["carrier_code"] = ALL_CARRIER_VALUE
        overall_summary = _summarize_groups(overall, [HISTORY_DATE_COLUMN, "carrier_code"])
        summary = pd.concat([carrier_summary, overall_summary], ignore_index=True)
    else:
        summary = carrier_summary

    summary["threshold"] = float(threshold)
    summary = summary.sort_values([HISTORY_DATE_COLUMN, "carrier_code"]).reset_index(
        drop=True
    )
    return summary


def load_performance_history(history_uri: str) -> pd.DataFrame:
    if not _file_exists(history_uri):
        return pd.DataFrame()

    history = _read_parquet(history_uri)
    if HISTORY_DATE_COLUMN in history.columns:
        history[HISTORY_DATE_COLUMN] = pd.to_datetime(
            history[HISTORY_DATE_COLUMN], errors="coerce"
        )
    return history


def update_performance_history_file(
    dataset: pd.DataFrame,
    history_uri: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> Optional[pd.DataFrame]:
    new_history = compute_daily_performance_history(dataset, threshold=threshold)
    if new_history.empty:
        return None

    if not _file_exists(history_uri):
        _write_parquet(history_uri, new_history)
        return new_history

    existing = load_performance_history(history_uri)
    if existing.empty:
        _write_parquet(history_uri, new_history)
        return new_history

    if HISTORY_DATE_COLUMN not in existing.columns:
        _write_parquet(history_uri, new_history)
        return new_history

    latest_date = pd.to_datetime(existing[HISTORY_DATE_COLUMN], errors="coerce").max()
    if pd.isna(latest_date):
        _write_parquet(history_uri, new_history)
        return new_history

    append_rows = new_history[new_history[HISTORY_DATE_COLUMN] > latest_date]
    if append_rows.empty:
        return existing

    combined = pd.concat([existing, append_rows], ignore_index=True)
    combined = combined.sort_values([HISTORY_DATE_COLUMN, "carrier_code"]).reset_index(
        drop=True
    )
    _write_parquet(history_uri, combined)
    return combined

