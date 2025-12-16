"""Shared helpers for snapshot dropdowns and labels."""
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd


def build_snapshot_options(values: Optional[Iterable[object]]) -> List[dict]:
    """Return dropdown options with ``"Snapshot <n>"`` labels in numeric order.

    The helper accepts any iterable (including a pandas Series), removes empty
    entries and duplicates, sorts the remaining snapshot numbers numerically, and
    returns options compatible with Dash dropdowns.
    """

    if values is None:
        return []

    series = pd.Series(values).dropna().drop_duplicates()
    if series.empty:
        return []

    numeric = pd.to_numeric(series, errors="coerce")
    order = pd.Series(range(len(series)))
    sort_keys = numeric.where(numeric.notna(), order)
    sorted_series = series.iloc[sort_keys.argsort(kind="mergesort")]

    options: List[dict] = []
    for value in sorted_series:
        label_value = value
        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric_value):
            if float(numeric_value).is_integer():
                label_value = int(numeric_value)
            else:
                label_value = numeric_value
        options.append({"label": f"Snapshot {label_value}", "value": str(value)})
    return options


__all__ = ["build_snapshot_options"]
