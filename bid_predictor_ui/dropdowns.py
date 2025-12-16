"""Shared helpers for building and selecting dropdown options."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def options_from_series(values: pd.Series) -> List[Dict[str, str]]:
    """Convert a pandas series into a list of Dash dropdown options."""

    if values is None:
        return []

    return [
        {"label": str(value), "value": str(value)}
        for value in values.dropna().drop_duplicates().sort_values()
    ]


def choose_dropdown_value(
    options: List[Dict[str, str]],
    requested_value: Optional[str],
    current_value: Optional[str],
) -> Optional[str]:
    """Pick the dropdown value that best matches the current context."""

    def _contains(value: Optional[str]) -> bool:
        return bool(value) and any(option["value"] == value for option in options)

    if _contains(requested_value):
        return requested_value
    if _contains(current_value):
        return current_value
    return options[0]["value"] if options else None


__all__ = ["options_from_series", "choose_dropdown_value"]
