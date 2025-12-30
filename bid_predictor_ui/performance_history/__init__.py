"""Performance history tab components."""
from __future__ import annotations

from .layout import build_performance_history_tab
from .view import register_performance_history_callbacks

__all__ = [
    "build_performance_history_tab",
    "register_performance_history_callbacks",
]
