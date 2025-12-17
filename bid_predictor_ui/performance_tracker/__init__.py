"""Performance tracker tab components."""
from __future__ import annotations

from .layout import build_performance_tab
from .view import register_performance_callbacks

__all__ = [
    "build_performance_tab",
    "register_performance_callbacks",
]
