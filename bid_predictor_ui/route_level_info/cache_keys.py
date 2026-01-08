from datetime import datetime

def audit_raw_cache_key(start: datetime, end: datetime) -> str:
    return f"audit:raw:{start:%Y-%m-%d}:{end:%Y-%m-%d}"

def route_metrics_cache_key(
    start: datetime,
    end: datetime,
    carrier: str,
    threshold: float,
) -> str:
    return (
        f"audit:routes:{carrier}:"
        f"{start:%Y-%m-%d}:{end:%Y-%m-%d}:"
        f"thr={threshold:.2f}"
    )
