"""Interactive Dash UI to explore bid acceptance predictions from an MLflow model."""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Optional

import io
import threading
import time
from datetime import datetime, timedelta
import mlflow
import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dcc, html
from mlflow.exceptions import MlflowException
from dotenv import load_dotenv
from pyarrow import fs as pyfs
try:  # pragma: no cover - optional dependency in some environments
    import redis  # type: ignore[import]
except ImportError:  # pragma: no cover
    redis = None
# from bid_predictor.utils import detect_execution_environment

from bid_predictor_ui import (
    DEFAULT_UI_FEATURE_CONFIG,
    build_model_name_options,
    build_model_stage_or_version_options,
    build_ui_feature_config,
    load_dataset_cached,
    load_model_cached,
)
from bid_predictor_ui.data_sources import (
    DEFAULT_ACCEPTANCE_TABLE,
    _cache_key,
    _normalize_config,
    _filter_files_by_recent_hours,
    _list_remote_files,
    cache_offer_statuses,
    enrich_with_offer_status,
    fetch_offer_statuses,
)
from bid_predictor_ui.feature_sensitivity import (
    build_feature_sensitivity_tab,
    register_feature_sensitivity_callbacks,
)
from bid_predictor_ui.acceptance_explorer import (
    build_acceptance_tab,
    load_acceptance_dataset,
    register_acceptance_callbacks,
)
from bid_predictor_ui.snapshot import (
    build_snapshot_tab,
    register_snapshot_callbacks,
)
from bid_predictor_ui.performance_tracker import (
    build_performance_tab,
    register_performance_callbacks,
)
from bid_predictor_ui.performance_history import (
    build_performance_history_tab,
    register_performance_history_callbacks,
)
from bid_predictor_ui.performance_history.data import (
    update_performance_history_from_source,
)
from bid_predictor_ui import data_sources as data_sources_module
from bid_predictor_ui.acceptance_explorer.view import _normalize_acceptance_dataset

load_dotenv()

arn = os.environ["MLFLOW_AWS_ARN"]
mlflow.set_tracking_uri(arn)
default_dataset_path = os.environ.get("DEFAULT_DATASET_PATH")


# Acceptance dataset configuration via S3 listing and Redis cache
S3_DATASET_LISTING_URI = os.environ.get("S3_DATASET_LISTING_URI")
DEFAULT_S3_LOOKBACK_HOURS = int(os.getenv("S3_DATASET_LOOKBACK_HOURS", "120"))
PERFORMANCE_HISTORY_S3_URI = os.getenv("PERFORMANCE_HISTORY_S3_URI")
PERFORMANCE_HISTORY_REFRESH_DAYS = int(os.getenv("PERFORMANCE_HISTORY_REFRESH_DAYS", "3"))
REDIS_URL = os.getenv("REDIS_URL")
# Rolling window cache: automatically refresh data every hour for this many hours
ROLLING_WINDOW_HOURS = int(os.getenv("ROLLING_WINDOW_HOURS", "240"))

def _get_redis_client() -> Optional["redis.Redis"]:
    """Return a Redis client if configured, otherwise None."""

    if not REDIS_URL or redis is None:  # type: ignore[truthy-function]
        return None
    try:
        return redis.Redis.from_url(REDIS_URL)  # type: ignore[no-any-return]
    except Exception:  # pragma: no cover - defensive
        return None


def _acceptance_cache_key(hours: int) -> str:
    """Legacy cache key for full window caching."""
    prefix = S3_DATASET_LISTING_URI or ""
    return f"acceptance_dataset:{prefix}:{hours}"


def _hour_bucket_cache_key(hour_timestamp: pd.Timestamp) -> str:
    """Cache key for a specific hour bucket."""
    prefix = S3_DATASET_LISTING_URI or ""
    hour_str = hour_timestamp.strftime("%Y-%m-%dT%H")
    return f"acceptance_dataset_hour:{prefix}:{hour_str}"


def _get_hour_buckets_for_window(hours: int) -> list[pd.Timestamp]:
    """Get list of hour timestamps for the requested window."""
    now = pd.Timestamp.now()
    buckets = []
    for i in range(hours):
        hour_ts = now - pd.Timedelta(hours=i)
        # Round down to the hour
        hour_ts = hour_ts.replace(minute=0, second=0, microsecond=0)
        buckets.append(hour_ts)
    return sorted(set(buckets))  # Remove duplicates and sort


def _fetch_hour_data_from_s3(hour_timestamp: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Fetch data for a specific hour from S3 by loading files and filtering by timestamp."""
    if not S3_DATASET_LISTING_URI:
        return None
    
    try:
        filesystem = pyfs.S3FileSystem()
        all_files = _list_remote_files(filesystem, S3_DATASET_LISTING_URI)
        
        # Load files that might contain data for this hour (files from last 2 hours to be safe)
        hour_start = hour_timestamp
        hour_end = hour_timestamp + pd.Timedelta(hours=1)
        lookback_start = hour_start - pd.Timedelta(hours=2)
        
        candidate_files = []
        for file_path in all_files:
            file_ts = _extract_timestamp_from_filename(file_path)
            if file_ts and lookback_start <= file_ts < hour_end:
                candidate_files.append(file_path)
        
        if not candidate_files:
            return pd.DataFrame()  # Empty DataFrame for hours with no data
        
        # Load and combine files, then filter by data timestamps
        frames = []
        for remote_path in candidate_files:
            suffix = remote_path.split(".")[-1].lower()
            try:
                with filesystem.open_input_file(remote_path) as handle:
                    if suffix in {"parquet", "pq"}:
                        frames.append(pd.read_parquet(handle))
                    elif suffix == "csv":
                        frames.append(pd.read_csv(handle))
            except Exception as exc:
                print(f"[Hourly refresh] Failed to load file {remote_path}: {exc}")
                continue
        
        if not frames:
            return pd.DataFrame()
        
        combined = pd.concat(frames, ignore_index=True, sort=False)
        
        # Filter by timestamp column in the data
        timestamp_cols = ["accept_prob_timestamp", "current_timestamp", "created_timestamp"]
        timestamp_col = next((col for col in timestamp_cols if col in combined.columns), None)
        
        if timestamp_col:
            timestamps = pd.to_datetime(combined[timestamp_col], errors="coerce")
            hour_mask = (timestamps >= hour_start) & (timestamps < hour_end)
            hour_data = combined[hour_mask].copy()
            return hour_data
        else:
            # No timestamp column, return all data (fallback)
            return combined
    except Exception as exc:
        print(f"[Hourly refresh] Failed to fetch hour {hour_timestamp}: {exc}")
        return None


def _extract_timestamp_from_filename(file_path: str) -> Optional[pd.Timestamp]:
    """Extract timestamp from S3 file path."""
    from pathlib import PurePosixPath
    from bid_predictor_ui.data_sources import _extract_timestamp_from_name
    
    filename = PurePosixPath(file_path).name
    return _extract_timestamp_from_name(filename)


def _refresh_hourly_cache() -> None:
    """Background task: refresh hourly cache buckets - only fetch newest hour."""
    if not S3_DATASET_LISTING_URI:
        return
    
    cache_client = _get_redis_client()
    if cache_client is None:
        print("[Hourly refresh] Redis not configured, skipping hourly refresh.")
        return
    
    print(f"[Hourly refresh] Starting refresh for {ROLLING_WINDOW_HOURS}h rolling window...")
    
    # Only fetch the NEWEST hour (current hour)
    now = pd.Timestamp.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    
    cache_key = _hour_bucket_cache_key(current_hour)
    cached = cache_client.get(cache_key)
    
    if not cached:
        # Fetch only the newest hour's data from S3
        hour_data = _fetch_hour_data_from_s3(current_hour)
        if hour_data is not None and not hour_data.empty:
            try:
                buffer = io.BytesIO()
                hour_data.to_parquet(buffer, index=False)
                # Cache for 2 hours (longer than refresh interval)
                cache_client.setex(cache_key, 7200, buffer.getvalue())
                print(
                    f"[Hourly refresh] Cached newest hour {current_hour.strftime('%Y-%m-%d %H:00')} "
                    f"({len(hour_data):,} rows)."
                )
                
                # Fetch and cache offer_status for this hour
                if "offer_id" in hour_data.columns:
                    offer_ids = hour_data["offer_id"].dropna().astype(str).unique().tolist()
                    if offer_ids:
                        offer_statuses = fetch_offer_statuses(offer_ids)
                        if offer_statuses:
                            cache_offer_statuses(
                                cache_client,
                                S3_DATASET_LISTING_URI or "",
                                current_hour,
                                offer_statuses,
                            )
            except Exception as exc:
                print(f"[Hourly refresh] Failed to cache hour {current_hour}: {exc}")
        else:
            print(f"[Hourly refresh] No data found for hour {current_hour.strftime('%Y-%m-%d %H:00')}.")
    else:
        print(f"[Hourly refresh] Hour {current_hour.strftime('%Y-%m-%d %H:00')} already cached, skipping.")
    
    # Remove old buckets outside the rolling window (both data and offer_status)
    try:
        prefix = S3_DATASET_LISTING_URI or ""
        now = pd.Timestamp.now()
        oldest_allowed = now - pd.Timedelta(hours=ROLLING_WINDOW_HOURS + 1)
        
        # Clean data buckets
        data_pattern = f"acceptance_dataset_hour:{prefix}:*"
        all_data_keys = cache_client.keys(data_pattern)
        for key in all_data_keys:
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            try:
                hour_str = key.split(":")[-1]
                hour_ts = pd.to_datetime(hour_str, format="%Y-%m-%dT%H")
                if hour_ts < oldest_allowed:
                    cache_client.delete(key)
                    print(f"[Hourly refresh] Removed old data bucket: {hour_str}")
            except Exception:
                pass
        
        # Clean offer_status buckets
        status_pattern = f"offer_status_hour:{prefix}:*"
        all_status_keys = cache_client.keys(status_pattern)
        for key in all_status_keys:
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            try:
                hour_str = key.split(":")[-1]
                hour_ts = pd.to_datetime(hour_str, format="%Y-%m-%dT%H")
                if hour_ts < oldest_allowed:
                    cache_client.delete(key)
                    print(f"[Hourly refresh] Removed old offer_status bucket: {hour_str}")
            except Exception:
                pass
    except Exception as exc:
        print(f"[Hourly refresh] Failed to clean old buckets: {exc}")
    
    print(f"[Hourly refresh] Refresh complete. Next refresh in 1 hour.")


def _background_refresh_worker() -> None:
    """Background worker thread that refreshes cache every hour."""
    # Initial refresh after 30 seconds (to let app start)
    time.sleep(30)
    
    while True:
        try:
            _refresh_hourly_cache()
        except Exception as exc:
            print(f"[Hourly refresh] Error in background refresh: {exc}")
        
        # Wait 1 hour before next refresh
        time.sleep(3600)


def _load_from_hour_buckets(hours: int) -> Optional[pd.DataFrame]:
    """Load dataset by combining hour buckets from cache."""
    cache_client = _get_redis_client()
    if cache_client is None:
        return None
    
    required_buckets = _get_hour_buckets_for_window(hours)
    frames = []
    missing_buckets = []
    
    for hour_ts in required_buckets:
        cache_key = _hour_bucket_cache_key(hour_ts)
        cached = cache_client.get(cache_key)
        
        if cached:
            try:
                buffer = io.BytesIO(cached)
                hour_data = pd.read_parquet(buffer)
                if not hour_data.empty:
                    frames.append(hour_data)
            except Exception as exc:
                print(f"[Hourly cache] Failed to load bucket {hour_ts}: {exc}")
                missing_buckets.append(hour_ts)
        else:
            missing_buckets.append(hour_ts)
    
    if missing_buckets:
        print(
            f"[Hourly cache] Missing {len(missing_buckets)} buckets, "
            f"will fetch from S3: {[b.strftime('%Y-%m-%d %H:00') for b in missing_buckets[:5]]}"
        )
        return None  # Indicate we need to fetch from S3
    
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        print(
            f"[Hourly cache] Combined {len(frames)} hour buckets "
            f"({len(combined):,} rows) for {hours}h window."
        )
        # Enrich with offer_status
        combined = enrich_with_offer_status(
            combined,
            cache_client=cache_client,
            cache_prefix=S3_DATASET_LISTING_URI or "",
            hour_timestamps=required_buckets,
        )
        return combined
    
    return pd.DataFrame()  # Empty result


def _populate_acceptance_cache(
    dataset: pd.DataFrame,
    dataset_config: dict,
    *,
    normalize_offer_status: bool = True,
) -> None:
    """Populate the internal acceptance dataset cache so dropdowns work."""
    try:
        normalized_dataset = dataset.copy()
        if normalize_offer_status:
            normalized_dataset = enrich_with_offer_status(
                normalized_dataset,
                cache_client=_get_redis_client(),
                cache_prefix=S3_DATASET_LISTING_URI or "",
            )
        normalized_dataset = _normalize_acceptance_dataset(normalized_dataset)
        
        # Build the cache key the same way load_dataset_from_source does
        normalized_config = _normalize_config(dataset_config)
        cache_key = _cache_key(normalized_config, _normalize_acceptance_dataset)
        
        # Store in the internal cache
        data_sources_module._DATA_CACHE[cache_key] = normalized_dataset
        print(
            f"[Acceptance loader] Populated internal cache for {dataset_config.get('hours', 'N/A')}h window "
            f"({len(normalized_dataset):,} rows)."
        )
    except Exception as exc:
        print(f"[Acceptance loader] Warning: Failed to populate internal cache: {exc}")


def _maybe_update_performance_history() -> None:
    """Update the performance history parquet if configured."""
    if not PERFORMANCE_HISTORY_S3_URI or not S3_DATASET_LISTING_URI:
        return
    cache_client = _get_redis_client()
    try:
        update_performance_history_from_source(
            PERFORMANCE_HISTORY_S3_URI,
            S3_DATASET_LISTING_URI,
            refresh_days=PERFORMANCE_HISTORY_REFRESH_DAYS,
            cache_client=cache_client,
        )
        print("[Performance history] Updated performance history file.")
    except Exception as exc:  # pragma: no cover - non-blocking background update
        print(f"[Performance history] Failed to update history file: {exc}")


# -- Dash application --------------------------------------------------------------------------


def create_app() -> Dash:
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )
    model_name_options = build_model_name_options()
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "Bid Predictor Playground",
                        style={"margin": "0", "color": "#1b4965"},
                    ),
                    html.P(
                        "Load a dataset snapshot and an MLflow-registered model to explore acceptance probabilities.",
                        style={"margin": "0", "color": "#16324f"},
                    ),
                ],
                className="app-hero",
                style={
                    "background": "linear-gradient(90deg, #e0fbfc 0%, #c2dfe3 100%)",
                    "padding": "1.5rem",
                    "borderRadius": "12px",
                    "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
                    "marginBottom": "1.5rem",
                },
            ),
            dcc.Tabs(
                id="main-tabs",
                value="snapshot",
                children=[
                    dcc.Tab(label="Snapshot explorer", value="snapshot"),
                    dcc.Tab(label="Feature sensitivity", value="sensitivity"),
                    dcc.Tab(label="Acceptance explorer", value="acceptance"),
                    dcc.Tab(label="Performance tracker", value="performance"),
                    dcc.Tab(label="Performance history", value="history"),
                ],
                style={"marginTop": "1rem", "marginBottom": "1rem"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Dataset path", style={"fontWeight": "600"}
                                    ),
                                    dcc.Input(
                                        id="dataset-path",
                                        type="text",
                                        value=default_dataset_path,
                                        placeholder="Path to bid_data_snapshots_v2.parquet",
                                        style={
                                            "width": "100%",
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    html.Button(
                                        "Load dataset",
                                        id="load-dataset",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "backgroundColor": "#1b4965",
                                            "color": "white",
                                            "border": "none",
                                            "padding": "0.6rem",
                                            "borderRadius": "6px",
                                        },
                                    ),
                                    html.Button(
                                        "Reload dataset",
                                        id="reload-dataset",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "marginTop": "0.5rem",
                                            "backgroundColor": "#457b9d",
                                            "color": "white",
                                            "border": "none",
                                            "padding": "0.5rem",
                                            "borderRadius": "6px",
                                        },
                                    ),
                                    dcc.Loading(
                                        id="dataset-loading",
                                        type="circle",
                                        children=html.Div(
                                            id="dataset-status",
                                            className="status-message",
                                            style={"marginTop": "0.5rem"},
                                        ),
                                    ),
                                ],
                                className="control-card",
                                style={
                                    "flex": "1",
                                    "padding": "1rem",
                                    "backgroundColor": "#f7fff7",
                                    "borderRadius": "12px",
                                    "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.05)",
                                },
                            ),
                            html.Div(
                                [
                                    html.Label("Model name", style={"fontWeight": "600"}),
                                    dcc.Dropdown(
                                        id="model-name",
                                        options=model_name_options,
                                        placeholder="Select a registered model",
                                        value=None,
                                        clearable=True,
                                        style={
                                            "width": "100%",
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    html.Label(
                                        "Model stage or version",
                                        style={"fontWeight": "600"},
                                    ),
                                    dcc.Dropdown(
                                        id="model-stage",
                                        options=[],
                                        placeholder="Select a stage or version",
                                        value=None,
                                        clearable=True,
                                        disabled=True,
                                        style={
                                            "width": "100%",
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    html.Button(
                                        "Load model",
                                        id="load-model",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "backgroundColor": "#ff6b6b",
                                            "color": "white",
                                            "border": "none",
                                            "padding": "0.6rem",
                                            "borderRadius": "6px",
                                        },
                                    ),
                                    dcc.Loading(
                                        id="model-loading",
                                        type="circle",
                                        children=html.Div(
                                            id="model-status",
                                            className="status-message",
                                            style={"marginTop": "0.5rem"},
                                        ),
                                    ),
                                ],
                                className="control-card",
                                style={
                                    "flex": "1",
                                    "padding": "1rem",
                                    "backgroundColor": "#f7fff7",
                                    "borderRadius": "12px",
                                    "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.05)",
                                },
                            ),
                        ],
                        id="standard-controls",
                        className="controls-grid",
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "gap": "1.5rem",
                            "marginBottom": "1.5rem",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Dataset path (optional)",
                                        style={"fontWeight": "600"},
                                    ),
                                    dcc.Input(
                                        id="acceptance-dataset-path",
                                        type="text",
                                        value="",
                                        placeholder="Path to acceptance dataset (.csv/.parquet)",
                                        style={
                                            "width": "100%",
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    html.Button(
                                        "Load file path",
                                        id="acceptance-path-apply",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "backgroundColor": "#1b4965",
                                            "color": "white",
                                            "border": "none",
                                            "padding": "0.5rem",
                                            "borderRadius": "6px",
                                            "marginBottom": "0.75rem",
                                        },
                                    ),
                                ],
                                style={"width": "100%"},
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Lookback (hours)",
                                        style={
                                            "fontWeight": "600",
                                            "marginRight": "0.5rem",
                                        },
                                    ),
                                    dcc.Input(
                                        id="acceptance-lookback-hours",
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=DEFAULT_S3_LOOKBACK_HOURS,
                                        style={
                                            "width": "5rem",
                                        },
                                    ),
                                    html.Button(
                                        "Apply",
                                        id="acceptance-lookback-apply",
                                        n_clicks=0,
                                        style={
                                            "marginLeft": "0.75rem",
                                            "padding": "0.25rem 0.75rem",
                                            "borderRadius": "6px",
                                            "border": "none",
                                            "backgroundColor": "#1b4965",
                                            "color": "white",
                                            "cursor": "pointer",
                                        },
                                    ),
                                ],
                                style={
                                    "marginTop": "0.5rem",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "width": "100%",
                                },
                            ),
                            html.Div(
                                id="acceptance-dataset-status",
                                style={
                                    "marginTop": "0.5rem",
                                    "width": "100%",
                                },
                            ),
                            html.Div(  # <-- forces block layout
                                dcc.Loading(
                                    id="acceptance-loader",
                                    type="circle",
                                    children=html.Div(
                                        id="acceptance-loader-status",
                                        style={"marginTop": "0.5rem"},
                                    ),
                                ),
                                style={"width": "100%"},
                            ),                                
                        ],
                        className="control-card",
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "background": "linear-gradient(90deg, #e0fbfc 0%, #c2dfe3 100%)",
                            "padding": "1.5rem",
                            "borderRadius": "12px",
                            "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
                            "marginBottom": "1.5rem",
                        },
                        id="acceptance-controls",                    
                    ),
                ],
                style={
                    "background": "linear-gradient(90deg, #e0fbfc 0%, #c2dfe3 100%)",
                    "padding": "1.5rem",
                    "borderRadius": "12px",
                    "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
                    "marginBottom": "1.5rem",
                },                          
            ),
            dcc.Store(id="dataset-path-store"),
            dcc.Store(id="acceptance-dataset-path-store"),
            dcc.Store(id="model-uri-store"),
            dcc.Store(id="bid-records-store"),
            dcc.Store(id="snapshot-meta-store"),
            dcc.Store(id="prediction-store"),
            dcc.Store(id="removed-bids-store"),
            dcc.Store(id="baseline-bid-records-store"),
            dcc.Store(id="baseline-snapshot-meta-store"),
            dcc.Store(id="scenario-records-store"),
            dcc.Store(id="scenario-original-records-store"),
            dcc.Store(id="scenario-removed-bids-store"),
            dcc.Store(id="snapshot-selection-request-store"),
            dcc.Store(id="selection-history-store", data=[]),
            dcc.Store(id="scenario-selection-request-store"),
            dcc.Store(id="scenario-selection-history-store", data=[]),
            dcc.Store(id="acceptance-selection-request-store"),
            dcc.Store(id="acceptance-selection-history-store", data=[]),
            dcc.Store(
                id="feature-config-store",
                data=deepcopy(DEFAULT_UI_FEATURE_CONFIG),
            ),
            dcc.Interval(
                id="acceptance-loader-interval",
                interval=500,
                n_intervals=0,
                max_intervals=1,
            ),            
            html.Div(
                id="tab-content",
                children=build_snapshot_tab().children,
            ),
        ],
        className="app-shell",
        style={
            "fontFamily": "'Segoe UI', sans-serif",
            "backgroundColor": "#fafafa",
            "padding": "1.5rem",
        },
    )

    register_snapshot_callbacks(app)
    register_feature_sensitivity_callbacks(app)
    register_acceptance_callbacks(app)
    register_performance_callbacks(app)
    register_performance_history_callbacks(app)

    # Start background hourly refresh thread
    if REDIS_URL and S3_DATASET_LISTING_URI:
        refresh_thread = threading.Thread(
            target=_background_refresh_worker,
            daemon=True,
            name="HourlyCacheRefresh"
        )
        refresh_thread.start()
        print(
            f"[Hourly refresh] Background refresh thread started. "
            f"Will refresh {ROLLING_WINDOW_HOURS}h rolling window every hour."
        )

    # Callbacks -----------------------------------------------------------------------------

    @app.callback(
        Output("standard-controls", "style"),
        Output("acceptance-controls", "style"),
        Input("main-tabs", "value"),
    )
    def toggle_control_panels(active_tab: str):
        standard_style = {
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "1.5rem",
            "marginBottom": "1.5rem",
        }
        acceptance_style = {
            "display": "none",
            "flexWrap": "wrap",
            "gap": "1.5rem",
            "marginBottom": "1.5rem",
        }
        if active_tab in {"acceptance", "performance"}:
            standard_style["display"] = "none"
            acceptance_style["display"] = "flex"
        if active_tab == "history":
            standard_style["display"] = "none"
        return standard_style, acceptance_style


    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab_content(active_tab: str):
        if active_tab == "snapshot":
            return build_snapshot_tab().children
        if active_tab == "sensitivity":
            return build_feature_sensitivity_tab().children
        if active_tab == "acceptance":
            return build_acceptance_tab().children
        if active_tab == "performance":
            return build_performance_tab().children
        if active_tab == "history":
            return build_performance_history_tab().children
        return build_snapshot_tab().children


    @app.callback(
        Output("dataset-status", "children"),
        Output("dataset-path-store", "data"),
        Input("load-dataset", "n_clicks"),
        Input("reload-dataset", "n_clicks"),
        State("dataset-path", "value"),
        prevent_initial_call=True,
    )
    def load_dataset(load_clicks: int, reload_clicks: int, path: str):
        if not path:
            return "Please provide a dataset path.", None

        triggered = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered
            else ""
        )
        reload_flag = triggered == "reload-dataset"

        try:
            dataset = load_dataset_cached(path, reload=reload_flag)
        except Exception as exc:  # pragma: no cover - user feedback
            return f"Failed to load dataset: {exc}", None

        status_prefix = "Reloaded" if reload_flag else "Loaded"
        status = f"{status_prefix} dataset with {len(dataset):,} rows."
        return status, path


    @app.callback(
        Output("acceptance-dataset-status", "children"),
        Output("acceptance-dataset-path-store", "data"),
        Output("acceptance-loader-status", "children"),
        Input("acceptance-loader-interval", "n_intervals"),
        Input("acceptance-lookback-apply", "n_clicks"),
        Input("acceptance-path-apply", "n_clicks"),
        State("acceptance-lookback-hours", "value"),
        State("acceptance-dataset-path", "value"),
        prevent_initial_call=False,
    )
    def load_acceptance_dataset_on_startup(
        n_intervals: int,
        apply_clicks: int,
        path_clicks: int,
        lookback_value: Optional[int],
        custom_path: Optional[str],
    ):
        path_value = (custom_path or "").strip()

        # Resolve lookback window in hours from user input
        try:
            hours = int(lookback_value) if lookback_value is not None else DEFAULT_S3_LOOKBACK_HOURS
        except (TypeError, ValueError):
            hours = DEFAULT_S3_LOOKBACK_HOURS
        if hours <= 0:
            hours = DEFAULT_S3_LOOKBACK_HOURS

        # Determine trigger source: initial interval vs user "Apply" click.
        trigger = ""
        if callback_context.triggered:
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        cache_client = _get_redis_client()

        if path_value:
            reload_flag = trigger == "acceptance-path-apply"
            dataset_config = {
                "source": "path",
                "path": path_value,
                "hours": None,
            }
            try:
                dataset = load_acceptance_dataset(
                    dataset_config,
                    cache_client=cache_client,
                    cache_prefix=S3_DATASET_LISTING_URI or "",
                    reload=reload_flag,
                )
            except Exception as exc:  # pragma: no cover - user feedback
                error_msg = f"Failed to load acceptance dataset: {exc}"
                return error_msg, None, error_msg

            status = f"Loaded acceptance dataset from {path_value} with {len(dataset):,} rows."
            loader_status = f"Acceptance data loaded from {path_value} ({len(dataset):,} rows)."
            _populate_acceptance_cache(dataset, dataset_config, normalize_offer_status=False)
            _maybe_update_performance_history()
            return status, dataset_config, loader_status

        if not S3_DATASET_LISTING_URI:
            message = "S3_DATASET_LISTING_URI is not configured."
            return message, None, message
        
        # First, try loading from hour buckets (new rolling window cache)
        if hours <= ROLLING_WINDOW_HOURS:
            bucket_data = _load_from_hour_buckets(hours)
            if bucket_data is not None:
                # Successfully loaded from hour buckets
                print(
                    f"[Acceptance loader] Loaded from hour buckets for {hours}h window "
                    f"({len(bucket_data):,} rows)."
                )
                status = (
                    f"Loaded acceptance dataset from cache (last {hours} hours) "
                    f"with {len(bucket_data):,} rows."
                )
                loader_status = (
                    f"Acceptance data loaded from hourly cache ({len(bucket_data):,} rows)."
                )
                dataset_config = {
                    "source": "path",
                    "path": S3_DATASET_LISTING_URI,
                    "hours": hours,
                }
                normalize_offer_status = "offer_status" not in bucket_data.columns
                _populate_acceptance_cache(
                    bucket_data,
                    dataset_config,
                    normalize_offer_status=normalize_offer_status,
                )
                _maybe_update_performance_history()
                return status, dataset_config, loader_status
        
        # Fall back to legacy full-window cache or S3
        cache_key = _acceptance_cache_key(hours)

        # Try Redis cache - exact match for requested hours (legacy)
        if cache_client is not None:
            cached = cache_client.get(cache_key)
            if cached:
                try:
                    buffer = io.BytesIO(cached)
                    dataset = pd.read_parquet(buffer)
                    if "offer_status" not in dataset.columns:
                        dataset = enrich_with_offer_status(
                            dataset,
                            cache_client=cache_client,
                            cache_prefix=S3_DATASET_LISTING_URI or "",
                        )
                    print(
                        f"[Acceptance loader] Using cached dataset from Redis for "
                        f"last {hours} hours ({len(dataset):,} rows)."
                    )
                    status = (
                        f"Loaded acceptance dataset from cache (last {hours} hours) "
                        f"with {len(dataset):,} rows."
                    )
                    loader_status = (
                        f"Acceptance data loaded from Redis cache ({len(dataset):,} rows)."
                    )
                    dataset_config = {
                        "source": "path",
                        "path": S3_DATASET_LISTING_URI,
                        "hours": hours,
                    }
                    # Populate internal cache so dropdowns work
                    normalize_offer_status = "offer_status" not in dataset.columns
                    _populate_acceptance_cache(
                        dataset,
                        dataset_config,
                        normalize_offer_status=normalize_offer_status,
                    )
                    _maybe_update_performance_history()
                    return status, dataset_config, loader_status
                except Exception as exc:  # pragma: no cover - cache decode issues
                    print(f"[Acceptance loader] Failed to read cached dataset: {exc}")
            
            # Optimization: Check if we have a larger cached window that we can reuse
            # Try common larger windows in ascending order (e.g., if requesting 20h, check 24h, 48h, etc.)
            larger_windows = sorted([24, 48, 72, 168])  # 1 day, 2 days, 3 days, 1 week
            for larger_hours in larger_windows:
                if larger_hours > hours:
                    larger_cache_key = _acceptance_cache_key(larger_hours)
                    larger_cached = cache_client.get(larger_cache_key)
                    if larger_cached:
                        try:
                            buffer = io.BytesIO(larger_cached)
                            larger_dataset = pd.read_parquet(buffer)
                            
                            # Check if dataset has timestamp columns we can filter by
                            timestamp_cols = ["accept_prob_timestamp", "current_timestamp", "created_timestamp"]
                            timestamp_col = next((col for col in timestamp_cols if col in larger_dataset.columns), None)
                            
                            if timestamp_col:
                                # Filter to requested hours window
                                timestamps = pd.to_datetime(larger_dataset[timestamp_col], errors="coerce")
                                if not timestamps.isna().all():
                                    latest_time = timestamps.max()
                                    cutoff_time = latest_time - pd.Timedelta(hours=hours)
                                    filtered_dataset = larger_dataset[timestamps >= cutoff_time].copy()
                                    
                                    if len(filtered_dataset) > 0:
                                        print(
                                            f"[Acceptance loader] Reusing cached {larger_hours}h dataset, "
                                            f"filtered to {hours}h ({len(filtered_dataset):,} rows from {len(larger_dataset):,})."
                                        )
                                        
                                        # Cache the filtered result for future use
                                        try:
                                            filter_buffer = io.BytesIO()
                                            filtered_dataset.to_parquet(filter_buffer, index=False)
                                            ttl_seconds = max(hours, 1) * 3600
                                            cache_client.setex(cache_key, ttl_seconds, filter_buffer.getvalue())
                                            print(
                                                f"[Acceptance loader] Cached filtered dataset in Redis for {hours} hours "
                                                f"(TTL {ttl_seconds} seconds)."
                                            )
                                        except Exception as exc:
                                            print(f"[Acceptance loader] Failed to cache filtered dataset: {exc}")
                                        
                                        if "offer_status" not in filtered_dataset.columns:
                                            filtered_dataset = enrich_with_offer_status(
                                                filtered_dataset,
                                                cache_client=cache_client,
                                                cache_prefix=S3_DATASET_LISTING_URI or "",
                                            )
                                        
                                        status = (
                                            f"Loaded acceptance dataset from cache (filtered from {larger_hours}h to {hours}h) "
                                            f"with {len(filtered_dataset):,} rows."
                                        )
                                        loader_status = (
                                            f"Acceptance data loaded from Redis cache (filtered from {larger_hours}h, "
                                            f"{len(filtered_dataset):,} rows)."
                                        )
                                        dataset_config = {
                                            "source": "path",
                                            "path": S3_DATASET_LISTING_URI,
                                            "hours": hours,
                                        }
                                        # Populate internal cache so dropdowns work
                                        normalize_offer_status = "offer_status" not in filtered_dataset.columns
                                        _populate_acceptance_cache(
                                            filtered_dataset,
                                            dataset_config,
                                            normalize_offer_status=normalize_offer_status,
                                        )
                                        _maybe_update_performance_history()
                                        return status, dataset_config, loader_status
                                    else:
                                        print(
                                            f"[Acceptance loader] Cached {larger_hours}h dataset filtered to empty "
                                            f"for {hours}h window, will fetch from S3."
                                        )
                                else:
                                    print(
                                        f"[Acceptance loader] Cached {larger_hours}h dataset has no valid timestamps, "
                                        f"will fetch from S3."
                                    )
                            else:
                                print(
                                    f"[Acceptance loader] Cached {larger_hours}h dataset has no timestamp columns, "
                                    f"will fetch from S3."
                                )
                        except Exception as exc:
                            print(f"[Acceptance loader] Failed to reuse larger cached dataset ({larger_hours}h): {exc}")
                            continue

        # No cache hit – load from S3 and populate cache
        print(
            f"[Acceptance loader] No cache found for {hours}h window. "
            f"Fetching from S3 (each lookback window is cached separately)."
        )
        try:
            filesystem = pyfs.S3FileSystem()
            all_files = _list_remote_files(filesystem, S3_DATASET_LISTING_URI)
            filtered_files = _filter_files_by_recent_hours(all_files, hours)

            print(
                f"[Acceptance loader] Found {len(filtered_files)} files under "
                f"{S3_DATASET_LISTING_URI} within last {hours} hours."
            )
            for path in filtered_files:
                print(f"[Acceptance loader] Using file: s3://{path}")
        except Exception as exc:
            error_msg = f"Failed to list S3 files: {exc}"
            return error_msg, None, error_msg

        dataset_config = {
            "source": "path",
            "path": S3_DATASET_LISTING_URI,
            "hours": hours,
        }

        hour_buckets = (
            _get_hour_buckets_for_window(hours) if hours <= ROLLING_WINDOW_HOURS else []
        )
        try:
            dataset = load_acceptance_dataset(
                dataset_config,
                cache_client=cache_client,
                cache_prefix=S3_DATASET_LISTING_URI or "",
                hour_timestamps=hour_buckets if hour_buckets else None,
                reload=True,
            )
        except Exception as exc:  # pragma: no cover - user feedback
            error_msg = f"Failed to load acceptance dataset: {exc}"
            return error_msg, None, error_msg

        # Write to Redis cache with TTL equal to hours window (in seconds)
        if cache_client is not None:
            try:
                buffer = io.BytesIO()
                dataset.to_parquet(buffer, index=False)
                ttl_seconds = 7200
                cache_client.setex(cache_key, ttl_seconds, buffer.getvalue())
                print(
                    f"[Acceptance loader] Cached dataset in Redis for {hours} hours "
                    f"(TTL {ttl_seconds} seconds, {len(dataset):,} rows)."
                )
            except Exception as exc:  # pragma: no cover - cache write issues
                print(f"[Acceptance loader] Failed to cache dataset in Redis: {exc}")
            
            # Also populate hour buckets if within rolling window
            if hours <= ROLLING_WINDOW_HOURS and "accept_prob_timestamp" in dataset.columns:
                try:
                    timestamps = pd.to_datetime(dataset["accept_prob_timestamp"], errors="coerce")
                    required_buckets = _get_hour_buckets_for_window(hours)
                    
                    for hour_ts in required_buckets:
                        hour_start = hour_ts
                        hour_end = hour_ts + pd.Timedelta(hours=1)
                        hour_mask = (timestamps >= hour_start) & (timestamps < hour_end)
                        hour_data = dataset[hour_mask].copy()
                        
                        if not hour_data.empty:
                            bucket_key = _hour_bucket_cache_key(hour_ts)
                            bucket_buffer = io.BytesIO()
                            hour_data.to_parquet(bucket_buffer, index=False)
                            cache_client.setex(bucket_key, 7200, bucket_buffer.getvalue())
                            print(
                                f"[Acceptance loader] Populated hour bucket "
                                f"{hour_ts.strftime('%Y-%m-%d %H:00')} ({len(hour_data):,} rows)."
                            )
                            
                            # Also cache offer_status for this hour if we have offer_ids
                            if "offer_id" in hour_data.columns and "offer_status" in hour_data.columns:
                                offer_statuses = dict(
                                    zip(
                                        hour_data["offer_id"].astype(str),
                                        hour_data["offer_status"]
                                    )
                                )
                                cache_offer_statuses(
                                    cache_client,
                                    S3_DATASET_LISTING_URI or "",
                                    hour_ts,
                                    offer_statuses,
                                )
                except Exception as exc:
                    print(f"[Acceptance loader] Failed to populate hour buckets: {exc}")

        file_count = len(filtered_files)
        status = (
            f"Loaded acceptance dataset from last {hours} hours "
            f"with {len(dataset):,} rows."
        )
        loader_status = (
            f"Acceptance data loaded from S3 ({file_count} files, {len(dataset):,} rows)."
        )
        # Populate internal cache so dropdowns work
        normalize_offer_status = "offer_status" not in dataset.columns
        _populate_acceptance_cache(
            dataset,
            dataset_config,
            normalize_offer_status=normalize_offer_status,
        )
        _maybe_update_performance_history()
        return status, dataset_config, loader_status


    @app.callback(
        Output("model-stage", "options"),
        Output("model-stage", "value"),
        Output("model-stage", "disabled"),
        Input("model-name", "value"),
    )
    def update_model_stage_options(model_name: Optional[str]):
        if not model_name:
            return [], None, True

        options = build_model_stage_or_version_options(model_name)
        return options, None, False


    @app.callback(
        Output("model-status", "children"),
        Output("model-uri-store", "data"),
        Output("feature-config-store", "data"),
        Input("load-model", "n_clicks"),
        State("model-name", "value"),
        State("model-stage", "value"),
        prevent_initial_call=True,
    )
    def load_model(n_clicks: int, model_name: str, stage_or_version: str):
        if not model_name:
            return (
                "Select a registered model name.",
                None,
                deepcopy(DEFAULT_UI_FEATURE_CONFIG),
            )

        if not stage_or_version:
            return (
                "Select a model stage or version.",
                None,
                deepcopy(DEFAULT_UI_FEATURE_CONFIG),
            )

        model_uri: Optional[str] = None
        try:
            stage_or_version = stage_or_version.strip()
            model_uri = f"models:/{model_name}/{stage_or_version}"
            model = load_model_cached(model_uri)
            raw_config = getattr(model, "feature_config_", None) or getattr(
                model, "feature_config", None
            )
            ui_feature_config = build_ui_feature_config(raw_config)
        except MlflowException as exc:  # pragma: no cover - user feedback
            return (
                f"Failed to load model: {exc}",
                None,
                deepcopy(DEFAULT_UI_FEATURE_CONFIG),
            )
        except Exception as exc:  # pragma: no cover
            return (
                f"Unexpected error while loading model: {exc}",
                None,
                deepcopy(DEFAULT_UI_FEATURE_CONFIG),
            )

        return f"Loaded model from {model_uri}", model_uri, ui_feature_config

    return app


def main():  # pragma: no cover - manual entry point
    app = create_app()
    # app.run_server(debug=True)
    port = int(os.getenv("PORT", 8000))  # App Runner passes a PORT sometimes, but default is fine
    app.run_server(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
