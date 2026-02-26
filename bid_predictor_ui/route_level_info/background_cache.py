"""
Background data pre-fetching and caching system.
Automatically fetches and caches data for all carriers and periods
so users don't have to wait on first request.
"""

import threading
import time
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .data_loader import load_audit_data_cached
from .route_offer_revenue_cache import get_cached_route_offer_revenue
from .route_metrics_cache import get_cached_route_metrics, set_cached_route_metrics
from .redshift_loader import load_offer_statuses_cached
from .metrics import compute_bucket_metrics
import pandas as pd
import numpy as np
import os

# Carrier-specific thresholds (from callbacks)
CARRIER_THRESHOLDS = {
    "EY": float(os.environ.get("ACCEPT_PROB_THRESHOLD_EY", 0.2)),
    "SV": float(os.environ.get("ACCEPT_PROB_THRESHOLD_SV", 0.1)),
}

# Periods to pre-cache
PERIODS_TO_CACHE = [7, 14, 21, 30]

# Horizons to compute
HORIZONS = [72, 48, 24]

scheduler = None
cache_lock = threading.Lock()


def get_threshold_for_carrier(carrier: str) -> float:
    """Get the acceptance probability threshold for a specific carrier."""
    return CARRIER_THRESHOLDS.get(carrier, 0.2)


def prefetch_and_cache_data():
    """
    Pre-fetch and cache data for all carriers and periods.
    This runs in the background and doesn't block user requests.
    """
    print("\n" + "="*80)
    print(f"[BACKGROUND CACHE] Starting prefetch at {datetime.utcnow().isoformat()}")
    print("="*80)
    
    with cache_lock:
        try:
            # Get all carriers from S3 data (using default 7-day fetch)
            df_all = load_audit_data_cached(days=6)  # Default 7-day window
            
            if df_all.empty:
                print("[BACKGROUND CACHE] No audit data available")
                return
            
            carriers = sorted(df_all["carrier_code"].dropna().unique())
            print(f"[BACKGROUND CACHE] Found {len(carriers)} carriers: {carriers}")
            
            # Iterate through each period and carrier
            total_routes_cached = 0
            
            for period_days in PERIODS_TO_CACHE:
                print(f"\n[BACKGROUND CACHE] ===== Processing Period: {period_days} days =====")
                
                # Load S3 data for this period
                s3_days_param = period_days - 1
                df = load_audit_data_cached(days=s3_days_param)
                
                if df.empty:
                    print(f"[BACKGROUND CACHE] No data for {period_days}-day period")
                    continue
                
                print(f"[BACKGROUND CACHE] Loaded {len(df)} rows for {period_days}-day period from S3")
                
                for carrier in carriers:
                    print(f"\n[BACKGROUND CACHE] --- Carrier: {carrier}, Period: {period_days}d ---")
                    
                    # Skip if already cached and valid
                    threshold = get_threshold_for_carrier(carrier)
                    cached = get_cached_route_metrics(carrier, threshold, period_days)
                    if cached is not None:
                        print(f"[BACKGROUND CACHE] {carrier} period={period_days}d already cached, skipping")
                        continue
                    
                    try:
                        # Filter by carrier
                        df_carrier = df[df["carrier_code"] == carrier].copy()
                        
                        if df_carrier.empty:
                            print(f"[BACKGROUND CACHE] No data for {carrier}")
                            continue
                        
                        # Parse timestamps
                        if "departure_timestamp" in df_carrier.columns:
                            df_carrier["departure_timestamp"] = pd.to_datetime(df_carrier["departure_timestamp"])
                        if "accept_prob_timestamp" in df_carrier.columns:
                            df_carrier["accept_prob_timestamp"] = pd.to_datetime(df_carrier["accept_prob_timestamp"])
                        
                        # Apply period-based departure_timestamp filter
                        from datetime import timedelta, time as dt_time
                        today = datetime.utcnow().date()
                        end_date = today - timedelta(days=1)
                        lookback_days = period_days - 1
                        start_date = end_date - timedelta(days=lookback_days)
                        
                        end_ts = datetime.combine(end_date, dt_time.max)
                        start_ts = datetime.combine(start_date, dt_time.min)
                        
                        df_carrier = df_carrier[
                            (df_carrier["departure_timestamp"] >= start_ts) & 
                            (df_carrier["departure_timestamp"] <= end_ts)
                        ]
                        
                        if df_carrier.empty:
                            print(f"[BACKGROUND CACHE] {carrier} period={period_days}d: No data after filtering")
                            continue
                        
                        print(f"[BACKGROUND CACHE] {carrier} period={period_days}d: {len(df_carrier)} rows after filter")
                        
                        # Route key
                        df_carrier["route"] = df_carrier["origination_code"] + "-" + df_carrier["destination_code"]
                        
                        # Load offer statuses
                        print(f"[BACKGROUND CACHE] {carrier} period={period_days}d: Loading offer statuses...")
                        offer_ids = df_carrier["offer_id"].dropna().unique().tolist()
                        status_df = load_offer_statuses_cached(offer_ids)
                        df_carrier = df_carrier.merge(status_df, on="offer_id", how="left")
                        
                        # Get route revenue data
                        print(f"[BACKGROUND CACHE] {carrier} period={period_days}d: Fetching route revenue...")
                        route_revenue_map = get_cached_route_offer_revenue(
                            carrier=carrier,
                            start_date=start_date,
                            end_date=end_date,
                            period_days=period_days,
                        )
                        
                        # Group by route and compute metrics
                        rows = []
                        for route, route_df in df_carrier.groupby("route"):
                            route_df_non_cancelled = route_df[
                                route_df["offer_status"] != "CANCELLED"
                            ]
                            
                            final_state = (
                                route_df_non_cancelled.sort_values("accept_prob_timestamp")
                                .groupby("offer_id", as_index=False)
                                .last()
                            )
                            
                            valid_final = final_state[
                                final_state["offer_status"].isin(
                                    ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY", "EXPIRED"]
                                )
                            ]
                            
                            accepted_mask = valid_final["offer_status"].isin(
                                ["TICKETED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"]
                            )
                            
                            origination, destination = route.split("-")
                            route_key = (origination, destination)
                            route_revenue = route_revenue_map.get(route_key)
                            
                            if route_revenue:
                                offers_usd = route_revenue["offers_usd"]
                                upgrades_usd = route_revenue["upgrades_usd"]
                                num_offers = route_revenue["offer_count"]
                                num_upgrades = route_revenue["num_actual_ticketed"]
                            else:
                                offers_usd = 0.0
                                upgrades_usd = 0.0
                                num_offers = 0
                                num_upgrades = 0
                            
                            acceptance_rate = (
                                upgrades_usd / offers_usd if offers_usd else np.nan
                            )
                            
                            horizon = compute_bucket_metrics(route_df, threshold)
                            
                            rows.append({
                                "route": route,
                                "offers_usd": round(offers_usd, 2),
                                "upgrades_usd": round(upgrades_usd, 2),
                                "acceptance_rate": round(acceptance_rate, 4),
                                "total_submitted_offers": int(num_offers),
                                "total_upgraded_offers": int(num_upgrades),
                                "offer_count_72h": horizon["72h"]["offer_count"],
                                "offer_count_48h": horizon["48h"]["offer_count"],
                                "offer_count_24h": horizon["24h"]["offer_count"],
                                "num_actual_ticketed_72h": horizon["72h"]["num_actual_ticketed"],
                                "num_actual_ticketed_48h": horizon["48h"]["num_actual_ticketed"],
                                "num_actual_ticketed_24h": horizon["24h"]["num_actual_ticketed"],
                                'num_actual_expired_72h': horizon["72h"]["num_actual_expired"],
                                'num_actual_expired_48h': horizon["48h"]["num_actual_expired"],
                                'num_actual_expired_24h': horizon["24h"]["num_actual_expired"],
                                "expiry_72h": horizon["72h"]["num_predicted_expired"],
                                "expiry_48h": horizon["48h"]["num_predicted_expired"],
                                "expiry_24h": horizon["24h"]["num_predicted_expired"],
                                "num_wrongly_expired_72h": horizon["72h"]["num_wrongly_expired"],
                                "negative_precision_72h": horizon["72h"]["negative_precision"],
                                "negative_recall_72h": horizon["72h"]["negative_recall"],
                                "num_wrongly_expired_48h": horizon["48h"]["num_wrongly_expired"],
                                "negative_precision_48h": horizon["48h"]["negative_precision"],
                                "negative_recall_48h": horizon["48h"]["negative_recall"],
                                "num_wrongly_expired_24h": horizon["24h"]["num_wrongly_expired"],
                                "negative_precision_24h": horizon["24h"]["negative_precision"],
                                "negative_recall_24h": horizon["24h"]["negative_recall"],
                            })
                        
                        if rows:
                            rows_df = pd.DataFrame(rows)
                            rows_dict = rows_df.to_dict("records")
                            set_cached_route_metrics(carrier, threshold, period_days, rows_dict)
                            print(f"[BACKGROUND CACHE] {carrier} period={period_days}d: Cached {len(rows_dict)} routes ✓")
                            
                            # ✅ ALSO CACHE THE FINAL RESULTS (for instant future loads)
                            try:
                                from ..utils.redis_client import get_redis_client
                                redis_client = get_redis_client()
                                final_cache_key = f"routes_table_final:{carrier}:threshold={threshold}:period={period_days}d"
                                redis_client.setex(
                                    final_cache_key,
                                    24 * 3600,  # 24 hour TTL
                                    json.dumps(rows_dict)
                                )
                                print(f"[BACKGROUND CACHE] {carrier} period={period_days}d: Cached final results ✓")
                            except Exception as e:
                                print(f"[BACKGROUND CACHE] Warning caching final results: {str(e)}")
                            
                            total_routes_cached += len(rows_dict)
                        else:
                            print(f"[BACKGROUND CACHE] {carrier} period={period_days}d: No routes to cache")
                    
                    except Exception as e:
                        print(f"[BACKGROUND CACHE ERROR] {carrier} period={period_days}d: {str(e)}")
                        import traceback
                        traceback.print_exc()
            
            print("\n" + "="*80)
            print(f"[BACKGROUND CACHE] Completed at {datetime.utcnow().isoformat()}")
            print(f"[BACKGROUND CACHE] Total routes cached: {total_routes_cached}")
            print("="*80 + "\n")
        
        except Exception as e:
            print(f"[BACKGROUND CACHE FATAL ERROR] {str(e)}")
            import traceback
            traceback.print_exc()


def start_background_cache_scheduler(cache_update_hour=2, cache_update_minute=0):
    """
    Start the background cache scheduler.
    
    Args:
        cache_update_hour: Hour of day to run cache (0-23, UTC). Default 2 AM.
        cache_update_minute: Minute of hour to run cache. Default 0.
    """
    global scheduler
    
    if scheduler is not None and scheduler.running:
        print("[BACKGROUND CACHE] Scheduler already running")
        return
    
    print(f"[BACKGROUND CACHE] Starting scheduler - will run daily at {cache_update_hour:02d}:{cache_update_minute:02d} UTC")
    
    scheduler = BackgroundScheduler()
    
    # Schedule daily cache refresh
    scheduler.add_job(
        prefetch_and_cache_data,
        trigger=CronTrigger(hour=cache_update_hour, minute=cache_update_minute),
        id='prefetch_cache_daily',
        name='Daily prefetch and cache',
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    
    try:
        scheduler.start()
        print("[BACKGROUND CACHE] Scheduler started successfully ✓")
        
        # Run startup pre-cache in a background thread (non-blocking)
        print("[BACKGROUND CACHE] Scheduling startup pre-cache (30 sec delay)...")
        startup_thread = threading.Thread(
            target=_startup_prefetch_wrapper,
            daemon=True,
            name="StartupPrefetchCache"
        )
        startup_thread.start()
        
    except Exception as e:
        print(f"[BACKGROUND CACHE ERROR] Failed to start scheduler: {str(e)}")


def _startup_prefetch_wrapper():
    """Wrapper for startup pre-cache with delay (runs in separate thread)"""
    import time
    time.sleep(30)  # Wait 30 seconds for app to fully initialize
    print("[BACKGROUND CACHE] Startup pre-cache starting...")
    prefetch_and_cache_data()


def stop_background_cache_scheduler():
    """Stop the background cache scheduler."""
    global scheduler
    
    if scheduler and scheduler.running:
        scheduler.shutdown()
        scheduler = None
        print("[BACKGROUND CACHE] Scheduler stopped")
