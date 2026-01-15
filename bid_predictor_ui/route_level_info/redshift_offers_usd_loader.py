from __future__ import annotations

import requests
import pandas as pd
from typing import Iterable
from datetime import date
import psycopg2
import os

CURRENCY_SERVICE_URL = (
    "https://currency.stg.internal.plusgrade.com"
    "/currency-app/service/currency/json/{src}/USD"
)

_currency_rate_cache: dict[str, float] = {}

REDSHIFT_CONN = {
    "host": os.environ["REDSHIFT_HOST"],
    "port": 5439,
    "dbname": os.environ["REDSHIFT_DATABASE"],
    "user": os.environ["REDSHIFT_USER"],
    "password": os.environ["REDSHIFT_PASSWORD"],
}

def _get_usd_rate(cur_code: str) -> float:
    cur_code = cur_code.upper()

    if cur_code == "USD":
        return 1.0

    if cur_code in _currency_rate_cache:
        return _currency_rate_cache[cur_code]

    resp = requests.get(
        CURRENCY_SERVICE_URL.format(src=cur_code),
        timeout=5,
    )
    resp.raise_for_status()

    rate = float(resp.json()["rate"])
    _currency_rate_cache[cur_code] = rate
    return rate


def load_offers_with_usd_amount(
    *,
    start_date: date,
    end_date: date,
    carriers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch offer-level data from Redshift and compute USD-normalized value.
    """

    carrier_filter_sql = ""
    if carriers:
        carrier_list = ",".join(f"'{c}'" for c in carriers)
        carrier_filter_sql = f"AND o.operating_carrier IN ({carrier_list})"

    query = f"""
        SELECT
            o.id          AS offer_id,
            o.operating_carrier AS carrier_code,
            o.travel_dt,
            fi.origin     AS origination,
            fi.dest       AS destination,
            o.amount,
            o.item_count,
            o.cur_code,
            o.offer_status
        FROM prd_offers_rds.offers o
        LEFT JOIN prd_offers_rds.flightinventory fi
            ON o.product_id = fi.id
        WHERE o.travel_dt BETWEEN %(start_date)s AND %(end_date)s
          {carrier_filter_sql}
          AND o.offer_status != 'CANCELLED'
    """

    # with get_redshift_connection() as conn:
    #     df = pd.read_sql(
    #         query,
    #         con=conn,
    #         params={
    #             "start_date": start_date,
    #             "end_date": end_date,
    #         },
    #     )

    with psycopg2.connect(**REDSHIFT_CONN) as conn:
        df = pd.read_sql(query, conn, params={"start_date": start_date, "end_date": end_date})

    if df.empty:
        df["offer_value_usd"] = []
        return df

    df["offer_value_native"] = (
        df["item_count"].astype(float) * df["amount"].astype(float)
    )

    df["offer_value_usd"] = df.apply(
        lambda r: r["offer_value_native"] * _get_usd_rate(r["cur_code"]),
        axis=1,
    )

    return df
