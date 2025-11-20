"""Session masters: get_setting(), build_masters(), _rebuild_prices_masters_light(), helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from .cache_prices import cache_append_prices, cache_read_prices
from .prices_fetch import _direct_one, fetch_yahoo_range, normalize_symbol
from .sheets_client import read_sheet

SESSION_TTL_MIN = 30


def _coerce_tx_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure transaction numeric columns are coerced for reliable calculations."""
    if df is None or df.empty:
        return df
    tx_df = df.copy()
    num_cols = ["Shares", "Price", "Fees", "Taxes", "FXRate", "GrossAmount", "NetAmount"]
    for col in num_cols:
        if col in tx_df.columns:
            tx_df[col] = pd.to_numeric(tx_df[col], errors="coerce")
    return tx_df


def get_setting(settings_df, key, default=None, cast=float):
    try:
        s = settings_df.loc[settings_df["Key"] == key, "Value"]
        if s.empty:
            return default
        return cast(s.values[0]) if cast else s.values[0]
    except Exception:
        return default


def need_build(name: str) -> bool:
    return name not in st.session_state


def masters_expired() -> bool:
    ts = st.session_state.get("_masters_built_at")
    if ts is None:
        return True
    return (datetime.utcnow() - ts).total_seconds() > SESSION_TTL_MIN * 60


def build_masters(sync: bool):
    tx_df = _coerce_tx_numeric(read_sheet("Transactions"))
    settings_df = read_sheet("Settings")
    benchmark = get_setting(settings_df, "Benchmark", "SPY", str)
    tickers = sorted(
        set(
            t
            for t in tx_df.get("Ticker", pd.Series(dtype=str))
            .astype(str)
            .apply(normalize_symbol)
            if t
        )
    )
    all_t = list(dict.fromkeys((tickers or []) + [benchmark]))
    if sync:
        cache = cache_read_prices(all_t, start_date=None)
        today = pd.Timestamp(datetime.utcnow().date())
        for t in all_t:
            have = pd.Series(dtype=float)
            if cache is not None and t in cache.columns:
                have = cache[t].dropna()
            if have.empty:
                start = (today - pd.Timedelta(days=365 * 3)).strftime("%Y-%m-%d")
                s = fetch_yahoo_range(t, start=start, end=None)
                if s.empty:
                    s2, _ = _direct_one(t, start=start, end=None)
                    if not s2.empty:
                        cache_append_prices(s2.to_frame())
                else:
                    cache_append_prices(s.to_frame())
            else:
                last = have.index.max()
                if pd.Timestamp(last).normalize() < today:
                    start = (
                        pd.to_datetime(last).normalize() + pd.Timedelta(days=1)
                    ).strftime("%Y-%m-%d")
                    s = fetch_yahoo_range(t, start=start, end=None)
                    if s.empty:
                        s2, _ = _direct_one(t, start=start, end=None)
                        if not s2.empty:
                            cache_append_prices(s2.to_frame())
                    else:
                        cache_append_prices(s.to_frame())
    prices = cache_read_prices(all_t, start_date=None)
    bench_ret = pd.Series(dtype=float)
    if prices is not None and not prices.empty and benchmark in prices.columns:
        bench_ret = prices[[benchmark]].pct_change().dropna()[benchmark]
        prices = prices.drop(columns=[benchmark], errors="ignore")
    last_map = {}
    if prices is not None and not prices.empty:
        last_map = prices.ffill().iloc[-1].dropna().astype(float).to_dict()
    st.session_state["tx_master"] = tx_df
    st.session_state["settings_master"] = settings_df
    st.session_state["prices_master"] = prices if prices is not None else pd.DataFrame()
    st.session_state["bench_ret_master"] = bench_ret
    st.session_state["last_map_master"] = last_map
    st.session_state["_masters_built_at"] = datetime.utcnow()


def _rebuild_prices_masters_light():
    """Reconstruye en sesión los masters que dependen de PricesCache sin tocar Transactions ni Settings."""
    # FIX: evitar evaluar DataFrames como booleanos para no disparar ValueError y lecturas innecesarias.
    settings_df = st.session_state.get("settings_master")
    if settings_df is None or settings_df.empty:
        settings_df = read_sheet("Settings")
    benchmark = get_setting(settings_df, "Benchmark", "SPY", str)

    tx_df = st.session_state.get("tx_master")
    if tx_df is None or tx_df.empty:
        tx_df = read_sheet("Transactions")
    tx_df = _coerce_tx_numeric(tx_df)
    tickers = sorted(
        set(
            t
            for t in tx_df.get("Ticker", pd.Series(dtype=str))
            .astype(str)
            .str.upper()
            .str.strip()
            .str.replace(" ", "", regex=False)
            if t
        )
    )
    all_t = list(dict.fromkeys((tickers or []) + [benchmark]))
    prices = cache_read_prices(all_t, start_date=None)
    bench_ret = pd.Series(dtype=float)
    if prices is not None and not prices.empty and benchmark in prices.columns:
        bench_ret = prices[[benchmark]].pct_change().dropna()[benchmark]
        prices = prices.drop(columns=[benchmark], errors="ignore")
    last_map = {}
    if prices is not None and not prices.empty:
        last_map = prices.ffill().iloc[-1].dropna().astype(float).to_dict()
    st.session_state["prices_master"] = prices if prices is not None else pd.DataFrame()
    st.session_state["bench_ret_master"] = bench_ret
    st.session_state["last_map_master"] = last_map
