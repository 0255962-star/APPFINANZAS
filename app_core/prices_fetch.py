"""Price utilities: normalize_symbol(), fetch_yahoo_range(), ensure_prices()."""

from __future__ import annotations

import os
import time
from datetime import timezone
from typing import Iterable, List

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from .cache_prices import cache_append_prices, cache_read_prices

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
HOSTS = ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]
os.environ.setdefault("YF_USER_AGENT", UA)


def normalize_symbol(t: str) -> str:
    """Normalize tickers to the expected Yahoo format."""
    if not isinstance(t, str):
        return ""
    s = t.upper().strip().replace(" ", "")
    s = s.replace("BRK.B", "BRK-B").replace("BF.B", "BF-B")
    s = s.replace("/", "-")
    return s


def _clean_tickers(tickers: Iterable[str]) -> List[str]:
    out = []
    for t in tickers:
        s = normalize_symbol(t)
        if s and s not in out:
            out.append(s)
    return out


def _unix(dt_) -> int:
    return int(pd.Timestamp(dt_, tz=timezone.utc).timestamp())


def _parse_chart_json(js) -> pd.Series:
    result = js.get("chart", {}).get("result", [])
    if not result:
        return pd.Series(dtype=float)
    r = result[0]
    ts = r.get("timestamp", [])
    if not ts:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(pd.Series(ts), unit="s", utc=True).dt.tz_convert(
        None
    ).normalize()
    try:
        adj = r.get("indicators", {}).get("adjclose", [])
        if adj and "adjclose" in adj[0]:
            vals = adj[0]["adjclose"]
            return pd.Series(vals, index=idx, dtype="float64").dropna()
    except Exception:
        pass
    try:
        q = r.get("indicators", {}).get("quote", [])
        if q and "close" in q[0]:
            vals = q[0]["close"]
            return pd.Series(vals, index=idx, dtype="float64").dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


def _direct_one(ticker, start=None, end=None, interval="1d", timeout=25):
    params = {
        "interval": interval,
        "includeAdjustedClose": "true",
        "events": "div,splits",
    }
    if start is None and end is None:
        params["range"] = "max"
    else:
        if start:
            params["period1"] = _unix(start)
        params["period2"] = int(pd.Timestamp.utcnow().timestamp())
    headers = {"User-Agent": UA, "Accept": "application/json,text/plain,*/*"}
    last_err = None
    for host in HOSTS:
        url = f"{host}/v8/finance/chart/{ticker}"
        for k in range(3):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=timeout)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}"
                    time.sleep(0.6 * (k + 1))
                    continue
                s = _parse_chart_json(r.json())
                if isinstance(s, pd.Series) and not s.empty and len(s.dropna()) >= 3:
                    s.name = ticker
                    return s, None
                last_err = "respuesta vacía"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(0.6 * (k + 1))
    return pd.Series(dtype=float), last_err


def fetch_yahoo_range(ticker, start, end=None):
    try:
        d = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if d is None or d.empty:
            return pd.Series(dtype=float)
        s = pd.Series(d["Close"]).dropna()
        s.index = pd.to_datetime(s.index).normalize()
        s.name = ticker
        return s
    except Exception:
        return pd.Series(dtype=float)


def ensure_prices(tickers, start: str, persist: bool = True) -> pd.DataFrame:
    """
    Guarantee cache coverage for tickers since a start date and return the data frame.
    """

    from .masters import _rebuild_prices_masters_light

    tlist = _clean_tickers(tickers)
    cached = cache_read_prices(tlist, start)
    need = [
        t
        for t in tlist
        if (
            cached.empty
            or t not in cached.columns
            or cached[t].dropna().empty
        )
    ]
    if not need:
        return cached.sort_index()
    new_cols = []
    for t in need:
        s = fetch_yahoo_range(t, start=start, end=None)
        if s is None or s.empty:
            s2, err = _direct_one(t, start=start, end=None)
            if s2 is None or s2.empty:
                st.warning(f"No pude obtener datos de {t} (Yahoo): {err or 'sin datos'}.")
                continue
            s = s2
        if s is not None and not s.empty:
            new_cols.append(s)
    if new_cols:
        df_new = pd.concat(new_cols, axis=1).sort_index()
        if persist:
            try:
                cache_append_prices(df_new)
            except Exception as e:
                st.warning(f"Se descargó pero no se pudo escribir en PricesCache: {e}")
        for k in ("prices_master", "bench_ret_master", "last_map_master"):
            st.session_state.pop(k, None)
        _rebuild_prices_masters_light()
    final = cache_read_prices(tlist, start)
    return final.sort_index()
