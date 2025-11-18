"""Prices cache helpers: find_cache_meta(), cache_read_prices(), cache_append_prices()."""

from __future__ import annotations

import re
from typing import List, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from .secrets import load_secrets
from .sheets_client import get_gspread_client, open_ws

CACHE_CANON = "PricesCache"


def _norm_cols(cols: Sequence[str]) -> List[str]:
    return [
        str(c)
        .strip()
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        for c in cols
    ]


def _parse_number(x) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (pd.Timestamp,)):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = re.sub(r"[^\d,\.\-\s]", "", s).replace(" ", "")
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if "," in s:
            s = s.replace(",", ".")
    if s in ("", ".", "-", "-.", ".-"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


@st.cache_resource(show_spinner=False)
def find_cache_meta() -> dict:
    """Detect the PricesCache worksheet and its column names."""
    secrets = load_secrets()
    client = get_gspread_client()
    sh = client.open_by_key(secrets["sheet_id"])
    candidates = []
    for ws in sh.worksheets():
        vals = ws.get_all_values()
        if not vals or len(vals) < 1:
            continue
        header = _norm_cols(vals[0])
        cols = {c: i for i, c in enumerate(header)}
        date_keys = [k for k in ("date", "fecha") if k in cols]
        ticker_keys = [k for k in ("ticker", "symbol", "simbolo") if k in cols]
        close_keys = [
            k
            for k in ("close", "cierre", "adj close", "adj_close", "adjclose")
            if k in cols
        ]
        score = int(len(date_keys) > 0) + int(len(ticker_keys) > 0) + int(
            len(close_keys) > 0
        )
        if score >= 2:
            candidates.append((ws.title, date_keys[:1], ticker_keys[:1], close_keys[:1]))
    for c in candidates:
        if c[0].strip().lower() == CACHE_CANON.lower():
            return {
                "sheet": c[0],
                "date_col": c[1][0] if c[1] else "date",
                "ticker_col": c[2][0] if c[2] else "ticker",
                "close_col": c[3][0] if c[3] else "close",
            }
    if candidates:
        c = candidates[0]
        return {
            "sheet": c[0],
            "date_col": c[1][0] if c[1] else "date",
            "ticker_col": c[2][0] if c[2] else "ticker",
            "close_col": c[3][0] if c[3] else "close",
        }
    ws = sh.add_worksheet(title=CACHE_CANON, rows=1000, cols=3)
    ws.update("A1:C1", [["Date", "Ticker", "Close"]])
    return {"sheet": CACHE_CANON, "date_col": "date", "ticker_col": "ticker", "close_col": "close"}


def _get_cache_ws():
    meta = find_cache_meta()
    return open_ws(meta["sheet"]), meta


@st.cache_data(ttl=1200, show_spinner=False)
def cache_read_prices(tickers, start_date=None):
    ws, meta = _get_cache_ws()
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    header = _norm_cols(values[0])
    df = pd.DataFrame(values[1:], columns=header)
    dcol = meta["date_col"]
    tcol = meta["ticker_col"]
    ccol = meta["close_col"]
    for col in (dcol, tcol, ccol):
        if col not in df.columns:
            return pd.DataFrame()
    df = df[(df[dcol] != "") & (df[tcol] != "") & (df[ccol] != "")]
    if df.empty:
        return pd.DataFrame()
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
    df[tcol] = (
        df[tcol].astype(str).str.upper().str.strip().str.replace(" ", "", regex=False)
    )
    df[ccol] = df[ccol].map(_parse_number)
    df = df.dropna(subset=[dcol, tcol, ccol])
    if start_date:
        df = df[df[dcol] >= pd.to_datetime(start_date).normalize()]
    if tickers:

        def _clean_tickers(tks):
            out = []
            for t in tks:
                s = str(t).upper().strip().replace(" ", "")
                if s and s not in out:
                    out.append(s)
            return out

        df = df[df[tcol].isin(set(_clean_tickers(tickers)))]
    if df.empty:
        return pd.DataFrame()
    wide = (
        df.pivot_table(index=dcol, columns=tcol, values=ccol, aggfunc="last")
        .sort_index()
    )
    return wide[~wide.index.duplicated(keep="last")]


def cache_append_prices(df_wide):
    if df_wide is None or df_wide.empty:
        return 0
    ws, meta = _get_cache_ws()
    df_wide = df_wide.copy()
    df_wide.index = pd.to_datetime(df_wide.index).normalize()
    existing = cache_read_prices(list(df_wide.columns))
    to_write = df_wide.copy()
    if existing is not None and not existing.empty:
        merged = pd.concat([existing, to_write]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        mask_new = merged.loc[to_write.index, to_write.columns].where(
            ~existing.reindex(merged.index).notna(), other=np.nan
        )
        to_write = mask_new.dropna(how="all")
    if to_write.empty:
        return 0
    header_real = ws.get_all_values()[0]
    header_norm = _norm_cols(header_real)
    idx = {c: i for i, c in enumerate(header_norm)}
    dcol = meta["date_col"]
    tcol = meta["ticker_col"]
    ccol = meta["close_col"]
    if dcol not in idx or tcol not in idx or ccol not in idx:
        ws.update("A1:C1", [["Date", "Ticker", "Close"]])
        header_real = ["Date", "Ticker", "Close"]
        header_norm = _norm_cols(header_real)
        idx = {c: i for i, c in enumerate(header_norm)}
        dcol, tcol, ccol = "date", "ticker", "close"
    rows = []
    for dt_i, row in to_write.sort_index().iterrows():
        for t, val in row.items():
            if pd.isna(val):
                continue
            r = [""] * len(header_real)
            r[idx[dcol]] = pd.Timestamp(dt_i).strftime("%Y-%m-%d")
            r[idx[tcol]] = str(t)
            r[idx[ccol]] = float(val)
            rows.append(r)
    batch = 800
    total = 0
    for i in range(0, len(rows), batch):
        ws.append_rows(rows[i : i + batch], value_input_option="RAW")
        total += len(rows[i : i + batch])
    return total
