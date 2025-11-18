"""Transactions helpers: tidy_transactions(), positions_from_tx(), delete_transactions_by_ticker()."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from .prices_fetch import normalize_symbol
from .sheets_client import open_ws


def _is_dtlike(x) -> bool:
    return isinstance(x, (pd.Timestamp, datetime, date))


def _parse_number(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if _is_dtlike(x):
        return np.nan
    import re

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


def tidy_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Normalize the Transactions dataframe."""
    if tx.empty:
        return tx
    df = tx.copy()
    for c in [
        "TradeID",
        "Account",
        "Ticker",
        "Name",
        "AssetType",
        "Currency",
        "TradeDate",
        "Side",
        "Shares",
        "Price",
        "Fees",
        "Taxes",
        "FXRate",
        "GrossAmount",
        "NetAmount",
        "LotID",
        "Source",
        "Notes",
    ]:
        if c not in df.columns:
            df[c] = np.nan
    df["Ticker"] = df["Ticker"].astype(str).apply(normalize_symbol)
    df = df.dropna(subset=["Ticker"])
    df["TradeDate"] = pd.to_datetime(df["TradeDate"], errors="coerce").dt.date
    for col in [
        "Shares",
        "Price",
        "Fees",
        "Taxes",
        "FXRate",
        "GrossAmount",
        "NetAmount",
    ]:
        if col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = np.nan
            df[col] = df[col].map(_parse_number)
    if "FXRate" in df.columns:
        df["FXRate"] = df["FXRate"].replace(0, np.nan).fillna(1.0)
    for col in ["Fees", "Taxes"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
            df.loc[df[col].abs() > 1e7, col] = 0.0

    def signed(row):
        s = str(row.get("Side", "")).lower().strip()
        q = float(row.get("Shares", 0) or 0)
        if s in ("sell", "venta", "vender", "-1"):
            return -abs(q)
        return abs(q)

    df["SignedShares"] = df.apply(signed, axis=1)
    return df


def positions_from_tx(tx: pd.DataFrame, last_hint_map: Optional[dict] = None):
    """Aggregate transactions into positions."""
    if tx.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Shares",
                "AvgCost",
                "Invested",
                "MarketPrice",
                "MarketValue",
                "UnrealizedPL",
            ]
        )
    df = tidy_transactions(tx)
    uniq = sorted(df["Ticker"].unique().tolist())
    if not uniq:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Shares",
                "AvgCost",
                "Invested",
                "MarketPrice",
                "MarketValue",
                "UnrealizedPL",
            ]
        )
    last_map = dict(last_hint_map or {})
    pos = []
    for t, grp in df.groupby("Ticker"):
        sh = float(grp["SignedShares"].sum())
        if abs(sh) < 1e-12:
            continue
        buys = grp["SignedShares"] > 0
        if buys.any():
            tot_sh = float(grp.loc[buys, "SignedShares"].sum())
            cost_leg = (grp.loc[buys, "SignedShares"] * grp.loc[buys, "Price"].fillna(0)).sum()
            fees_leg = (
                grp.loc[buys, "Fees"].fillna(0).sum() + grp.loc[buys, "Taxes"].fillna(0).sum()
            )
            avg = (cost_leg + fees_leg) / tot_sh if tot_sh > 0 else np.nan
        else:
            avg = np.nan
        px = last_map.get(t, np.nan)
        mv = sh * px if not np.isnan(px) else np.nan
        inv = sh * avg if not np.isnan(avg) else np.nan
        pl = mv - inv if not (np.isnan(mv) or np.isnan(inv)) else np.nan
        pos.append([t, sh, avg, inv, px, mv, pl])
    dfp = pd.DataFrame(
        pos,
        columns=[
            "Ticker",
            "Shares",
            "AvgCost",
            "Invested",
            "MarketPrice",
            "MarketValue",
            "UnrealizedPL",
        ],
    )
    return dfp.sort_values("MarketValue", ascending=False)


def delete_transactions_by_ticker(ticker: str) -> int:
    """Delete all Transactions rows for a ticker."""
    ws = open_ws("Transactions")
    values = ws.get_all_values()
    if not values:
        return 0
    headers = values[0]
    try:
        tcol = headers.index("Ticker")
    except ValueError:
        st.error("No encontré la columna 'Ticker' en Transactions.")
        return 0
    to_delete = []
    for i, row in enumerate(values[1:], start=2):
        if len(row) > tcol and normalize_symbol(row[tcol]) == normalize_symbol(ticker):
            to_delete.append(i)
    for ridx in reversed(to_delete):
        ws.delete_rows(ridx)
    return len(to_delete)
