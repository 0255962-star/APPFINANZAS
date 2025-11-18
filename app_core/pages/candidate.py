"""Candidate evaluation page renderer: render_candidate_page(window)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from ..masters import build_masters, get_setting, masters_expired, need_build
from ..metrics import (
    annualize_return,
    annualize_vol,
    calmar,
    max_drawdown,
    sharpe,
    sortino,
)
from ..prices_fetch import ensure_prices, normalize_symbol
from ..tx_positions import positions_from_tx


def _start_date_for_window(window: str) -> Optional[str]:
    period_map = {"6M": 180, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5}
    if window == "Max":
        return None
    days = period_map.get(window)
    if not days:
        return None
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")


@st.cache_data(ttl=1800, show_spinner=False)
def get_company_info(ticker: str):
    """Return cached metadata for a ticker."""
    t = yf.Ticker(ticker)
    info = {}
    try:
        fi = getattr(t, "fast_info", {}) or {}
        info.update({k: fi.get(k) for k in ["last_price", "market_cap", "beta"] if k in fi})
    except Exception:
        pass
    try:
        gi = t.get_info()
        if gi:
            info.update(
                {
                    "longName": gi.get("longName") or gi.get("shortName"),
                    "sector": gi.get("sector"),
                    "industry": gi.get("industry"),
                    "website": gi.get("website"),
                    "country": gi.get("country"),
                    "longBusinessSummary": gi.get("longBusinessSummary"),
                    "forwardPE": gi.get("forwardPE"),
                    "trailingPE": gi.get("trailingPE"),
                    "dividendYield": gi.get("dividendYield"),
                }
            )
    except Exception:
        pass
    return info or {}


def render_candidate_page(window: str) -> None:
    """Render the Evaluar Candidato tab."""
    start_date = _start_date_for_window(window)
    st.title("🔎 Evaluar Candidato")

    if need_build("prices_master") or masters_expired():
        build_masters(sync=False)

    settings_df = st.session_state["settings_master"]
    prices_master = st.session_state["prices_master"]
    bench_ret_full = st.session_state["bench_ret_master"]
    benchmark = get_setting(settings_df, "Benchmark", "SPY", str)

    if start_date and prices_master is not None and not prices_master.empty:
        prices = prices_master.loc[pd.Timestamp(start_date) :]
    else:
        prices = prices_master.copy()

    bench_ret = pd.Series(dtype=float)
    if bench_ret_full is not None and not bench_ret_full.empty:
        if prices is not None and not prices.empty:
            bench_ret = bench_ret_full.reindex(prices.index).ffill()
        else:
            bench_ret = bench_ret_full

    c1, c2 = st.columns([2, 1])
    with c1:
        user_query = st.text_input(
            "Ticker del candidato (ej. NVDA, KO, COST, BRK.B):",
            value="",
            placeholder="Escribe un ticker",
        )
    with c2:
        weight_new = st.slider(
            "Peso a simular",
            min_value=0.0,
            max_value=0.3,
            value=0.05,
            step=0.01,
            help="Peso del candidato en el portafolio hipotético",
        )

    if not user_query:
        st.info("Ingresa un **ticker** para evaluar el candidato.")
        st.stop()

    cand = normalize_symbol(user_query)
    if not re.match(r"^[A-Z0-9\.\-]+$", cand):
        st.error("Ticker inválido.")
        st.stop()

    start_for_chart = start_date or (
        datetime.utcnow() - timedelta(days=365 * 3)
    ).strftime("%Y-%m-%d")
    pxdf = ensure_prices([cand, benchmark], start_for_chart, persist=True)

    if (
        pxdf is None
        or pxdf.empty
        or cand not in pxdf.columns
        or benchmark not in pxdf.columns
    ):
        st.warning("No pude obtener suficientes datos para el candidato / SPY tras reintento.")
        st.stop()

    info = get_company_info(cand)
    name = info.get("longName") or cand
    st.subheader(f"{name} ({cand})")
    ks1, ks2, ks3, ks4, ks5, ks6 = st.columns(6)
    ks1.metric("Precio", f"{info.get('last_price', np.nan):,.2f}" if info.get("last_price") else "—")
    ks2.metric("Cap. de mercado", f"{(info.get('market_cap') or 0):,}")
    ks3.metric("Beta", f"{info.get('beta') or '—'}")
    ks4.metric("PE (TTM)", f"{info.get('trailingPE') or '—'}")
    ks5.metric("PE (fwd)", f"{info.get('forwardPE') or '—'}")
    dy = info.get("dividendYield")
    ks6.metric("Div. Yield", f"{(dy*100):.2f}%" if dy else "—")
    colA, colB = st.columns([2, 1])
    with colB:
        st.markdown(
            f"**Sector:** {info.get('sector','—')}  \n"
            f"**Industria:** {info.get('industry','—')}  \n"
            f"**País:** {info.get('country','—')}"
        )
        if info.get("website"):
            st.markdown(f"[Sitio web]({info.get('website')})")
    with colA:
        desc = info.get("longBusinessSummary", "")
        if desc:
            st.write(desc[:800] + ("…" if len(desc) > 800 else ""))

    norm = pxdf.ffill() / pxdf.ffill().iloc[0]
    st.plotly_chart(
        px.line(norm[[cand, benchmark]], title=f"Comportamiento normalizado: {cand} vs {benchmark}"),
        use_container_width=True,
    )

    with st.expander("📈 ¿Cómo cambiaría el portafolio si agrego este activo?"):
        tx_df = st.session_state["tx_master"]
        prices_all = st.session_state["prices_master"]
        if start_date and prices_all is not None and not prices_all.empty:
            prices = prices_all.loc[pd.Timestamp(start_date) :]
        else:
            prices = prices_all.copy()

        last_map = st.session_state.get("last_map_master", {})
        pos_df = positions_from_tx(tx_df, last_hint_map=last_map)
        tot_mv = pos_df["MarketValue"].fillna(0).sum()
        w_now = (
            (pos_df.set_index("Ticker")["MarketValue"] / tot_mv).dropna()
            if tot_mv > 0
            else pd.Series(dtype=float)
        )

        if prices is None or prices.empty or len(w_now) == 0 or prices.shape[0] < 2:
            st.info("No hay suficientes datos para simular.")
        else:
            rets = prices.pct_change().dropna(how="all")
            wv = w_now.reindex(rets.columns).fillna(0).values
            port_ret_now = (rets * wv).sum(axis=1)

            cand_ret = (
                pxdf[[cand]].pct_change().dropna().reindex(port_ret_now.index).ffill()[cand]
            )
            w_new = weight_new
            port_ret_new = port_ret_now * (1 - w_new) + cand_ret * w_new

            rf = get_setting(settings_df, "RF", 0.03, float)

            def mset(s):
                return (
                    annualize_return(s),
                    annualize_vol(s),
                    sharpe(s, rf),
                    max_drawdown((1 + s).cumprod()),
                    sortino(s, rf),
                    calmar(s),
                )

            r0, v0, sh0, dd0, so0, ca0 = mset(port_ret_now)
            r1, v1, sh1, dd1, so1, ca1 = mset(port_ret_new)
            dfm = pd.DataFrame(
                {
                    "Métrica": [
                        "Rend. anualizado",
                        "Vol. anualizada",
                        "Sharpe",
                        "Max Drawdown",
                        "Sortino",
                        "Calmar",
                    ],
                    "Actual": [
                        f"{(r0 or 0)*100:,.2f}%",
                        f"{(v0 or 0)*100:,.2f}%",
                        f"{(sh0 or 0):.2f}",
                        f"{(dd0 or 0)*100:,.2f}%",
                        f"{(so0 or 0):.2f}",
                        f"{(ca0 or 0):.2f}",
                    ],
                    "Con candidato": [
                        f"{(r1 or 0)*100:,.2f}%",
                        f"{(v1 or 0)*100:,.2f}%",
                        f"{(sh1 or 0):.2f}",
                        f"{(dd1 or 0)*100:,.2f}%",
                        f"{(so1 or 0):.2f}",
                        f"{(ca1 or 0):.2f}",
                    ],
                    "Δ": [
                        f"{((r1 or 0)-(r0 or 0))*100:,.2f} pp",
                        f"{((v1 or 0)-(v0 or 0))*100:,.2f} pp",
                        f"{((sh1 or 0)-(sh0 or 0)):.2f}",
                        f"{((dd1 or 0)-(dd0 or 0))*100:,.2f} pp",
                        f"{((so1 or 0)-(so0 or 0)):.2f}",
                        f"{((ca1 or 0)-(ca0 or 0)):.2f}",
                    ],
                }
            )
            st.dataframe(dfm, use_container_width=True)
