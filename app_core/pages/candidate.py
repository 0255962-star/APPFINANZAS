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

from ..analysis_utils import build_metrics_comparison
from ..masters import build_masters, get_setting, masters_expired, need_build
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
    chart_benchmark = "SPY"

    if need_build("prices_master") or masters_expired():
        build_masters(sync=False)

    settings_df = st.session_state["settings_master"]
    prices_master = st.session_state["prices_master"]

    if start_date and prices_master is not None and not prices_master.empty:
        prices = prices_master.loc[pd.Timestamp(start_date) :]
    else:
        prices = prices_master.copy()

    # --- Nueva funcionalidad: entrada del candidato y simulador interactivo ---
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
    pxdf = ensure_prices([cand, chart_benchmark], start_for_chart, persist=True)

    if (
        pxdf is None
        or pxdf.empty
        or cand not in pxdf.columns
        or chart_benchmark not in pxdf.columns
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
            f"**Empresa:** {name}  \n"
            f"**Sector:** {info.get('sector','—')}  \n"
            f"**Industria:** {info.get('industry','—')}  \n"
            f"**País:** {info.get('country','—')}"
        )
        if info.get("website"):
            st.markdown(f"[Sitio web oficial]({info.get('website')})")
    with colA:
        desc = info.get("longBusinessSummary", "")
        if desc:
            st.write(desc[:800] + ("…" if len(desc) > 800 else ""))

    norm = pxdf[[cand, chart_benchmark]].ffill()
    for col in norm.columns:
        series = norm[col].dropna()
        if not series.empty:
            norm[col] = norm[col] / series.iloc[0]
    norm = norm.dropna(how="all")
    if norm.empty:
        st.info("No hay datos suficientes para graficar la comparación normalizada.")
    else:
        st.plotly_chart(
            px.line(
                norm,
                title=f"{cand} vs {chart_benchmark} (normalizado)",
                labels={"value": "Índice (base = 1)", "index": "Fecha"},
            ),
            use_container_width=True,
        )

    with st.expander("📈 ¿Cómo cambiaría el portafolio si agrego este activo?"):
        st.caption("Simulación hipotética; no modifica tus transacciones reales.")
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

            sim_start = prices.index[0].strftime("%Y-%m-%d")
            cand_hist = ensure_prices([cand], sim_start, persist=True)
            if cand_hist is None or cand_hist.empty or cand not in cand_hist.columns:
                st.info("No pude obtener precios históricos suficientes del candidato para simular.")
            else:
                cand_px = cand_hist[cand].reindex(prices.index).ffill()
                cand_ret = cand_px.pct_change().dropna()
                combo = pd.concat(
                    [
                        port_ret_now.rename("port"),
                        cand_ret.rename("cand"),
                    ],
                    axis=1,
                ).dropna()
                if combo.empty:
                    st.info("No hubo traslape suficiente entre el portafolio y el ticker para simular.")
                else:
                    port_base = combo["port"]
                    cand_aligned = combo["cand"]
                    w_new = weight_new
                    port_ret_new = port_base * (1 - w_new) + cand_aligned * w_new

                    rf = get_setting(settings_df, "RF", 0.03, float)
                    comp = build_metrics_comparison(port_base, port_ret_new, rf)
                    st.dataframe(comp.table, use_container_width=True)

    # --- Fin nueva funcionalidad: simulador de incorporación ---
