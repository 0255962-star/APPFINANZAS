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


def _clean_value(val):
    if isinstance(val, dict):
        return val.get("raw") if "raw" in val else val.get("fmt")
    return val


def _to_number(val):
    val = _clean_value(val)
    if val in (None, "", 0):
        return None if val in (None, "") else 0
    if isinstance(val, (int, float, np.number)):
        try:
            return float(val)
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip().upper().replace(",", "")
        multiplier = 1
        if s.endswith("T"):
            multiplier = 1_000_000_000_000
            s = s[:-1]
        elif s.endswith("B"):
            multiplier = 1_000_000_000
            s = s[:-1]
        elif s.endswith("M"):
            multiplier = 1_000_000
            s = s[:-1]
        elif s.endswith("K"):
            multiplier = 1_000
            s = s[:-1]
        try:
            return float(s) * multiplier
        except Exception:
            return None
    return None

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

    def _safe_data(attr_name, fallback_attrs=()):
        for attr in (attr_name, *fallback_attrs):
            try:
                data = getattr(t, attr, {}) or {}
                if callable(data):
                    data = data()
                if data:
                    return data
            except Exception:
                continue
        return {}

    price_data = _safe_data("price", ("info",))
    profile_data = _safe_data("summary_profile", ("profile", "info"))
    summary_data = _safe_data("summary_detail", ("info",))
    if not summary_data:
        summary_data = {}
    stats_data = _safe_data("key_stats", ("key_stats", "info"))

    info.update(
        {
            "longName": price_data.get("longName") or price_data.get("shortName"),
            "last_price": _to_number(price_data.get("regularMarketPrice")),
            "market_cap": _to_number(price_data.get("marketCap"))
            or _to_number(summary_data.get("marketCap")),
            "beta": _to_number(stats_data.get("beta"))
            or _to_number(price_data.get("beta")),
            "trailingPE": _to_number(summary_data.get("trailingPE")),
            "forwardPE": _to_number(summary_data.get("forwardPE")),
            "dividendYield": _to_number(summary_data.get("dividendYield")),
            "sector": profile_data.get("sector"),
            "industry": profile_data.get("industry"),
            "country": profile_data.get("country"),
            "website": profile_data.get("website"),
            "longBusinessSummary": profile_data.get("longBusinessSummary"),
        }
    )
    try:
        gi = t.get_info()
    except Exception:
        gi = {}
        try:
            gi = getattr(t, "info", {}) or {}
        except Exception:
            gi = {}
    if gi:
        def _fill(target, names, numeric=True):
            if info.get(target) not in (None, "", 0):
                return
            for name in names:
                val = gi.get(name)
                use_val = _to_number(val) if numeric else val
                if use_val not in (None, ""):
                    info[target] = use_val
                    return

        _fill("longName", ("longName", "shortName"), numeric=False)
        _fill("last_price", ("regularMarketPrice", "previousClose"))
        _fill("market_cap", ("marketCap",))
        _fill("beta", ("beta",))
        _fill("trailingPE", ("trailingPE",))
        _fill("forwardPE", ("forwardPE",))
        _fill("dividendYield", ("dividendYield", "trailingAnnualDividendYield"))
        _fill("sector", ("sector",), numeric=False)
        _fill("industry", ("industry",), numeric=False)
        _fill("country", ("country",), numeric=False)
        _fill("website", ("website", "websiteUrl"), numeric=False)
        _fill("longBusinessSummary", ("longBusinessSummary",), numeric=False)
    try:
        fast_obj = getattr(t, "fast_info", None)
    except Exception:
        fast_obj = None

    def _fast_fetch(names):
        if fast_obj is None:
            return None
        for name in names:
            try:
                if isinstance(fast_obj, dict) and name in fast_obj:
                    return fast_obj.get(name)
                val = getattr(fast_obj, name, None)
                if val is not None:
                    return val
            except Exception:
                continue
        return None

    # FIX: yfinance cambió los nombres en fast_info (snake_case vs camelCase).
    fast_map = {
        "last_price": ("last_price", "lastPrice"),
        "market_cap": ("market_cap", "marketCap"),
        "beta": ("beta",),
    }
    for target, candidates in fast_map.items():
        if target not in info:
            val = _fast_fetch(candidates)
            if val is not None:
                info[target] = val
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
    # FIX: permitir continuar incluso si alguna serie no está disponible.
    pxdf = ensure_prices([cand, chart_benchmark], start_for_chart, persist=True)
    if pxdf is None:
        pxdf = pd.DataFrame()
    cand_prices = pxdf.get(cand)
    bench_prices = pxdf.get(chart_benchmark)
    cand_has_prices = cand_prices is not None and not cand_prices.dropna().empty
    bench_has_prices = bench_prices is not None and not bench_prices.dropna().empty
    if not cand_has_prices:
        st.warning("No pude obtener suficientes datos históricos del candidato tras reintento.")
    if not bench_has_prices:
        st.warning(f"No pude obtener suficientes datos de {chart_benchmark} tras reintento.")

    info = get_company_info(cand)
    # FIX: usar precios descargados como respaldo cuando la metadata de Yahoo viene vacía.
    fallback_price = None
    if cand_has_prices:
        try:
            fallback_price = float(cand_prices.ffill().iloc[-1])
        except Exception:
            fallback_price = None
    if fallback_price is not None and not info.get("last_price"):
        info["last_price"] = fallback_price
    beta_calc = None
    beta_current = info.get("beta")
    needs_beta = (
        beta_current is None
        or (isinstance(beta_current, (float, int)) and np.isnan(beta_current))
        or beta_current == 0
    )
    if needs_beta and cand_has_prices and bench_has_prices:
        aligned = pxdf[[cand, chart_benchmark]].dropna()
        if aligned.shape[0] >= 5:
            rets = aligned.pct_change().dropna()
            cov = rets[[cand, chart_benchmark]].cov()
            var_b = cov.loc[chart_benchmark, chart_benchmark]
            if var_b and not np.isnan(var_b) and var_b != 0:
                beta_calc = cov.loc[cand, chart_benchmark] / var_b
    if beta_calc is not None and not np.isnan(beta_calc):
        info["beta"] = beta_calc

    def _fmt_value(val, fmt="{:,.2f}"):
        num = _to_number(val)
        if num is None:
            return "—"
        return fmt.format(num)

    def _fmt_cap(val):
        num = _to_number(val)
        if num is None:
            return "—"
        units = [
            (1_000_000_000_000, "T"),
            (1_000_000_000, "B"),
            (1_000_000, "M"),
        ]
        for divisor, suffix in units:
            if num >= divisor:
                scaled = num / divisor
                return f"{scaled:,.2f} {suffix}"
        return f"{num:,.0f}"

    name = info.get("longName") or cand
    st.subheader(f"{name} ({cand})")
    ks1, ks2, ks3, ks4, ks5, ks6 = st.columns(6)
    ks1.metric("Precio", _fmt_value(info.get("last_price")))
    market_cap_val = info.get("market_cap")
    ks2.metric("Cap. de mercado", _fmt_cap(market_cap_val))
    ks3.metric("Beta", _fmt_value(info.get("beta"), "{:,.2f}"))
    ks4.metric("PE (TTM)", _fmt_value(info.get("trailingPE"), "{:,.2f}"))
    ks5.metric("PE (fwd)", _fmt_value(info.get("forwardPE"), "{:,.2f}"))
    dy = info.get("dividendYield")
    ks6.metric("Div. Yield", f"{float(dy)*100:,.2f}%" if dy else "—")
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

    if cand_has_prices and bench_has_prices:
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
    else:
        st.info("No se pudo graficar la comparación normalizada por falta de datos completos.")

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
