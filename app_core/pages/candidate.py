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
from ..sheets_client import open_ws
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


def _clean_text(val):
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in {"none", "n/a", "nan", "null"}:
            return None
        return s
    return str(val)

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

    info = {}

    def _safe_fast(key):
        try:
            fast_obj = getattr(t, "fast_info", None)
            if fast_obj is None:
                return None
            if isinstance(fast_obj, dict):
                return fast_obj.get(key)
            return getattr(fast_obj, key, None)
        except Exception:
            return None

    info["last_price"] = _to_number(_safe_fast("last_price") or _safe_fast("lastPrice"))
    info["market_cap"] = _to_number(_safe_fast("market_cap") or _safe_fast("marketCap"))
    info["beta"] = _to_number(_safe_fast("beta"))

    base = {}
    for attr in ("get_info", "info"):
        try:
            obj = getattr(t, attr)
            data = obj() if callable(obj) else obj
            if data:
                base = data
                break
        except Exception:
            continue

    def _fill_numeric(key, *candidates):
        if info.get(key) is not None:
            return
        for name in candidates:
            val = _to_number(base.get(name))
            if val is not None:
                info[key] = val
                return

    def _fill_text(key, *candidates):
        if info.get(key):
            return
        for name in candidates:
            val = _clean_text(base.get(name))
            if val:
                info[key] = val
                return

    _fill_text("longName", "longName", "shortName")
    _fill_numeric("last_price", "regularMarketPrice", "currentPrice", "previousClose")
    _fill_numeric("market_cap", "marketCap")
    _fill_numeric("beta", "beta")
    _fill_numeric("trailingPE", "trailingPE")
    _fill_numeric("forwardPE", "forwardPE")
    _fill_numeric("dividendYield", "dividendYield", "trailingAnnualDividendYield")
    _fill_text("sector", "sector")
    _fill_text("industry", "industry")
    _fill_text("country", "country")
    _fill_text("website", "website", "websiteUrl")
    _fill_text("longBusinessSummary", "longBusinessSummary")

    return info


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
        shares_to_simulate = st.number_input(
            "Acciones a simular",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Cantidad hipotética de acciones a evaluar en el portafolio.",
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

    def _fmt_text(val):
        cleaned = _clean_text(val)
        return cleaned if cleaned else "—"

    name = info.get("longName") or cand
    st.subheader(f"{name} ({cand})")

    price_now = info.get("last_price") or fallback_price
    investment_sim = (shares_to_simulate or 0) * (price_now or 0)

    # Fundamental cards
    cards = [
        ("Precio actual", price_now, _fmt_value),
        ("Cap. de mercado", info.get("market_cap"), _fmt_cap),
        ("Beta", info.get("beta"), lambda v: _fmt_value(v, "{:,.2f}")),
        ("PE (TTM)", info.get("trailingPE"), lambda v: _fmt_value(v, "{:,.2f}")),
        ("PE (forward)", info.get("forwardPE"), lambda v: _fmt_value(v, "{:,.2f}")),
        ("Dividend Yield", info.get("dividendYield"), lambda v: f"{float(v)*100:,.2f}%"),
    ]
    visible_cards = [c for c in cards if _to_number(c[1]) is not None]
    if visible_cards:
        cols = st.columns(len(visible_cards))
        for col, (label, value, fmtfn) in zip(cols, visible_cards):
            col.metric(label, fmtfn(value))

    details_cols = st.columns([2, 1])
    with details_cols[0]:
        desc = info.get("longBusinessSummary", "")
        if desc:
            st.markdown("#### Descripción breve")
            st.write(desc[:800] + ("…" if len(desc) > 800 else ""))
        fundamentals = []
        for lbl, key in (
            ("Payout Ratio", "payoutRatio"),
            ("Revenue TTM", "revenue"),
            ("Revenue Growth", "revenueGrowth"),
            ("EPS TTM", "trailingEps"),
            ("EPS Growth", "earningsGrowth"),
            ("Gross Margin", "grossMargins"),
            ("Operating Margin", "operatingMargins"),
            ("Profit Margin", "profitMargins"),
            ("Total Cash", "totalCash"),
            ("Total Debt", "totalDebt"),
            ("Quick Ratio", "quickRatio"),
            ("Current Ratio", "currentRatio"),
        ):
            val = info.get(key)
            if val is None:
                continue
            if "Margin" in lbl or "Growth" in lbl or "Yield" in lbl or "Ratio" in lbl:
                fundamentals.append(f"**{lbl}:** {_fmt_value(val, '{:,.2%}')}")
            elif "Cash" in lbl or "Debt" in lbl or "Revenue" in lbl:
                fundamentals.append(f"**{lbl}:** {_fmt_cap(val)}")
            else:
                fundamentals.append(f"**{lbl}:** {_fmt_value(val)}")
        if fundamentals:
            st.markdown("#### Indicadores clave")
            st.markdown("  \n".join(fundamentals))
    with details_cols[1]:
        st.markdown(
            f"**Sector:** {_fmt_text(info.get('sector'))}  \n"
            f"**Industria:** {_fmt_text(info.get('industry'))}  \n"
            f"**País:** {_fmt_text(info.get('country'))}"
        )
        if info.get("website"):
            st.markdown(f"[Sitio oficial]({info.get('website')})")

    # Gráfico Candidate vs SPY
    st.markdown("### 📊 Precio normalizado")
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

    # Simulación de portafolio
    tx_df = st.session_state["tx_master"]
    prices_all = st.session_state["prices_master"]
    if start_date and prices_all is not None and not prices_all.empty:
        prices = prices_all.loc[pd.Timestamp(start_date) :]
    else:
        prices = prices_all.copy()

    last_map = st.session_state.get("last_map_master", {})
    pos_df = positions_from_tx(tx_df, last_hint_map=last_map)
    tot_mv = pos_df["MarketValue"].fillna(0).sum()

    st.markdown("### 🧪 Impacto en el portafolio")
    if (
        prices is None
        or prices.empty
        or tot_mv <= 0
        or prices.shape[0] < 2
        or price_now is None
    ):
        st.info("Necesito precios del portafolio, del candidato y un valor razonable del portafolio para simular.")
        port_ret_now = pd.Series(dtype=float)
        port_ret_new = pd.Series(dtype=float)
        bench_series = pd.Series(dtype=float)
        cand_ret = pd.Series(dtype=float)
    else:
        rets = prices.pct_change().dropna(how="all")
        weight_series = (
            (pos_df.set_index("Ticker")["MarketValue"] / tot_mv)
            if tot_mv > 0
            else pd.Series(dtype=float)
        )
        wv = weight_series.reindex(rets.columns).fillna(0.0).values
        port_ret_now = (rets * wv).sum(axis=1).dropna()

        sim_start = prices.index[0].strftime("%Y-%m-%d")
        cand_hist = ensure_prices([cand], sim_start, persist=True)
        cand_ret = pd.Series(dtype=float)
        if cand_hist is not None and not cand_hist.empty and cand in cand_hist.columns:
            cand_px = cand_hist[cand].reindex(prices.index).ffill()
            cand_ret = cand_px.pct_change().reindex(port_ret_now.index).fillna(0.0)

        total_after = tot_mv + investment_sim
        if total_after > 0 and investment_sim > 0 and not cand_ret.empty:
            current_scale = tot_mv / total_after
            cand_weight = investment_sim / total_after
        else:
            current_scale = 1.0
            cand_weight = 0.0
        port_ret_new = port_ret_now * current_scale + cand_ret * cand_weight

        bench_series = pd.Series(dtype=float)
        if bench_has_prices:
            bench_px = bench_prices.reindex(prices.index).ffill()
            bench_series = bench_px.pct_change().dropna()

        st.caption(
            f"Acciones simuladas: **{shares_to_simulate:,.2f}** | Inversión: **${investment_sim:,.2f}** | Peso estimado: **{(cand_weight*100):,.2f}%**"
        )

        curve = pd.DataFrame(
            {
                "Actual": (1 + port_ret_now).cumprod(),
                "Con candidato": (1 + port_ret_new).cumprod(),
            }
        ).dropna(how="all")
        if not curve.empty:
            st.plotly_chart(
                px.line(curve, title="Evolución del portafolio (base=1.0)"),
                use_container_width=True,
            )

    def _metrics_summary(ret_series, bench_series, rf):
        result = {
            "ret": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "mdd": np.nan,
            "calmar": np.nan,
            "beta": np.nan,
            "tracking_error": np.nan,
        }
        if ret_series is None or ret_series.empty:
            return result
        result["ret"] = annualize_return(ret_series)
        result["vol"] = annualize_vol(ret_series)
        result["sharpe"] = sharpe(ret_series, rf)
        result["sortino"] = sortino(ret_series, rf)
        result["mdd"] = max_drawdown((1 + ret_series).cumprod())
        result["calmar"] = calmar(ret_series)
        if bench_series is not None and not bench_series.empty:
            aligned = pd.concat([ret_series, bench_series], axis=1).dropna()
            if not aligned.empty:
                cov = aligned.cov().iloc[0, 1]
                var_b = aligned.iloc[:, 1].var()
                if var_b and var_b > 0:
                    result["beta"] = cov / var_b
                diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                result["tracking_error"] = diff.std(ddof=0) * np.sqrt(252)
        return result

    rf = get_setting(st.session_state["settings_master"], "RF", 0.03, float)
    bench_for_base = bench_series.reindex(port_ret_now.index).dropna()
    base_for_metrics = port_ret_now.reindex(bench_for_base.index).dropna() if not bench_for_base.empty else port_ret_now
    bench_for_new = bench_series.reindex(port_ret_new.index).dropna()
    new_for_metrics = port_ret_new.reindex(bench_for_new.index).dropna() if not bench_for_new.empty else port_ret_new

    metrics_current = _metrics_summary(base_for_metrics, bench_for_base, rf)
    metrics_new = _metrics_summary(new_for_metrics, bench_for_new, rf)
    corr_current = np.nan
    corr_new = np.nan
    if not cand_ret.empty and not port_ret_now.empty:
        aligned_corr = pd.concat([port_ret_now, cand_ret], axis=1).dropna()
        if not aligned_corr.empty:
            corr_current = aligned_corr.corr().iloc[0, 1]
        aligned_corr_new = pd.concat([port_ret_new, cand_ret], axis=1).dropna()
        if not aligned_corr_new.empty:
            corr_new = aligned_corr_new.corr().iloc[0, 1]

    rows = []
    metric_rows = [
        ("Rendimiento anualizado", "ret", True),
        ("Volatilidad anualizada", "vol", True),
        ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False),
        ("Max Drawdown", "mdd", True),
        ("Calmar", "calmar", False),
        ("Beta vs SPY", "beta", False),
        ("Tracking Error", "tracking_error", True),
    ]
    for label, key, is_pct in metric_rows:
        cur = metrics_current.get(key)
        new = metrics_new.get(key)
        if cur is None and new is None:
            continue
        fmt = "{:,.2f}%" if is_pct else "{:,.2f}"
        rows.append(
            {
                "Métrica": label,
                "Actual": "—" if cur is None or np.isnan(cur) else fmt.format(cur * 100 if is_pct else cur),
                "Con candidato": "—" if new is None or np.isnan(new) else fmt.format(new * 100 if is_pct else new),
                "Δ": "—"
                if cur is None or new is None or np.isnan(cur) or np.isnan(new)
                else fmt.format((new - cur) * 100 if is_pct else (new - cur)),
            }
        )
    rows.append(
        {
            "Métrica": "Correlación candidato-portafolio",
            "Actual": "—" if np.isnan(corr_current) else f"{corr_current:,.2f}",
            "Con candidato": "—" if np.isnan(corr_new) else f"{corr_new:,.2f}",
            "Δ": "—"
            if np.isnan(corr_current) or np.isnan(corr_new)
            else f"{(corr_new - corr_current):,.2f}",
        }
    )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No pude calcular métricas del portafolio para este escenario.")

    # Expander para registrar en Sheets
    with st.expander("🧾 Registrar esta acción en mi portafolio"):
        st.caption("La siguiente operación se escribirá en la hoja *Transactions*.")
        with st.form("register_candidate_trade"):
            today = datetime.utcnow().date()
            trade_date = st.date_input("Fecha de compra", value=today)
            shares_real = st.number_input(
                "Acciones a registrar",
                min_value=0.0,
                value=float(shares_to_simulate) if shares_to_simulate > 0 else 0.0,
                step=1.0,
            )
            price_entry = st.number_input(
                "Precio de compra",
                min_value=0.0,
                value=float(price_now or 0),
                step=0.01,
            )
            note = st.text_input("Nota (opcional)")
            submitted = st.form_submit_button("Agregar al portafolio")
            if submitted:
                if shares_real <= 0:
                    st.error("Necesito una cantidad de acciones mayor que cero.")
                elif price_entry <= 0:
                    st.error("El precio debe ser mayor que cero.")
                else:
                    try:
                        ws = open_ws("Transactions")
                        headers = st.session_state["tx_master"].columns.tolist()
                        if not headers:
                            headers = ws.row_values(1)
                        row = []
                        for col in headers:
                            col_norm = col.strip().lower()
                            if col_norm == "tradedate":
                                row.append(trade_date.isoformat())
                            elif col_norm == "ticker":
                                row.append(cand)
                            elif col_norm == "side":
                                row.append("BUY")
                            elif col_norm == "shares":
                                row.append(shares_real)
                            elif col_norm == "price":
                                row.append(price_entry)
                            elif col_norm in {"notes", "nota"}:
                                row.append(note)
                            elif col_norm in {"sector"}:
                                row.append(_fmt_text(info.get("sector")) or "")
                            elif col_norm in {"industry"}:
                                row.append(_fmt_text(info.get("industry")) or "")
                            elif col_norm in {"country"}:
                                row.append(_fmt_text(info.get("country")) or "")
                            else:
                                row.append("")
                        ws.append_row(row, value_input_option="USER_ENTERED")
                    except Exception as exc:
                        st.error(f"No pude registrar la compra: {exc}")
                    else:
                        st.success("Se registró la operación. Refresca el portafolio para verla reflejada.")
