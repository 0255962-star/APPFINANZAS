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
    _fill_numeric("payoutRatio", "payoutRatio")
    _fill_numeric("revenue", "totalRevenue", "revenue")
    _fill_numeric("revenueGrowth", "revenueGrowth")
    _fill_numeric("trailingEps", "trailingEps")
    _fill_numeric("earningsGrowth", "earningsGrowth")
    _fill_numeric("grossMargins", "grossMargins")
    _fill_numeric("operatingMargins", "operatingMargins")
    _fill_numeric("profitMargins", "profitMargins")
    _fill_numeric("totalCash", "totalCash")
    _fill_numeric("totalDebt", "totalDebt")
    _fill_numeric("quickRatio", "quickRatio")
    _fill_numeric("currentRatio", "currentRatio")
    _fill_text("sector", "sector")
    _fill_text("industry", "industry")
    _fill_text("country", "country")
    _fill_text("website", "website", "websiteUrl")
    _fill_text("longBusinessSummary", "longBusinessSummary")

    return info


def _fmt_value(val, fmt="{:,.2f}"):
    num = _to_number(val)
    if num is None:
        return "—"
    return fmt.format(num)


def _fmt_cap(val):
    num = _to_number(val)
    if num is None:
        return "—"
    for divisor, suffix in (
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
    ):
        if num >= divisor:
            return f"{num / divisor:,.2f} {suffix}"
    return f"{num:,.0f}"


def _fmt_text(val):
    cleaned = _clean_text(val)
    return cleaned if cleaned else "—"


def _fmt_percent(val):
    num = _to_number(val)
    if num is None:
        return "—"
    return f"{num * 100:,.2f}%"


def render_candidate_page(window: str) -> None:
    """Render the Evaluar Candidato tab."""
    start_date = _start_date_for_window(window)
    st.title("🔎 Evaluar Candidato")
    chart_benchmark = "SPY"

    if need_build("prices_master") or masters_expired():
        build_masters(sync=False)

    settings_df = st.session_state["settings_master"]
    prices_master = st.session_state["prices_master"]
    tx_df = st.session_state["tx_master"]

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
            help="Cantidad hipotética de acciones para evaluar.",
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
    fallback_price = None
    if cand_has_prices:
        try:
            fallback_price = float(cand_prices.ffill().iloc[-1])
        except Exception:
            fallback_price = None
    if fallback_price is not None and not info.get("last_price"):
        info["last_price"] = fallback_price

    price_now = info.get("last_price") or fallback_price
    investment_sim = (shares_to_simulate or 0) * (price_now or 0)

    # Portafolio actual para calcular pesos y simulaciones
    tx_df = st.session_state["tx_master"]
    prices_all = st.session_state["prices_master"]
    if start_date and prices_all is not None and not prices_all.empty:
        port_prices = prices_all.loc[pd.Timestamp(start_date) :]
    else:
        port_prices = prices_all.copy()

    last_map = st.session_state.get("last_map_master", {})
    pos_df = positions_from_tx(tx_df, last_hint_map=last_map)
    tot_mv = pos_df["MarketValue"].fillna(0).sum()

    name = info.get("longName") or cand
    st.subheader(f"{name} ({cand})")
    c_price, c_beta = st.columns(2)
    c_price.metric("Precio actual", _fmt_value(price_now))
    beta_display = _fmt_value(info.get("beta"), "{:,.2f}")
    if beta_display == "—":
        beta_display = "N/D"
    inv_text = f"${investment_sim:,.2f}"
    if tot_mv > 0:
        pct_text = f"{(investment_sim / tot_mv)*100:,.2f}% del portafolio"
    else:
        pct_text = "No se puede estimar"
    with c_beta:
        st.markdown("**Beta y peso simulado**")
        st.write(f"Beta: {beta_display} | {inv_text} ({pct_text})")

    fundamentals_data = []
    for lbl, key, fmt_fn in (
        ("Cap. de mercado", "market_cap", _fmt_cap),
        ("PE (TTM)", "trailingPE", lambda v: _fmt_value(v, "{:,.2f}")),
        ("PE (forward)", "forwardPE", lambda v: _fmt_value(v, "{:,.2f}")),
        ("Dividend Yield", "dividendYield", _fmt_percent),
        ("Payout Ratio", "payoutRatio", _fmt_percent),
        ("Revenue TTM", "revenue", _fmt_cap),
        ("Revenue Growth", "revenueGrowth", _fmt_percent),
        ("EPS TTM", "trailingEps", lambda v: _fmt_value(v, "{:,.2f}")),
        ("EPS Growth", "earningsGrowth", _fmt_percent),
        ("Gross Margin", "grossMargins", _fmt_percent),
        ("Operating Margin", "operatingMargins", _fmt_percent),
        ("Profit Margin", "profitMargins", _fmt_percent),
        ("Total Cash", "totalCash", _fmt_cap),
        ("Total Debt", "totalDebt", _fmt_cap),
        ("Quick Ratio", "quickRatio", lambda v: _fmt_value(v, "{:,.2f}")),
        ("Current Ratio", "currentRatio", lambda v: _fmt_value(v, "{:,.2f}")),
    ):
        val = fmt_fn(info.get(key)) if info.get(key) is not None else None
        if val and val != "—":
            fundamentals_data.append((lbl, val))

    description_text = info.get("longBusinessSummary", "")
    geo_lines = []
    for label, key in (("Sector", "sector"), ("Industria", "industry"), ("País", "country")):
        val = _clean_text(info.get(key))
        if val:
            geo_lines.append(f"**{label}:** {val}")
    if info.get("website"):
        geo_lines.append(f"[Sitio oficial]({info.get('website')})")

    port_ret_now = pd.Series(dtype=float)
    port_ret_new = pd.Series(dtype=float)
    cand_ret = pd.Series(dtype=float)
    bench_series = pd.Series(dtype=float)
    curve = pd.DataFrame()
    cand_weight = 0.0

    if (
        port_prices is not None
        and not port_prices.empty
        and tot_mv > 0
        and port_prices.shape[0] >= 2
        and price_now is not None
    ):
        rets = port_prices.pct_change().dropna(how="all")
        weights = (pos_df.set_index("Ticker")["MarketValue"] / tot_mv) if tot_mv > 0 else pd.Series(dtype=float)
        wv = weights.reindex(rets.columns).fillna(0.0).values
        port_ret_now = (rets * wv).sum(axis=1).dropna()

        sim_start = port_prices.index[0].strftime("%Y-%m-%d")
        cand_hist = ensure_prices([cand], sim_start, persist=True)
        if cand_hist is not None and not cand_hist.empty and cand in cand_hist.columns:
            cand_px = cand_hist[cand].reindex(port_prices.index).ffill()
            cand_ret = cand_px.pct_change().reindex(port_ret_now.index).fillna(0.0)

        total_after = tot_mv + investment_sim
        if total_after > 0 and investment_sim > 0 and not cand_ret.empty:
            cand_weight = investment_sim / total_after
            current_scale = tot_mv / total_after
        else:
            cand_weight = 0.0
            current_scale = 1.0
        port_ret_new = port_ret_now * current_scale + cand_ret * cand_weight

        if bench_has_prices:
            bench_px = bench_prices.reindex(port_prices.index).ffill()
            bench_series = bench_px.pct_change().dropna()

        curve = pd.DataFrame(
            {
                "Actual": (1 + port_ret_now).cumprod(),
                "Con candidato": (1 + port_ret_new).cumprod(),
            }
        ).dropna(how="all")

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

    st.markdown("### 📈 Portafolio actual vs simulado")
    st.caption(
        f"Acciones simuladas: **{shares_to_simulate:,.2f}** | Inversión: **${investment_sim:,.2f}** | Peso estimado: **{(cand_weight*100):,.2f}%**"
    )
    if not curve.empty:
        st.plotly_chart(
            px.line(curve, title="Evolución del portafolio (base = 1.0)"),
            use_container_width=True,
        )
    else:
        st.info("No hay suficiente histórico para comparar el portafolio con y sin el candidato.")

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

    rf = get_setting(settings_df, "RF", 0.03, float)
    bench_for_base = bench_series.reindex(port_ret_now.index).dropna()
    base_for_metrics = (
        port_ret_now.reindex(bench_for_base.index).dropna()
        if not bench_for_base.empty
        else port_ret_now
    )
    bench_for_new = bench_series.reindex(port_ret_new.index).dropna()
    new_for_metrics = (
        port_ret_new.reindex(bench_for_new.index).dropna()
        if not bench_for_new.empty
        else port_ret_new
    )

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

    st.markdown("### 📋 Métricas del portafolio")
    metric_rows = [
        ("Rendimiento anualizado", "ret", True, True),
        ("Volatilidad anualizada", "vol", True, False),
        ("Sharpe", "sharpe", False, True),
        ("Sortino", "sortino", False, True),
        ("Max Drawdown", "mdd", True, False),
        ("Calmar", "calmar", False, True),
        ("Beta vs SPY", "beta", False, None),
        ("Tracking Error", "tracking_error", True, False),
    ]

    metrics_data = []
    for label, key, is_pct, better_high in metric_rows:
        cur = metrics_current.get(key)
        new = metrics_new.get(key)
        delta = (
            new - cur
            if cur is not None
            and not np.isnan(cur)
            and new is not None
            and not np.isnan(new)
            else np.nan
        )
        metrics_data.append(
            {
                "Métrica": label,
                "ActualValue": cur,
                "NewValue": new,
                "Delta": delta,
                "is_pct": is_pct,
                "better_high": better_high,
            }
        )

    metrics_data.append(
        {
            "Métrica": "Correlación candidato-portafolio",
            "ActualValue": corr_current,
            "NewValue": corr_new,
            "Delta": (
                corr_new - corr_current
                if not np.isnan(corr_current) and not np.isnan(corr_new)
                else np.nan
            ),
            "is_pct": False,
            "better_high": None,
        }
    )

    df_metrics = pd.DataFrame(metrics_data)

    def _format_metric(val, is_pct):
        if val is None or np.isnan(val):
            return "—"
        return f"{val * 100:,.2f}%" if is_pct else f"{val:,.2f}"

    if not df_metrics.empty:
        display_df = pd.DataFrame(
            {
                "Métrica": df_metrics["Métrica"],
                "Actual": [
                    _format_metric(v, pct)
                    for v, pct in zip(df_metrics["ActualValue"], df_metrics["is_pct"])
                ],
                "Con candidato": [
                    _format_metric(v, pct)
                    for v, pct in zip(df_metrics["NewValue"], df_metrics["is_pct"])
                ],
                "Δ": [
                    _format_metric(v, pct)
                    for v, pct in zip(df_metrics["Delta"], df_metrics["is_pct"])
                ],
            }
        )

        def _delta_style(row):
            idx = row.name
            better = df_metrics.loc[idx, "better_high"]
            delta_val = df_metrics.loc[idx, "Delta"]
            if better is None or delta_val is None or np.isnan(delta_val):
                return [""]
            improved = delta_val >= 0 if better else delta_val <= 0
            color = "#22c55e" if improved else "#ef4444"
            return [f"color: {color}; font-weight:bold;"]

        styled = display_df.style.apply(_delta_style, subset=["Δ"], axis=1)
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("No pude calcular métricas del portafolio para este escenario.")

    if description_text or fundamentals_data or geo_lines:
        st.markdown("### 🧾 Resumen fundamental")
        if description_text:
            st.write(description_text[:800] + ("…" if len(description_text) > 800 else ""))
        if geo_lines:
            st.markdown("  \n".join(geo_lines))
        if fundamentals_data:
            cols = st.columns(min(3, len(fundamentals_data)))
            for idx, (label, value) in enumerate(fundamentals_data):
                with cols[idx % len(cols)]:
                    st.markdown(f"**{label}:** {value}")

    with st.expander("🧾 Registrar esta acción en mi portafolio"):
        st.caption("Esta operación se agregará a la hoja *Transactions* respetando su estructura.")
        ws = open_ws("Transactions")
        values = ws.get_all_values() or []
        default_headers = [
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
        ]
        headers = values[0] if values else default_headers

        def _norm_header(h: str) -> str:
            return re.sub(r"[^a-z0-9]", "", h.lower())

        header_norm = [_norm_header(h) for h in headers]
        col_index = {name: idx for idx, name in enumerate(header_norm)}

        def sample_value(col_name: str, default=""):
            idx = col_index.get(_norm_header(col_name))
            if idx is None:
                return default
            for row in values[1:]:
                if len(row) > idx and row[idx]:
                    return row[idx]
            return default

        trade_idx = col_index.get("tradeid")
        trade_ids = []
        if trade_idx is not None:
            for row in values[1:]:
                if len(row) > trade_idx:
                    try:
                        trade_ids.append(int(float(row[trade_idx])))
                    except Exception:
                        continue
        next_trade_id = (max(trade_ids) if trade_ids else 0) + 1

        lot_idx = col_index.get("lotid")
        lot_numbers = []
        if lot_idx is not None:
            for row in values[1:]:
                if len(row) > lot_idx:
                    match = re.findall(r"\d+", row[lot_idx])
                    if match:
                        lot_numbers.append(int(match[-1]))
        next_lot_id = (max(lot_numbers) if lot_numbers else 0) + 1

        account_default = sample_value("Account", "Cuenta Principal")
        asset_type_default = sample_value("AssetType", "Stock")
        currency_default = sample_value("Currency", "USD")
        source_default = sample_value("Source", "manual")

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
            fees_input = st.number_input("Comisiones", min_value=0.0, value=0.0, step=0.01)
            taxes_input = st.number_input("Impuestos", min_value=0.0, value=0.0, step=0.01)
            note = st.text_input("Nota (opcional)", value="")
            submitted = st.form_submit_button("Agregar al portafolio")

            if submitted:
                if shares_real <= 0:
                    st.error("Necesito una cantidad de acciones mayor que cero.")
                elif price_entry <= 0:
                    st.error("El precio debe ser mayor que cero.")
                elif not headers:
                    st.error("No encontré encabezados válidos en la hoja Transactions.")
                else:
                    gross_amount = shares_real * price_entry
                    net_amount = gross_amount - fees_input - taxes_input

                    row_out = [""] * len(headers)

                    def set_field(col_name: str, value):
                        idx = col_index.get(_norm_header(col_name))
                        if idx is not None and idx < len(row_out):
                            row_out[idx] = value

                    set_field("TradeID", str(next_trade_id))
                    set_field("Account", account_default)
                    set_field("Ticker", cand)
                    set_field("Name", info.get("longName") or cand)
                    set_field("AssetType", asset_type_default)
                    set_field("Currency", currency_default)
                    set_field("TradeDate", trade_date.isoformat())
                    set_field("Side", "Buy")
                    set_field("Shares", shares_real)
                    set_field("Price", price_entry)
                    set_field("Fees", fees_input)
                    set_field("Taxes", taxes_input)
                    set_field("FXRate", 1.0)
                    set_field("GrossAmount", gross_amount)
                    set_field("NetAmount", net_amount)
                    set_field("LotID", f"L{next_lot_id}")
                    set_field("Source", source_default)
                    set_field("Notes", note)

                    try:
                        ws.append_row(row_out, value_input_option="USER_ENTERED")
                    except Exception as exc:
                        st.error(f"No pude registrar la compra: {exc}")
                    else:
                        st.success("Se registró la operación. Refresca 'Mi Portafolio' para verla reflejada.")
