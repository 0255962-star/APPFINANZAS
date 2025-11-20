"""Portfolio page renderer: render_portfolio_page(window)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..masters import build_masters, get_setting, masters_expired, need_build
from ..metrics import (
    annualize_return,
    annualize_vol,
    calmar,
    max_drawdown,
    sharpe,
    sortino,
)
from ..prices_fetch import normalize_symbol
from ..sheets_client import open_ws
from ..tx_positions import positions_from_tx
from ..ui_config import safe_rerun

DEFAULT_TX_HEADERS = [
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


def _norm_header(val: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(val).lower())


def _sample_value(values, col_index, column_name, default):
    idx = col_index.get(_norm_header(column_name))
    if idx is None:
        return default
    for row in values[1:]:
        if len(row) > idx and row[idx]:
            return row[idx]
    return default


def _next_numeric(values, col_index, column_name):
    idx = col_index.get(_norm_header(column_name))
    numbers = []
    if idx is not None:
        for row in values[1:]:
            if len(row) > idx:
                try:
                    numbers.append(int(float(row[idx])))
                except Exception:
                    continue
    return (max(numbers) if numbers else 0) + 1


def _register_transaction_row(
    ticker: str,
    name: str,
    shares: float,
    price: float,
    side: str,
    note: str = "",
    fees: float = 0.0,
    taxes: float = 0.0,
):
    ws = open_ws("Transactions")
    values = ws.get_all_values() or []
    headers = values[0] if values else DEFAULT_TX_HEADERS.copy()
    header_norm = [_norm_header(h) for h in headers]
    col_index = {norm: idx for idx, norm in enumerate(header_norm)}

    trade_id = _next_numeric(values, col_index, "TradeID")
    lot_id = _next_numeric(values, col_index, "LotID")
    account = _sample_value(values, col_index, "Account", "Cuenta Principal")
    asset_type = _sample_value(values, col_index, "AssetType", "Stock")
    currency = _sample_value(values, col_index, "Currency", "USD")
    source = _sample_value(values, col_index, "Source", "manual")

    gross = shares * price
    net = gross - fees - taxes
    trade_date = datetime.utcnow().date().isoformat()

    row_out = [""] * len(headers)

    def set_field(col_name: str, value):
        idx = col_index.get(_norm_header(col_name))
        if idx is not None and idx < len(row_out):
            row_out[idx] = value

    set_field("TradeID", str(trade_id))
    set_field("Account", account)
    set_field("Ticker", normalize_symbol(ticker))
    set_field("Name", name or ticker)
    set_field("AssetType", asset_type)
    set_field("Currency", currency)
    set_field("TradeDate", trade_date)
    set_field("Side", side)
    set_field("Shares", shares)
    set_field("Price", price)
    set_field("Fees", fees)
    set_field("Taxes", taxes)
    set_field("FXRate", 1.0)
    set_field("GrossAmount", gross)
    set_field("NetAmount", net)
    set_field("LotID", f"L{lot_id}")
    set_field("Source", source)
    set_field("Notes", note)

    ws.append_row(row_out, value_input_option="USER_ENTERED")


def _start_date_for_window(window: str) -> Optional[str]:
    period_map = {"6M": 180, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5}
    if window == "Max":
        return None
    days = period_map.get(window)
    if not days:
        return None
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")


def render_portfolio_page(window: str) -> None:
    """Render the portfolio tab."""
    start_date = _start_date_for_window(window)
    st.title("💼 Mi Portafolio")

    if st.button("🔄 Refrescar datos", use_container_width=False, type="secondary"):
        for k in (
            "tx_master",
            "settings_master",
            "prices_master",
            "bench_ret_master",
            "last_map_master",
            "_masters_built_at",
        ):
            st.session_state.pop(k, None)
        build_masters(sync=True)
        safe_rerun()

    if need_build("prices_master") or masters_expired():
        build_masters(sync=masters_expired())

    tx_df = st.session_state["tx_master"]
    name_lookup = {}
    if {"Ticker", "Name"}.issubset(tx_df.columns):
        tmp = tx_df.dropna(subset=["Ticker", "Name"])
        for _, row in tmp.iterrows():
            key = normalize_symbol(row["Ticker"])
            name_lookup[key] = row["Name"]
    settings_df = st.session_state["settings_master"]
    benchmark = get_setting(settings_df, "Benchmark", "SPY", str)
    rf = get_setting(settings_df, "RF", 0.03, float)

    prices_master = st.session_state["prices_master"]
    bench_ret_full = st.session_state["bench_ret_master"]
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

    last_hint_map = dict(st.session_state.get("last_map_master", {}))
    pos_df = positions_from_tx(tx_df, last_hint_map=last_hint_map)
    if pos_df.empty:
        st.info("No hay posiciones válidas tras procesar Transactions.")
        st.stop()

    total_mv = pos_df["MarketValue"].fillna(0).sum()
    w = (
        (pos_df.set_index("Ticker")["MarketValue"] / total_mv).sort_values(
            ascending=False
        )
        if total_mv > 0
        else pd.Series(dtype=float)
    )
    pos_df_view = (
        pos_df.set_index("Ticker").reindex(w.index).reset_index()
        if len(w) > 0
        else pos_df.copy()
    )

    since_buy = (
        pos_df.set_index("Ticker")["MarketPrice"] / pos_df.set_index("Ticker")["AvgCost"]
        - 1
    ).replace([np.inf, -np.inf], np.nan)
    window_change = pd.Series(index=pos_df_view["Ticker"], dtype=float)
    if prices is not None and not prices.empty and prices.shape[0] >= 2:
        window_change = (
            prices.ffill().iloc[-1] / prices.ffill().iloc[0] - 1
        ).reindex(pos_df_view["Ticker"]).fillna(np.nan)

    view = pd.DataFrame(
        {
            "Ticker": pos_df_view["Ticker"].values,
            "Shares": pos_df_view["Shares"].values,
            "Avg Buy": pos_df_view["AvgCost"].values,
            "Last": pos_df_view["MarketPrice"].values,
            "P/L $": pos_df_view["UnrealizedPL"].values,
            "P/L % (compra)": (
                since_buy.reindex(pos_df_view["Ticker"]).values * 100.0
            ),
            "Δ % ventana": (
                window_change.reindex(pos_df_view["Ticker"]).values * 100.0
            ),
            "Peso %": (w.reindex(pos_df_view["Ticker"]).values * 100.0),
            "Valor": pos_df_view["MarketValue"].values,
            "➖": [False] * len(pos_df_view),
        }
    ).replace([np.inf, -np.inf], np.nan)

    display_df = view.drop(columns=["➖"]).copy()

    def _colorize(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color:#22c55e;font-weight:bold;"
        if val < 0:
            return "color:#ef4444;font-weight:bold;"
        return ""

    if not display_df.empty:
        highlight_cols = [
            col
            for col in ["P/L $", "P/L % (compra)", "Δ % ventana"]
            if col in display_df.columns
        ]
        if highlight_cols:
            styler = (
                display_df.style.applymap(_colorize, subset=highlight_cols).hide(axis="index")
            )
        else:
            styler = display_df.style.hide(axis="index")
        st.dataframe(styler, use_container_width=True)
    else:
        st.info("No hay posiciones para mostrar en la tabla principal.")

    colcfg = {
        "➖": st.column_config.CheckboxColumn(
            label="➖",
            help="Marcar para registrar la venta completa de este ticker",
            width="small",
            default=False,
        )
    }
    selector_key = f"positions_selector_{window}"
    selector_df = st.data_editor(
        view[["Ticker", "➖"]],
        hide_index=True,
        use_container_width=True,
        column_config=colcfg,
        key=selector_key,
    )

    prev_key = f"prev_editor_df_{window}"
    prev = st.session_state.get(prev_key)
    if prev is None:
        st.session_state[prev_key] = selector_df.copy()
    else:
        mark = (selector_df["➖"] == True) & (prev["➖"] != True)
        if mark.any():
            row_idx = mark[mark].index[0]
            ticker_sel = str(selector_df.iloc[row_idx]["Ticker"])
            st.session_state["delete_candidate"] = ticker_sel
            details = {}
            if ticker_sel in view["Ticker"].values:
                details = (
                    view.loc[view["Ticker"] == ticker_sel].iloc[0].to_dict()
                )
            st.session_state["delete_info"] = details
        st.session_state[prev_key] = selector_df.copy()

    if st.session_state.get("delete_candidate"):
        tkr = st.session_state["delete_candidate"]
        info_row = st.session_state.get("delete_info", {})
        shares_to_sell = float(info_row.get("Shares", 0) or 0)
        price_to_use = float(info_row.get("Last", 0) or 0)
        st.warning(
            f"Registrarás la venta completa de **{shares_to_sell:,.2f}** acciones de **{tkr}**."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Sí, registrar venta"):
                if shares_to_sell <= 0 or price_to_use <= 0:
                    st.error("No tengo datos suficientes del precio o de las acciones para esta venta.")
                else:
                    try:
                        _register_transaction_row(
                            ticker=tkr,
                            name=name_lookup.get(normalize_symbol(tkr), name_lookup.get(tkr, tkr)),
                            shares=shares_to_sell,
                            price=price_to_use,
                            side="Sell",
                            note="Venta registrada desde Mi Portafolio",
                        )
                    except Exception as exc:
                        st.error(f"No pude registrar la venta: {exc}")
                    else:
                        st.success("Se registró la venta en la hoja Transactions.")
                        st.session_state["delete_candidate"] = ""
                        st.session_state.pop("delete_info", None)
                        st.session_state[prev_key]["➖"] = False
                        for k in (
                            "tx_master",
                            "settings_master",
                            "prices_master",
                            "bench_ret_master",
                            "last_map_master",
                        ):
                            st.session_state.pop(k, None)
                        build_masters(sync=False)
                        safe_rerun()
        with c2:
            if st.button("❌ No, cancelar"):
                st.session_state["delete_candidate"] = ""
                st.session_state.pop("delete_info", None)
                st.session_state[prev_key]["➖"] = False
                st.info("Operación cancelada.")

    port_ret = pd.Series(dtype=float)

    if prices is not None and not prices.empty and prices.shape[0] >= 2 and len(w) > 0:
        c_top1, c_top2 = st.columns([2, 1])
        with c_top1:
            rets = prices.pct_change().dropna(how="all")
            wv = w.reindex(rets.columns).fillna(0).values if len(w) > 0 else np.array([])
            port_ret = (
                (rets * wv).sum(axis=1) if rets.shape[1] and len(wv) else pd.Series(dtype=float)
            )

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Rend. anualizado", f"{(annualize_return(port_ret) or 0)*100:,.2f}%")
            c2.metric("Vol. anualizada", f"{(annualize_vol(port_ret) or 0)*100:,.2f}%")
            c3.metric("Sharpe", f"{(sharpe(port_ret, rf) or 0):.2f}")
            cum = (1 + port_ret).cumprod()
            mdd = max_drawdown(cum)
            c4.metric("Max Drawdown", f"{(mdd or 0)*100:,.2f}%")
            c5.metric("Sortino", f"{(sortino(port_ret, rf) or 0):.2f}")
            c6.metric("Calmar", f"{(calmar(port_ret) or 0):.2f}")

            curve = pd.DataFrame({"Portafolio": (1 + port_ret).cumprod()})
            if bench_ret is not None and not bench_ret.empty:
                curve["Benchmark"] = (
                    (1 + bench_ret).cumprod().reindex(curve.index).ffill()
                )
            st.plotly_chart(px.line(curve, title="Crecimiento de 1.0"), use_container_width=True)
        with c_top2:
            alloc = pd.DataFrame({"Ticker": w.index, "Weight": w.values})
            st.plotly_chart(px.pie(alloc, names="Ticker", values="Weight", title="Asignación"), use_container_width=True)
    elif len(w) == 0:
        st.info("No hay pesos válidos para calcular métricas.")
    else:
        st.info("Necesito al menos 2 días de histórico para calcular métricas.")

    st.markdown("### 🕑 Historial")
    history_df = tx_df.copy()
    if not history_df.empty:
        amount_col = "NetAmount" if "NetAmount" in history_df.columns else "GrossAmount"
        if "TradeDate" in history_df.columns:
            history_df["TradeDate"] = pd.to_datetime(history_df["TradeDate"], errors="coerce")
            history_df = history_df.sort_values("TradeDate", ascending=False)
        columns = ["TradeDate", "Ticker", "Name", "Side", "Shares", "Price"]
        if amount_col:
            columns.append(amount_col)
        columns = [c for c in columns if c in history_df.columns]
        hist_display = history_df[columns].head(7).copy()
        if "TradeDate" in hist_display.columns:
            hist_display["TradeDate"] = pd.to_datetime(
                hist_display["TradeDate"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        hist_display = hist_display.rename(
            columns={amount_col: "Importe"} if amount_col in hist_display.columns else {}
        )
        st.dataframe(hist_display, use_container_width=True)
    else:
        st.info("No hay movimientos registrados en la hoja de Transacciones.")
