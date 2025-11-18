"""Portfolio page renderer: render_portfolio_page(window)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..analysis_utils import build_metrics_comparison, interpret_candidate_effect
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
from ..tx_positions import delete_transactions_by_ticker, positions_from_tx
from ..ui_config import safe_rerun


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

    colcfg = {
        "➖": st.column_config.CheckboxColumn(
            label="➖",
            help="Marcar para eliminar este ticker",
            width="small",
            default=False,
        )
    }
    editor_key = f"positions_editor_{window}"
    disabled_cols = [c for c in view.columns if c != "➖"]
    edited = st.data_editor(
        view,
        hide_index=True,
        use_container_width=True,
        column_config=colcfg,
        disabled=disabled_cols,
        key=editor_key,
    )

    prev_key = f"prev_editor_df_{window}"
    prev = st.session_state.get(prev_key)
    if prev is None:
        st.session_state[prev_key] = edited.copy()
    else:
        mark = (edited["➖"] == True) & (prev["➖"] != True)
        if mark.any():
            row_idx = mark[mark].index[0]
            st.session_state["delete_candidate"] = str(
                edited.iloc[row_idx]["Ticker"]
            )
        st.session_state[prev_key] = edited.copy()

    if st.session_state.get("delete_candidate"):
        tkr = st.session_state["delete_candidate"]
        st.warning(
            f"¿Seguro que quieres eliminar **todas** las transacciones de **{tkr}** en *Transactions*?"
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Sí, eliminar"):
                deleted = delete_transactions_by_ticker(tkr)
                st.session_state["delete_candidate"] = ""
                st.session_state[prev_key]["➖"] = False
                if deleted > 0:
                    st.success(f"Se eliminaron {deleted} fila(s) de {tkr}.")
                else:
                    st.info("No se encontraron filas para eliminar.")
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

    # --- Nueva funcionalidad: expander para pre-evaluar candidatos ---
    with st.expander("Evaluar candidato antes de añadirlo"):
        st.caption("Simulación rápida para estimar el impacto antes de registrar un nuevo peso.")
        eval_cols = st.columns([2, 1])
        eval_key = f"pre_eval_{window}"
        with eval_cols[0]:
            cand_input = st.text_input(
                "Ticker a evaluar",
                placeholder="Ej. NVDA, KO, BRK.B",
                key=f"{eval_key}_ticker",
            )
        with eval_cols[1]:
            weight_sim = st.slider(
                "Peso hipotético",
                min_value=0.0,
                max_value=0.10,
                value=0.02,
                step=0.01,
                key=f"{eval_key}_weight",
                help="Peso hipotético que tendría el nuevo activo en tu portafolio.",
            )
        trigger = st.button("Evaluar impacto", key=f"{eval_key}_btn")

        if trigger:
            if not cand_input:
                st.info("Ingresa un ticker para evaluar su incorporación.")
            elif prices is None or prices.empty or prices.shape[0] < 2 or port_ret.empty:
                st.info("Necesito histórico del portafolio para correr la simulación.")
            else:
                cand = normalize_symbol(cand_input)
                if not re.match(r"^[A-Z0-9\.\-]+$", cand):
                    st.error("Ticker inválido.")
                else:
                    sim_start = prices.index[0].strftime("%Y-%m-%d")
                    cand_hist = ensure_prices([cand], sim_start, persist=True)
                    if cand_hist is None or cand_hist.empty or cand not in cand_hist.columns:
                        st.warning("No encontré datos suficientes de ese ticker.")
                    else:
                        cand_px = cand_hist[cand].reindex(prices.index).ffill()
                        cand_ret = cand_px.pct_change().dropna()
                        combo = pd.concat(
                            [port_ret.rename("port"), cand_ret.rename("cand")], axis=1
                        ).dropna()
                        if combo.empty:
                            st.info("No hubo intersección suficiente de fechas para correr la simulación.")
                        else:
                            port_base = combo["port"]
                            cand_aligned = combo["cand"]
                            port_ret_new = port_base * (1 - weight_sim) + cand_aligned * weight_sim
                            comp = build_metrics_comparison(port_base, port_ret_new, rf)
                            st.dataframe(comp.table, use_container_width=True)

                            growth = pd.DataFrame(
                                {
                                    "Actual": (1 + port_base).cumprod(),
                                    f"+{cand}": (1 + port_ret_new).cumprod(),
                                }
                            )
                            st.plotly_chart(
                                px.line(
                                    growth,
                                    title=f"Crecimiento de 1.0: Actual vs +{cand}",
                                    labels={"index": "Fecha", "value": "Índice"},
                                ),
                                use_container_width=True,
                            )
                            interp = interpret_candidate_effect(
                                comp.current, comp.new, cand, weight_sim
                            )
                            st.markdown(
                                f"> {interp.replace(chr(10), '<br>')}",
                                unsafe_allow_html=True,
                            )
    # --- Fin nueva funcionalidad: expander para pre-evaluar candidatos ---
