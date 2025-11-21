"""Scenario simulator page: render_scenario_page(window)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..masters import build_masters, masters_expired, need_build
from ..metrics import annualize_vol, max_drawdown
from ..tx_positions import positions_from_tx
from ..ui_config import safe_rerun, style_signed_numbers


GROWTH_TICKERS = {
    "AMZN",
    "AAPL",
    "MSFT",
    "GOOGL",
    "NVDA",
    "TSLA",
    "META",
    "NFLX",
}
DEFENSIVE_TICKERS = {
    "KO",
    "PEP",
    "PG",
    "COST",
    "WMT",
    "JNJ",
    "MCD",
}


def _start_date_for_window(window: str):
    period_map = {"6M": 180, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5}
    if window == "Max":
        return None
    days = period_map.get(window)
    if not days:
        return None
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")


def _bucket_for(ticker: str) -> str:
    t = ticker.upper().strip()
    if t in GROWTH_TICKERS:
        return "Crecimiento / alto beta"
    if t in DEFENSIVE_TICKERS:
        return "Defensivo / consumo"
    return "Mixto"


def _scenario_shocks(
    option: str,
    tickers: Iterable[str],
    custom_controls: Dict[str, float],
    custom_ticker: str = "",
    custom_ticker_shock: float = 0.0,
) -> Dict[str, float]:
    shocks = {}
    tset = [t.upper().strip() for t in tickers]

    def apply_all(val: float):
        for t in tset:
            shocks[t] = val

    if option == "Caída del mercado -10%":
        apply_all(-0.10)
    elif option == "Caída del mercado -20%":
        apply_all(-0.20)
    elif option == "Rally +15% en acciones growth":
        for t in tset:
            bucket = _bucket_for(t)
            if bucket == "Crecimiento / alto beta":
                shocks[t] = 0.15
            elif bucket == "Defensivo / consumo":
                shocks[t] = 0.02
            else:
                shocks[t] = 0.05
    elif option == "Shock en tasas +1%":
        for t in tset:
            bucket = _bucket_for(t)
            if bucket == "Crecimiento / alto beta":
                shocks[t] = -0.08
            elif bucket == "Defensivo / consumo":
                shocks[t] = -0.04
            else:
                shocks[t] = -0.06
    elif option == "Escenario personalizado":
        for t in tset:
            bucket = _bucket_for(t)
            if bucket == "Crecimiento / alto beta":
                shocks[t] = custom_controls.get("high_beta", 0.0)
            elif bucket == "Defensivo / consumo":
                shocks[t] = custom_controls.get("defensive", 0.0)
            else:
                shocks[t] = custom_controls.get("other", 0.0)
    else:
        apply_all(0.0)

    if custom_ticker:
        shocks[custom_ticker.upper().strip()] = custom_ticker_shock

    return shocks


def _compute_portfolio_returns(prices: pd.DataFrame, weights: pd.Series, start_date):
    if prices is None or prices.empty or weights.empty:
        return pd.Series(dtype=float)
    subset = prices.copy()
    if start_date:
        subset = subset[subset.index >= start_date]
    subset = subset[weights.index.intersection(subset.columns)]
    if subset.empty:
        return pd.Series(dtype=float)
    returns = subset.pct_change().dropna()
    if returns.empty:
        return pd.Series(dtype=float)
    aligned_weights = weights.reindex(returns.columns).fillna(0)
    return returns.dot(aligned_weights)


def _scenario_metrics(port_returns: pd.Series, port_change: float) -> Tuple[float, float, float, float]:
    base_vol = annualize_vol(port_returns) if not port_returns.empty else np.nan
    base_mdd = max_drawdown((1 + port_returns).cumprod()) if not port_returns.empty else np.nan
    scenario_vol = base_vol * (1 + abs(port_change)) if not np.isnan(base_vol) else np.nan
    scenario_mdd = base_mdd
    if np.isnan(base_mdd):
        scenario_mdd = -abs(port_change) if port_change < 0 else -abs(port_change) * 0.5
    else:
        shock_drawdown = -abs(port_change)
        scenario_mdd = min(base_mdd, shock_drawdown)
    return (
        base_vol,
        scenario_vol if not np.isnan(scenario_vol) else np.nan,
        base_mdd,
        scenario_mdd,
    )


def _format_pct(val: float) -> str:
    if val is None or np.isnan(val):
        return "N/D"
    return f"{val * 100:,.2f}%"


def _format_money(val: float) -> str:
    if val is None or np.isnan(val):
        return "N/D"
    return f"${val:,.2f}"


def _claude_scenario_explanation(prompt: str) -> str:
    import requests

    api_key = st.secrets.get("CLAUDE_API_KEY")
    if not api_key:
        return "No encontré la clave de Claude en los secrets."

    preferred_model = st.secrets.get("CLAUDE_MODEL", "claude-3-sonnet-20240229")
    fallback_model = "claude-3-haiku-20240307"

    def _attempt(model_name: str):
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept": "application/json",
            "anthropic-beta": "messages-2023-12-15",
        }
        payload = {
            "model": model_name,
            "max_tokens": 700,
            "temperature": 0.35,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - network guarded
            return None, f"No pude obtener la explicación de IA. Detalle técnico: {exc}"

        try:
            data = resp.json()
        except Exception as exc:  # pragma: no cover - network guarded
            return None, f"No pude interpretar la respuesta de IA. Detalle técnico: {exc}"
        return data, None

    data, err = _attempt(preferred_model)
    if err and preferred_model != fallback_model:
        data, err = _attempt(fallback_model)

    if err:
        return err

    content = data.get("content") if isinstance(data, dict) else None
    if isinstance(content, list) and content:
        blocks = [b.get("text") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        blocks = [b for b in blocks if b]
        if blocks:
            return "\n\n".join(blocks)
    if isinstance(content, str):
        return content
    return "No recibí texto de respuesta de Claude."


def render_scenario_page(window: str) -> None:
    start_date = _start_date_for_window(window)
    st.title("Simulador avanzado de escenarios")

    def _build_or_fail(sync: bool = False):
        try:
            build_masters(sync=sync)
        except Exception as exc:  # pragma: no cover - defensive UI handling
            st.error(
                "No pude cargar los datos desde Google Sheets. "
                "Revisa tus secrets o la conexión e inténtalo de nuevo."
            )
            st.info(f"Detalle técnico: {exc}")
            st.stop()

    if st.button("Refrescar datos", type="secondary"):
        st.cache_data.clear()
        for k in (
            "tx_master",
            "settings_master",
            "prices_master",
            "bench_ret_master",
            "last_map_master",
            "_masters_built_at",
        ):
            st.session_state.pop(k, None)
        _build_or_fail(sync=True)
        safe_rerun()

    if need_build("prices_master") or masters_expired():
        _build_or_fail(sync=masters_expired())

    tx_df = st.session_state.get("tx_master")
    if tx_df is None or tx_df.empty:
        st.info("No hay transacciones cargadas para simular escenarios.")
        return

    last_map = st.session_state.get("last_map_master", {})
    positions = positions_from_tx(tx_df, last_map)
    positions["Ticker"] = positions["Ticker"].astype(str).str.upper()
    positions = positions.dropna(subset=["MarketPrice", "MarketValue"])
    positions = positions[positions["MarketValue"].abs() > 0]

    if positions.empty:
        st.info("No se encontraron posiciones vigentes para simular.")
        return

    total_value = positions["MarketValue"].sum()
    weights = (positions.set_index("Ticker")["MarketValue"] / total_value).fillna(0)
    price_history = st.session_state.get("prices_master", pd.DataFrame())
    port_returns = _compute_portfolio_returns(price_history, weights, start_date)

    scenario_option = st.radio(
        "Selecciona un escenario:",
        [
            "Caída del mercado -10%",
            "Caída del mercado -20%",
            "Rally +15% en acciones growth",
            "Shock en tasas +1%",
            "Escenario personalizado",
        ],
        index=0,
        horizontal=True,
    )

    custom_ticker = ""
    custom_controls = {"high_beta": -0.1, "defensive": -0.05, "other": -0.08}
    custom_ticker_shock = 0.0
    if scenario_option == "Escenario personalizado":
        with st.expander("Ajustes personalizados", expanded=True):
            custom_controls["high_beta"] = st.slider(
                "Cambio % en acciones de alto beta / growth",
                -30.0,
                30.0,
                value=-10.0,
                step=1.0,
            )
            custom_controls["defensive"] = st.slider(
                "Cambio % en acciones defensivas",
                -30.0,
                30.0,
                value=-5.0,
                step=1.0,
            )
            custom_controls["other"] = st.slider(
                "Cambio % en el resto del portafolio",
                -30.0,
                30.0,
                value=-8.0,
                step=1.0,
            )
            custom_ticker = st.text_input(
                "Ticker con shock específico (opcional)", value=""
            ).strip()
            if custom_ticker:
                custom_ticker_shock = st.slider(
                    "Shock % para el ticker específico",
                    -50.0,
                    50.0,
                    value=-15.0,
                    step=1.0,
                )
        custom_controls = {k: v / 100 for k, v in custom_controls.items()}
        custom_ticker_shock = custom_ticker_shock / 100 if custom_ticker else 0.0

    shock_map = _scenario_shocks(
        scenario_option,
        positions["Ticker"].tolist(),
        custom_controls,
        custom_ticker,
        custom_ticker_shock,
    )

    positions["ShockPct"] = positions["Ticker"].map(shock_map).fillna(0.0)
    positions["ScenarioPrice"] = positions["MarketPrice"] * (1 + positions["ShockPct"])
    positions["ScenarioValue"] = positions["ScenarioPrice"] * positions["Shares"]
    positions["DeltaValue"] = positions["ScenarioValue"] - positions["MarketValue"]

    scenario_total = positions["ScenarioValue"].sum()
    change_abs = scenario_total - total_value
    change_pct = change_abs / total_value if total_value else 0.0
    base_vol, scenario_vol, base_mdd, scenario_mdd = _scenario_metrics(port_returns, change_pct)

    col1, col2, col3 = st.columns(3)
    col1.metric("Valor actual", _format_money(total_value))
    col2.metric("Valor en escenario", _format_money(scenario_total), _format_pct(change_pct))
    col3.metric(
        "Volatilidad anualizada (aprox)",
        _format_pct(base_vol),
        _format_pct((scenario_vol - base_vol) if not np.isnan(scenario_vol) and not np.isnan(base_vol) else np.nan),
    )

    col4, col5, _ = st.columns(3)
    col4.metric("Max drawdown base (aprox)", _format_pct(base_mdd))
    col5.metric("Max drawdown en escenario (aprox)", _format_pct(scenario_mdd))

    st.subheader("Impacto por posición")
    display_df = positions[["Ticker", "MarketPrice", "ScenarioPrice", "MarketValue", "ScenarioValue", "ShockPct", "DeltaValue"]]
    display_df = display_df.rename(
        columns={
            "MarketPrice": "Precio actual",
            "ScenarioPrice": "Precio escenario",
            "MarketValue": "Valor actual",
            "ScenarioValue": "Valor escenario",
            "ShockPct": "% cambio",
            "DeltaValue": "Δ valor",
        }
    )
    styled_positions = style_signed_numbers(display_df, ["% cambio", "Δ valor"])
    styled_positions = styled_positions.format(
        {
            "Precio actual": "{:.2f}",
            "Precio escenario": "{:.2f}",
            "Valor actual": "{:.2f}",
            "Valor escenario": "{:.2f}",
            "% cambio": "{:.2%}",
            "Δ valor": "{:.2f}",
        }
    )
    st.dataframe(styled_positions, use_container_width=True)

    tab_val, tab_impact, tab_dist = st.tabs([
        "Valor del portafolio",
        "Impacto por ticker / bucket",
        "Distribución simulada",
    ])

    with tab_val:
        chart_df = pd.DataFrame(
            {
                "Escenario": ["Actual", scenario_option],
                "Valor": [total_value, scenario_total],
            }
        )
        fig = px.bar(chart_df, x="Escenario", y="Valor", text=["", _format_pct(change_pct)])
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_impact:
        positions["Bucket"] = positions["Ticker"].apply(_bucket_for)
        bucket_impact = (
            positions.groupby("Bucket")["DeltaValue"].sum().reset_index().sort_values("DeltaValue")
        )
        impact_fig = px.bar(
            positions,
            x="Ticker",
            y="ShockPct",
            color="Bucket",
            labels={"ShockPct": "% cambio simulado"},
            text=positions["ShockPct"].apply(lambda x: f"{x*100:.1f}%"),
        )
        impact_fig.update_layout(template="plotly_dark")
        st.plotly_chart(impact_fig, use_container_width=True)

        st.caption("Contribución por bucket (Δ valor total)")
        bucket_fig = px.bar(bucket_impact, x="Bucket", y="DeltaValue", text="DeltaValue")
        bucket_fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(bucket_fig, use_container_width=True)

    with tab_dist:
        scenario_weights = (
            positions.set_index("Ticker")["ScenarioValue"] / scenario_total if scenario_total else pd.Series(dtype=float)
        )
        dist_df = scenario_weights.reset_index()
        dist_df.columns = ["Ticker", "Peso"]
        pie = px.pie(dist_df, values="Peso", names="Ticker")
        pie.update_layout(template="plotly_dark")
        st.plotly_chart(pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Explicación de IA del escenario")
    ia_container = st.container()

    def _prompt_for_claude():
        bucket_view = (
            positions.groupby("Bucket")
            .agg(
                valor_actual=("MarketValue", "sum"),
                valor_escenario=("ScenarioValue", "sum"),
                delta=("DeltaValue", "sum"),
            )
            .reset_index()
        )
        bucket_lines = []
        for _, row in bucket_view.iterrows():
            delta_pct = row["delta"] / row["valor_actual"] if row["valor_actual"] else 0
            bucket_lines.append(
                f"- {row['Bucket']}: { _format_money(row['valor_actual']) } → { _format_money(row['valor_escenario']) } (Δ { _format_pct(delta_pct) })"
            )

        top_impacts = positions.copy()
        top_impacts["Impacto %"] = top_impacts.apply(
            lambda r: r["DeltaValue"] / r["MarketValue"] if r["MarketValue"] else 0, axis=1
        )
        top_impacts = top_impacts.sort_values("Impacto %", key=lambda s: s.abs(), ascending=False).head(5)
        impact_lines = [
            f"- {row['Ticker']}: shock {_format_pct(row['ShockPct'])}, Δ valor {_format_money(row['DeltaValue'])}"
            for _, row in top_impacts.iterrows()
        ]

        prompt = (
            "Contexto: eres un profesor de finanzas explicando a un estudiante mexicano de 7º semestre. "
            "Tono formal-casual, claro y técnico, con alguna frase ligera ocasional.\n"
            f"Simulamos el escenario '{scenario_option}'.\n"
            "Datos clave (úsalos para interpretar, no para recitar):\n"
            f"- Valor actual del portafolio: {_format_money(total_value)}\n"
            f"- Valor bajo el escenario: {_format_money(scenario_total)} (Δ {_format_pct(change_pct)})\n"
            f"- Volatilidad anualizada base: {_format_pct(base_vol)}; simulada: {_format_pct(scenario_vol)}\n"
            f"- Max drawdown base: {_format_pct(base_mdd)}; estimado en escenario: {_format_pct(scenario_mdd)}\n"
            f"- Buckets y contribuciones:\n{chr(10).join(bucket_lines) if bucket_lines else '- Sin desglose disponible'}\n"
            f"- Principales impactos por ticker:\n{chr(10).join(impact_lines)}\n"
            "Instrucciones de salida: organiza en bloques (impacto global; concentración de daño/riesgo; estructura del portafolio y ajustes conceptuales). "
            "No repitas todos los números, interprétalos para el alumno. Si el escenario es severo, dilo directo. "
            "Cierra con una conclusión de 3–4 líneas sobre qué tan preparado está el portafolio, qué perfil de inversionista lo tolera y si conviene diversificar o ajustar concentraciones."
        )
        return prompt

    with ia_container:
        if st.button("Explicación de IA del escenario"):
            with st.spinner("Consultando a Claude…"):
                prompt = _prompt_for_claude()
                response = _claude_scenario_explanation(prompt)
                st.write(response)

