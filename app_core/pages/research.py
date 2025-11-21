"""Research/exploration page renderer: render_research_page(window)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import requests

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


def _start_date_for_window(window: str) -> Optional[str]:
    days_map = {
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
        "3Y": 365 * 3,
        "5Y": 365 * 5,
    }
    if window == "Max":
        return None
    days = days_map.get(window)
    if not days:
        return None
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")


def _safe_number(val, default=None):
    if val in (None, "", "NaN"):
        return default
    if isinstance(val, (int, float, np.number)):
        try:
            return float(val)
        except Exception:
            return default
    if isinstance(val, str):
        cleaned = val.strip().replace(",", "")
        try:
            return float(cleaned)
        except Exception:
            return default
    return default


def _fmt_percent(val) -> str:
    num = _safe_number(val)
    if num is None:
        return "—"
    return f"{num*100:,.2f}%" if abs(num) < 2 else f"{num:,.2f}%"


def _secret_or_none(key: str):
    try:
        return st.secrets.get(key)
    except Exception:
        return None


@st.cache_data(ttl=1200, show_spinner=False)
def fetch_ticker_snapshot(ticker: str) -> Dict:
    t = yf.Ticker(ticker)
    info: Dict = {}

    def _alpha_overview(symbol: str) -> Dict:
        api_key = _secret_or_none("ALPHAVANTAGE_API_KEY")
        if not api_key:
            return {}
        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "OVERVIEW",
                    "symbol": symbol,
                    "apikey": api_key,
                },
                timeout=10,
            )
            payload = resp.json() if resp.ok else {}
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _from_fast(key):
        try:
            fast = getattr(t, "fast_info", None)
            if fast is None:
                return None
            if isinstance(fast, dict):
                return fast.get(key)
            return getattr(fast, key, None)
        except Exception:
            return None

    def _from_info(*keys):
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
        for key in keys:
            if key in base:
                return base.get(key)
        return None

    overview = _alpha_overview(ticker)

    info["longName"] = _from_info("longName") or _from_info("shortName")
    info["symbol"] = ticker.upper()
    info["currency"] = _from_fast("currency") or _from_info("currency") or overview.get("Currency")
    info["regularMarketPrice"] = (
        _from_fast("last_price")
        or _from_fast("lastPrice")
        or _from_info("regularMarketPrice", "currentPrice")
        or _safe_number(overview.get("50DayMovingAverage"))
    )
    info["regularMarketPreviousClose"] = _from_fast("previousClose") or _from_info(
        "previousClose"
    )
    info["regularMarketChangePercent"] = _from_fast("regularMarketChangePercent") or _from_info(
        "regularMarketChangePercent"
    )
    info["marketCap"] = _from_fast("marketCap") or _from_info("marketCap") or _safe_number(
        overview.get("MarketCapitalization")
    )
    info["sector"] = _from_info("sector") or overview.get("Sector")
    info["industry"] = _from_info("industry") or overview.get("Industry")
    info["country"] = _from_info("country") or overview.get("Country")
    info["logo_url"] = _from_info("logo_url")
    info["fiftyTwoWeekLow"] = (
        _from_fast("yearLow")
        or _from_info("fiftyTwoWeekLow")
        or _safe_number(overview.get("52WeekLow"))
    )
    info["fiftyTwoWeekHigh"] = (
        _from_fast("yearHigh")
        or _from_info("fiftyTwoWeekHigh")
        or _safe_number(overview.get("52WeekHigh"))
    )
    info["averageVolume"] = _from_fast("tenDayAverageVolume") or _from_info(
        "averageVolume"
    )
    info["trailingPE"] = _from_info("trailingPE") or _safe_number(overview.get("PERatio"))
    info["forwardPE"] = _from_info("forwardPE") or _safe_number(overview.get("ForwardPE"))
    info["dividendYield"] = _from_info("dividendYield") or _from_info(
        "trailingAnnualDividendYield"
    )
    if info.get("dividendYield") is None:
        info["dividendYield"] = _safe_number(overview.get("DividendYield"))
    info["pegRatio"] = _from_info("pegRatio") or _safe_number(overview.get("PEGRatio"))
    info["priceToBook"] = _from_info("priceToBook") or _safe_number(overview.get("PriceToBookRatio"))
    info["operatingMargins"] = _from_info("operatingMargins") or _safe_number(
        overview.get("OperatingMarginTTM")
    )
    info["profitMargins"] = _from_info("profitMargins") or _safe_number(
        overview.get("ProfitMargin")
    )
    info["returnOnEquity"] = _from_info("returnOnEquity") or _safe_number(
        overview.get("ReturnOnEquityTTM")
    )
    info["returnOnAssets"] = _from_info("returnOnAssets") or _safe_number(
        overview.get("ReturnOnAssetsTTM")
    )
    info["debtToEquity"] = _from_info("debtToEquity") or _safe_number(
        overview.get("DebtToEquity")
    )

    if _safe_number(info.get("regularMarketPrice")) is None:
        api_key = _secret_or_none("ALPHAVANTAGE_API_KEY")
        if api_key:
            try:
                resp = requests.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "GLOBAL_QUOTE",
                        "symbol": ticker,
                        "apikey": api_key,
                    },
                    timeout=10,
                )
                quote = resp.json().get("Global Quote", {})
                last = _safe_number(quote.get("05. price"))
                prev = _safe_number(quote.get("08. previous close"))
                if last is not None:
                    info["regularMarketPrice"] = last
                if prev is not None:
                    info["regularMarketPreviousClose"] = prev
                if info.get("currency") is None:
                    info["currency"] = "USD"
            except Exception:
                pass
    return info


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str, start: Optional[str]):
    def _alpha_history(symbol: str, want_full: bool) -> pd.DataFrame:
        api_key = _secret_or_none("ALPHAVANTAGE_API_KEY")
        if not api_key:
            return pd.DataFrame()
        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": symbol,
                    "outputsize": "full" if want_full else "compact",
                    "apikey": api_key,
                },
                timeout=15,
            )
            payload = resp.json()
            if not isinstance(payload, dict):
                return pd.DataFrame()
            if "Note" in payload or "Error Message" in payload:
                return pd.DataFrame()
            data = payload.get("Time Series (Daily)", {})
        except Exception:
            return pd.DataFrame()
        if not data:
            return pd.DataFrame()

        records = []
        for date_str, vals in data.items():
            try:
                records.append(
                    {
                        "Date": datetime.strptime(date_str, "%Y-%m-%d"),
                        "AdjClose": float(vals.get("5. adjusted close", "nan")),
                        "Close": float(vals.get("4. close", "nan")),
                        "Volume": float(vals.get("6. volume", 0)),
                    }
                )
            except Exception:
                continue
        if not records:
            return pd.DataFrame()

        df_alpha = pd.DataFrame(records).set_index("Date").sort_index()
        df_alpha.attrs["currency"] = "USD"
        return df_alpha

    want_full_history = False
    if start:
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            want_full_history = (datetime.utcnow() - start_dt).days > 150
        except Exception:
            want_full_history = False

    try:
        df = yf.download(
            ticker,
            start=start or None,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df.empty:
            df = yf.download(
                ticker, period="max", progress=False, auto_adjust=False, threads=False
            )
        if df.empty:
            df = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        df = _alpha_history(ticker, want_full_history)
    if df.empty:
        return df

    # Normalize column names and ensure AdjClose exists even when yfinance omits it.
    df = df.rename(columns={"Adj Close": "AdjClose"})
    if "AdjClose" not in df.columns and "Close" in df.columns:
        df["AdjClose"] = df["Close"]
    # Drop rows that have no usable price information.
    if "AdjClose" in df.columns:
        df = df.dropna(subset=["AdjClose"]).copy()
    return df


def _render_snapshot_card(info: Dict):
    name = info.get("longName") or info.get("symbol")
    symbol = info.get("symbol")
    logo_url = info.get("logo_url")
    cols = st.columns([3, 1])
    with cols[0]:
        st.subheader(f"{name} ({symbol})")
    with cols[1]:
        if logo_url:
            st.image(logo_url, width=60)

    price = _safe_number(info.get("regularMarketPrice"))
    prev_close = _safe_number(info.get("regularMarketPreviousClose"))
    pct = None
    if price is not None and prev_close not in (None, 0):
        pct = (price - prev_close) / prev_close * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Precio", f"{price:,.2f}" if price is not None else "—", delta=f"{pct:,.2f}%" if pct is not None else None)
    m2.metric("Cap. de mercado", f"{_safe_number(info.get('marketCap'), 0)/1_000_000_000:,.2f} B" if _safe_number(info.get('marketCap')) else "—")
    m3.metric("Div. yield", _fmt_percent(info.get("dividendYield")))

    cols = st.columns(4)
    cols[0].write(f"**Sector:** {info.get('sector') or '—'}")
    cols[1].write(f"**Industria:** {info.get('industry') or '—'}")
    cols[2].write(f"**País:** {info.get('country') or '—'}")
    cols[3].write(f"**Divisa:** {info.get('currency') or '—'}")

    cols = st.columns(4)
    low_52 = _safe_number(info.get('fiftyTwoWeekLow'))
    high_52 = _safe_number(info.get('fiftyTwoWeekHigh'))
    cols[0].write(
        f"**Rango 52S:** {low_52:,.2f}" if low_52 is not None else "**Rango 52S:** —"
        + (f" / {high_52:,.2f}" if high_52 is not None else " / —")
    )

    avg_vol = _safe_number(info.get('averageVolume'))
    pe = _safe_number(info.get('trailingPE'))
    fpe = _safe_number(info.get('forwardPE'))

    cols[1].write(f"**Vol. promedio:** {avg_vol:,.0f}" if avg_vol is not None else "**Vol. promedio:** —")
    cols[2].write(f"**PE (TTM):** {pe:,.2f}" if pe is not None else "**PE (TTM):** —")
    cols[3].write(f"**Forward PE:** {fpe:,.2f}" if fpe is not None else "**Forward PE:** —")


def _filter_history(hist: pd.DataFrame, window: str) -> pd.DataFrame:
    if hist.empty:
        return hist
    start = _start_date_for_window(window)
    if not start:
        return hist
    return hist.loc[hist.index >= start]


def _render_price_tab(ticker: str, hist: pd.DataFrame):
    st.subheader("Precio histórico")
    if hist.empty or "AdjClose" not in hist.columns:
        st.info("Sin datos históricos para graficar este ticker.")
        return

    hist = hist.copy()
    hist["MA50"] = hist["AdjClose"].rolling(50).mean()
    hist["MA200"] = hist["AdjClose"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["AdjClose"], mode="lines", name="Precio"))
    if hist["MA50"].notna().sum() > 0:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["MA50"], mode="lines", name="MA 50"))
    if hist["MA200"].notna().sum() > 0:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["MA200"], mode="lines", name="MA 200"))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _render_volume_volatility(hist: pd.DataFrame):
    st.subheader("Volumen y volatilidad")
    if hist.empty or "AdjClose" not in hist.columns:
        st.info("Sin datos de volumen/volatilidad disponibles.")
        return

    if "Volume" not in hist.columns:
        st.info("No hay datos de volumen disponibles para este ticker.")
    else:
        vol_fig = px.bar(hist, x=hist.index, y="Volume", labels={"x": "Fecha", "Volume": "Volumen"})
        vol_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))

    returns = hist["AdjClose"].pct_change().dropna()
    if returns.empty:
        st.info("No hay suficientes datos para calcular volatilidad.")
        return

    rolling_vol = returns.rolling(30).std() * np.sqrt(252)
    vol_line = px.line(
        rolling_vol,
        labels={"value": "Volatilidad 30d anualizada", "index": "Fecha"},
    )
    vol_line.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))

    c1, c2 = st.columns(2)
    with c1:
        if "Volume" in hist.columns:
            st.plotly_chart(vol_fig, use_container_width=True)
    with c2:
        st.plotly_chart(vol_line, use_container_width=True)


def _render_ratios(info: Dict):
    st.subheader("Ratios y fundamentales")
    data = {
        "P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "PEG": info.get("pegRatio"),
        "Price/Book": info.get("priceToBook"),
        "Margen operativo": info.get("operatingMargins"),
        "Margen neto": info.get("profitMargins"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "Deuda/Equity": info.get("debtToEquity"),
        "Dividend yield": info.get("dividendYield"),
    }

    formatted = {
        key: (_fmt_percent(val) if "Margen" in key or "ROE" in key or "ROA" in key or "Dividend" in key else f"{_safe_number(val):,.2f}" if _safe_number(val) is not None else "—")
        for key, val in data.items()
    }
    df = pd.DataFrame(formatted.items(), columns=["Métrica", "Valor"])
    st.table(df)


def _render_benchmark_compare(ticker: str, hist: pd.DataFrame, start: Optional[str]):
    st.subheader("Comparación vs benchmark (SPY)")
    if hist.empty or "AdjClose" not in hist.columns:
        st.info("Sin datos para comparar contra el benchmark.")
        return

    spy = fetch_history("SPY", start)
    if spy.empty or "AdjClose" not in spy.columns:
        st.info("No se pudieron obtener datos de SPY para comparar.")
        return

    merged = pd.DataFrame({
        ticker.upper(): hist["AdjClose"],
        "SPY": spy["AdjClose"],
    }).dropna()
    if merged.empty:
        st.info("Sin suficientes datos para la comparación.")
        return

    normed = merged / merged.iloc[0]
    fig = px.line(normed, labels={"value": "Rendimiento acumulado", "index": "Fecha", "variable": "Símbolo"})
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    perf = (merged.iloc[-1] / merged.iloc[0] - 1) * 100
    st.write(
        f"Rendimiento en el periodo — {ticker.upper()}: {perf[ticker.upper()]:,.2f}%, SPY: {perf['SPY']:,.2f}%"
    )


def render_research_page(window: str) -> None:
    st.title("Explorar / Research")

    with st.form(key="ticker_search"):
        raw_ticker = st.text_input("Ticker (ej. AAPL, NVDA, KO):", value="")
        submitted = st.form_submit_button("Buscar")

    ticker = raw_ticker.strip().upper()

    if not submitted:
        st.info("Ingresa un ticker para explorar su información.")
        return

    if not ticker:
        st.warning("Debes ingresar un ticker válido.")
        return

    base_start = _start_date_for_window(window)

    with st.spinner("Cargando información del ticker..."):
        try:
            info = fetch_ticker_snapshot(ticker)
            hist = fetch_history(ticker, base_start)
        except Exception as exc:  # pragma: no cover - defensive for network errors
            st.error(
                f"No se pudo obtener información para el ticker {ticker.upper()}. Verifica que esté bien escrito."
            )
            st.caption(str(exc))
            return

    # If fast/info endpoints don't surface price data, fall back to the latest close from history.
    if hist is not None and not isinstance(hist, pd.DataFrame):
        hist = pd.DataFrame()
    if _safe_number((info or {}).get("regularMarketPrice")) is None and isinstance(hist, pd.DataFrame) and not hist.empty:
        last_close = None
        prev_close = None
        if "Close" in hist.columns:
            last_close = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else None
        elif "AdjClose" in hist.columns:
            last_close = hist["AdjClose"].iloc[-1]
            prev_close = hist["AdjClose"].iloc[-2] if len(hist) > 1 else None
        if last_close is not None:
            info["regularMarketPrice"] = last_close
            if info.get("currency") is None:
                info["currency"] = hist.attrs.get("currency")
        if prev_close is not None:
            info["regularMarketPreviousClose"] = prev_close

    if not info or (_safe_number(info.get("regularMarketPrice")) is None and (hist is None or hist.empty)):
        st.error(
            f"No se pudo obtener información para el ticker {ticker.upper()}. Verifica que esté bien escrito."
        )
        return

    # Enrich snapshot with history-based fallbacks when API responses omit metadata.
    if hist is not None and isinstance(hist, pd.DataFrame) and not hist.empty:
        if _safe_number(info.get("averageVolume")) is None and "Volume" in hist.columns:
            info["averageVolume"] = _safe_number(hist["Volume"].tail(60).mean())
        if _safe_number(info.get("fiftyTwoWeekLow")) is None or _safe_number(info.get("fiftyTwoWeekHigh")) is None:
            last_year = hist.loc[hist.index >= (hist.index.max() - timedelta(days=365))]
            if not last_year.empty and "AdjClose" in last_year.columns:
                info.setdefault("fiftyTwoWeekLow", _safe_number(last_year["AdjClose"].min()))
                info.setdefault("fiftyTwoWeekHigh", _safe_number(last_year["AdjClose"].max()))

    _render_snapshot_card(info)

    window_options = ["1M", "3M", "6M", "1Y", "3Y", "Max"]
    default_index = window_options.index(window) if window in window_options else 2
    view_window = st.selectbox(
        "Ventana histórica",
        window_options,
        index=default_index,
    )
    view_start = _start_date_for_window(view_window)

    def _as_dt(val: Optional[str]) -> Optional[datetime]:
        if not val:
            return None
        try:
            return datetime.strptime(val, "%Y-%m-%d")
        except Exception:
            return None

    base_dt = _as_dt(base_start)
    view_dt = _as_dt(view_start)

    if view_dt and base_dt and view_dt < base_dt:
        # Re-fetch with a wider window to honor the user's selection.
        hist = fetch_history(ticker, view_start)

    hist_window = _filter_history(hist, view_window)

    tabs = st.tabs([
        "Precio",
        "Volumen y volatilidad",
        "Ratios",
        "Benchmark",
    ])

    with tabs[0]:
        _render_price_tab(ticker, hist_window)
    with tabs[1]:
        _render_volume_volatility(hist_window)
    with tabs[2]:
        _render_ratios(info)
    with tabs[3]:
        _render_benchmark_compare(ticker, hist_window, _start_date_for_window(view_window))

