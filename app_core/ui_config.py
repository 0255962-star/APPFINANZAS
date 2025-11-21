"""UI config helpers: setup_ui() and safe_rerun()."""

from __future__ import annotations

import logging

import pandas as pd

import streamlit as st
import yfinance as yf


def setup_ui() -> None:
    """Apply Streamlit page config, CSS tweaks, and quiet noisy logs."""
    st.set_page_config(
        page_title="APP Finanzas – Portafolio Activo",
        page_icon="AF",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --color-bg: #0c1220;
            --color-panel: #111a2c;
            --color-surface: #162036;
            --color-primary: #2b6cb0;
            --color-primary-strong: #3b82f6;
            --color-text: #e5e7eb;
            --color-muted: #9ca3af;
            --color-border: #1f2a3d;
            --color-success: #3fb27f;
            --color-danger: #ef4444;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background-color: var(--color-bg);
            color: var(--color-text);
        }
        .block-container {padding-top: 1rem;}
        section[data-testid="stSidebar"]{
            border-right:1px solid var(--color-border);
            background-color:#0a0f1d;
        }
        h1, h2, h3, h4 {color: var(--color-text);} 
        p, label, span, .stMarkdown {color: var(--color-text);} 
        div[data-testid="stMetricValue"]{font-size:1.4rem;}
        div[data-testid="stMetricLabel"]{
            font-size:0.95rem;
            color: var(--color-muted);
            white-space: normal;
            line-height: 1.3;
        }
        div[data-testid="stMetric"]{
            min-width: 180px;
            background: var(--color-panel);
            border:1px solid var(--color-border);
            border-radius:10px;
            padding:0.85rem 0.9rem;
            margin-bottom:0.85rem;
        }
        .stButton>button {
            background: var(--color-primary);
            color: var(--color-text);
            border:1px solid var(--color-primary);
            border-radius:8px;
        }
        .stButton>button:hover {
            border-color: var(--color-primary-strong);
            background: #2f5f94;
        }
        .stButton>button:active {
            border-color: var(--color-primary-strong);
            background: #274a73;
        }
        .stDataFrame table {
            color: var(--color-text);
            width: 100%;
        }
        .stDataFrame th {
            background: var(--color-surface);
            color: var(--color-text);
            border-bottom: 1px solid var(--color-border);
            white-space: nowrap;
        }
        .stDataFrame th:last-child, .stDataFrame td:last-child {
            width: 80px !important;
            text-align: center;
        }
        .stDataFrame td {
            background: var(--color-panel);
            color: var(--color-text);
            font-size: 0.95rem;
            white-space: nowrap;
            padding: 6px 10px;
        }
        .stDataFrame tbody tr:nth-child(even) td {background: #0f182a;}
        .stDataFrame tbody tr:hover td {background: #1b2640;}
        div[data-baseweb="tab-list"] button {
            color: var(--color-text);
        }
        div[data-baseweb="tab-highlight"] {
            background: var(--color-primary);
        }
        td:has(input[type="checkbox"]){text-align:center}
        </style>
        """,
        unsafe_allow_html=True,
    )
    try:
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        yf.utils.get_yf_logger().setLevel(logging.CRITICAL)
    except Exception:
        pass


def safe_rerun() -> None:
    """Trigger a Streamlit rerun without crashing older versions."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


def style_signed_numbers(df: pd.DataFrame, columns: list[str]) -> pd.io.formats.style.Styler | pd.DataFrame:
    """Return a Styler with green/red text for signed numeric columns.

    If the input is not a DataFrame or no requested columns exist, the input
    is returned unchanged to avoid impacting calling logic.
    """

    if df is None or not isinstance(df, pd.DataFrame):
        return df
    subset = [c for c in columns if c in df.columns]
    if not subset:
        return df

    def _signed_style(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: var(--color-success); font-weight: 600;"
        if val < 0:
            return "color: var(--color-danger); font-weight: 600;"
        return "color: var(--color-text);"

    return df.style.applymap(_signed_style, subset=subset)

