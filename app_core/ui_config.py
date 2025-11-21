"""UI config helpers: setup_ui() and safe_rerun()."""

from __future__ import annotations

import logging

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
        div[data-testid="stMetricValue"]{font-size:1.6rem}
        div[data-testid="stMetric"]{
            background: var(--color-panel);
            border:1px solid var(--color-border);
            border-radius:10px;
            padding:0.75rem;
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
        }
        .stDataFrame th {
            background: var(--color-surface);
            color: var(--color-text);
            border-bottom: 1px solid var(--color-border);
        }
        .stDataFrame th:last-child, .stDataFrame td:last-child {
            width: 80px !important;
            text-align: center;
        }
        .stDataFrame td {
            background: var(--color-panel);
            color: var(--color-text);
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

