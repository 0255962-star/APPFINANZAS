"""UI config helpers: setup_ui() and safe_rerun()."""

from __future__ import annotations

import logging

import streamlit as st
import yfinance as yf


def setup_ui() -> None:
    """Apply Streamlit page config, CSS tweaks, and quiet noisy logs."""
    st.set_page_config(
        page_title="APP Finanzas – Portafolio Activo",
        page_icon="💼",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container{padding-top:.8rem}
        section[data-testid="stSidebar"]{border-right:1px solid #1e2435}
        div[data-testid="stMetricValue"]{font-size:1.6rem}
        th:has(> div:contains("➖")){color:#ff4d4f !important; width:48px !important}
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

