"""Streamlit entry-point orchestrating sidebar + page routing."""

from __future__ import annotations

import streamlit as st

from app_core.pages import (
    render_candidate_page,
    render_placeholder_page,
    render_portfolio_page,
)
from app_core.ui_config import setup_ui


def main() -> None:
    setup_ui()
    st.sidebar.title("📊 Navegación")
    page = st.sidebar.radio(
        "Ir a:",
        [
            "Mi Portafolio",
            "Optimizar y Rebalancear",
            "Evaluar Candidato",
            "Explorar / Research",
            "Diagnóstico",
        ],
    )
    window = st.sidebar.selectbox(
        "Ventana histórica", ["6M", "1Y", "3Y", "5Y", "Max"], index=2
    )

    if page == "Mi Portafolio":
        render_portfolio_page(window)
    elif page == "Evaluar Candidato":
        render_candidate_page(window)
    else:
        render_placeholder_page(page)


if __name__ == "__main__":
    main()

