"""Streamlit entry-point orchestrating sidebar + page routing."""

from __future__ import annotations

import importlib
import traceback

import streamlit as st


def _import_page(module_name: str, attr: str):
    """Import a page module defensively and surface clear diagnostics."""

    try:
        module = importlib.import_module(module_name)
    except SyntaxError as exc:  # pragma: no cover - syntax issues must be corrected
        full_tb = traceback.format_exc()
        st.error(
            "Hay un error de sintaxis en los módulos de páginas. Corrige el archivo "
            f"{exc.filename} en la línea {exc.lineno} y reinicia la app."
        )
        st.info(f"Detalle técnico: {exc.msg}")
        st.caption(full_tb)
        st.stop()
    except Exception as exc:  # pragma: no cover - keep startup resilient
        traceback.print_exc()
        st.error(
            "No pude cargar las páginas de la app. Revisa el mensaje técnico y los "
            "logs para más detalle."
        )
        st.info(f"Detalle técnico: {exc}")
        st.stop()

    try:
        return getattr(module, attr)
    except AttributeError:  # pragma: no cover - API contract enforcement
        st.error(
            f"El módulo {module_name} no expone {attr}. Revisa el código y vuelve a probar."
        )
        st.stop()


render_candidate_page = _import_page("app_core.pages.candidate", "render_candidate_page")
render_placeholder_page = _import_page("app_core.pages.placeholder", "render_placeholder_page")
render_portfolio_page = _import_page("app_core.pages.portfolio", "render_portfolio_page")
render_scenario_page = _import_page("app_core.pages.scenario", "render_scenario_page")
render_research_page = _import_page("app_core.pages.research", "render_research_page")
from app_core.ui_config import setup_ui


def main() -> None:
    setup_ui()
    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Ir a:",
        [
            "Mi Portafolio",
            "Simulador avanzado de escenarios",
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
    elif page == "Simulador avanzado de escenarios":
        render_scenario_page(window)
    elif page == "Evaluar Candidato":
        render_candidate_page(window)
    elif page == "Explorar / Research":
        render_research_page(window)
    else:
        render_placeholder_page(page)


if __name__ == "__main__":
    main()

