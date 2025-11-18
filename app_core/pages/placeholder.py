"""Placeholder page renderer: render_placeholder_page(title)."""

from __future__ import annotations

import streamlit as st


def render_placeholder_page(title: str) -> None:
    """Render static placeholder content for unfinished tabs."""
    st.title(title)
    st.info("Contenido no modificado. Esta sección mantiene el mismo formato de la app original.")

