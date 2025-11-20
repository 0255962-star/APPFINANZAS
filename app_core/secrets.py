"""Secrets helpers: load_secrets() and build_credentials()."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import streamlit as st
from google.oauth2.service_account import Credentials
from streamlit.runtime.scriptrunner import get_script_run_ctx


def _fail_secrets(msg: str):
    """Show a user-friendly error when secrets are missing and halt execution."""
    # If we are running inside Streamlit, present the message and stop cleanly
    # so the user sees the guidance without a Python traceback.
    if get_script_run_ctx() is not None:
        st.error(msg)
        st.stop()

    # Outside of a Streamlit run (e.g., tests or CLI), raise so callers fail fast.
    raise RuntimeError(msg)


def _fail_secrets(msg: str):
    """Show a user-friendly error when secrets are missing and halt execution."""
    try:
        st.error(msg)
        st.stop()
    except Exception:
        # When Streamlit context is unavailable (e.g., bare Python run), continue to raise.
        pass
    # Always raise so callers in any environment fail deterministically.
    raise RuntimeError(msg)
        # When Streamlit context is unavailable (e.g., bare Python run), raise a clear error.
        raise RuntimeError(msg)


def load_secrets() -> Dict[str, Any]:
    """Return validated secrets needed by the application."""
    try:
        sheet_id = st.secrets.get("SHEET_ID") or st.secrets.get(
            "GSHEETS_SPREADSHEET_NAME", ""
        )
        gcp_sa = st.secrets.get("gcp_service_account", {})
    except FileNotFoundError:
        _fail_secrets(
            "No se encontró secrets.toml. Coloca el archivo en .streamlit/secrets.toml "
            "con las claves SHEET_ID y gcp_service_account."
        )

    if not sheet_id:
        _fail_secrets("Falta `SHEET_ID` en secrets.")
    if not gcp_sa:
        _fail_secrets("Falta `gcp_service_account` en secrets.")
    return {"sheet_id": sheet_id, "gcp_service_account": gcp_sa}


def build_credentials(scopes: Iterable[str]) -> Credentials:
    """Construct service-account credentials using the provided OAuth scopes."""
    secrets = load_secrets()
    return Credentials.from_service_account_info(
        secrets["gcp_service_account"], scopes=list(scopes)
    )
