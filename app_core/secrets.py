"""Secrets helpers: load_secrets() and build_credentials()."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import streamlit as st
from google.oauth2.service_account import Credentials


def load_secrets() -> Dict[str, Any]:
    """Return validated secrets needed by the application."""
    sheet_id = st.secrets.get("SHEET_ID") or st.secrets.get(
        "GSHEETS_SPREADSHEET_NAME", ""
    )
    gcp_sa = st.secrets.get("gcp_service_account", {})
    if not sheet_id:
        st.error("Falta `SHEET_ID` en secrets.")
        st.stop()
    if not gcp_sa:
        st.error("Falta `gcp_service_account` en secrets.")
        st.stop()
    return {"sheet_id": sheet_id, "gcp_service_account": gcp_sa}


def build_credentials(scopes: Iterable[str]) -> Credentials:
    """Construct service-account credentials using the provided OAuth scopes."""
    secrets = load_secrets()
    return Credentials.from_service_account_info(
        secrets["gcp_service_account"], scopes=list(scopes)
    )

