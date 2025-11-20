"""Google Sheets access: get_gspread_client(), open_ws(), read_sheet()."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import gspread

from .secrets import build_credentials, load_secrets


@st.cache_resource(show_spinner=False)
def get_gspread_client() -> gspread.Client:
    """Authorize and cache a gspread client."""
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = build_credentials(scopes)
    return gspread.authorize(credentials)


def open_ws(name: str) -> gspread.Worksheet:
    """Open a worksheet by name using the cached client."""
    secrets = load_secrets()
    client = get_gspread_client()
    return client.open_by_key(secrets["sheet_id"]).worksheet(name)


@st.cache_data(ttl=1200, show_spinner=False)
def read_sheet(name: str) -> pd.DataFrame:
    """Read an entire worksheet into a DataFrame with unformatted values."""
    worksheet = open_ws(name)
    return pd.DataFrame(
        worksheet.get_all_records(value_render_option="UNFORMATTED_VALUE")
    )

