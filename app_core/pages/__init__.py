"""Page renderers for the Streamlit app."""

from .portfolio import render_portfolio_page  # noqa: F401
from .candidate import render_candidate_page  # noqa: F401
from .placeholder import render_placeholder_page  # noqa: F401

__all__ = [
    "render_portfolio_page",
    "render_candidate_page",
    "render_placeholder_page",
]

