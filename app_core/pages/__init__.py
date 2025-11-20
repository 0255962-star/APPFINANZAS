"""Page renderers for the Streamlit app."""

from .candidate import render_candidate_page  # noqa: F401
from .portfolio import render_portfolio_page  # noqa: F401
from .scenario import render_scenario_page  # noqa: F401
from .research import render_research_page  # noqa: F401
from .placeholder import render_placeholder_page  # noqa: F401

__all__ = [
    "render_portfolio_page",
    "render_candidate_page",
    "render_scenario_page",
    "render_research_page",
    "render_placeholder_page",
]

