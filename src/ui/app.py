from __future__ import annotations

import streamlit as st

from src.ui.components import render_header
from src.ui.state import init_state
from src.ui.pages.overview import render_overview_page
from src.ui.pages.search import render_search_page
from src.ui.pages.pipeline import render_pipeline_page
from src.ui.pages.analytics import render_analytics_page

st.set_page_config(page_title="JobPulse", layout="wide")

init_state()
render_header()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Search", "Analytics", "Pipeline"],
    index=["Overview", "Search", "Analytics", "Pipeline"].index(st.session_state.page)
)

st.session_state.page = page

if page == "Overview":
    render_overview_page()
elif page == "Search":
    render_search_page()
elif page == "Analytics":
    render_analytics_page()
else:
    render_pipeline_page()