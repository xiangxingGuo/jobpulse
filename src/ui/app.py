from __future__ import annotations

import streamlit as st

from src.ui.components import render_header
from src.ui.state import init_state
from src.ui.views.overview import render_overview_page
from src.ui.views.search import render_search_page
from src.ui.views.analytics import render_analytics_page
from src.ui.views.pipeline import render_pipeline_page
from src.ui.views.resume_match import render_resume_match_page


st.set_page_config(page_title="JobPulse", layout="wide")

init_state()
render_header()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Search", "Analytics", "Pipeline", "Resume Match"],
    index=["Overview", "Search", "Analytics", "Pipeline", "Resume Match"].index(st.session_state.page)
)

st.session_state.page = page

if page == "Overview":
    render_overview_page()
elif page == "Search":
    render_search_page()
elif page == "Analytics":
    render_analytics_page()
elif page == "Pipeline":
    render_pipeline_page()
elif page == "Resume Match":
    render_resume_match_page()