from __future__ import annotations

import streamlit as st


def init_state() -> None:
    defaults = {
        "page": "Overview",
        "search_query": "entry level machine learning engineer pytorch mlops",
        "search_top_k": 5,
        "search_results": [],
        "selected_job_id": None,
        "selected_job_detail": None,
        "selected_similar_jobs": [],
        "metrics_cache": None,
        "recent_runs_cache": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value