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
        "resume_text": "",
        "resume_top_k": 5,
        "resume_match_result": None,
        "resume_uploaded_name": None,
        "resume_parse_meta": None,
        "resume_selected_job_id": None,
        "resume_selected_job_detail": None,
        "resume_selected_similar_jobs": [],
        "search_expanded_job_id": None,
        "search_expanded_job_detail": None,
        "search_expanded_similar_jobs": [],
        "resume_expanded_job_id": None,
        "resume_expanded_job_detail": None,
        "resume_expanded_similar_jobs": [],



    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value