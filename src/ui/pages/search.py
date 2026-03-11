from __future__ import annotations

import streamlit as st

from src.ui.api_client import get_job, get_similar_jobs, search_jobs
from src.ui.components import render_job_detail, render_result_card, render_similar_jobs


def render_search_page() -> None:
    st.subheader("Semantic Search")

    with st.sidebar:
        st.markdown("## Search Controls")

        query = st.text_input("Query", value=st.session_state.search_query)
        top_k = st.slider("Top K", min_value=1, max_value=20, value=st.session_state.search_top_k)

        if st.button("Run Search", use_container_width=True):
            data = search_jobs(query=query, top_k=top_k)
            st.session_state.search_query = query
            st.session_state.search_top_k = top_k
            st.session_state.search_results = data.get("results", [])
            st.session_state.selected_job_id = None
            st.session_state.selected_job_detail = None
            st.session_state.selected_similar_jobs = []
            st.rerun()

        if st.button("Clear Search", use_container_width=True):
            st.session_state.search_results = []
            st.session_state.selected_job_id = None
            st.session_state.selected_job_detail = None
            st.session_state.selected_similar_jobs = []
            st.rerun()

    results = st.session_state.search_results
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown(f"### Results ({len(results)})")
        for job in results:
            show_details, show_similar = render_result_card(job)
            job_id = str(job.get("job_id"))

            if show_details:
                detail = get_job(job_id)
                st.session_state.selected_job_id = job_id
                st.session_state.selected_job_detail = detail
                st.rerun()

            if show_similar:
                similar = get_similar_jobs(job_id, top_k=5)
                st.session_state.selected_job_id = job_id
                st.session_state.selected_similar_jobs = similar.get("results", [])
                if not st.session_state.selected_job_detail:
                    st.session_state.selected_job_detail = get_job(job_id)
                st.rerun()

    with right:
        render_job_detail(st.session_state.selected_job_detail)
        st.markdown("---")
        render_similar_jobs(st.session_state.selected_similar_jobs)