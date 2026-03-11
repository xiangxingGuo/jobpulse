from __future__ import annotations

import streamlit as st

from src.ui.api_client import get_job, get_similar_jobs, match_resume, parse_resume_file
from src.ui.components import (
    render_inline_job_detail,
    render_inline_similar_jobs,
    render_match_card,
)


def render_resume_match_page() -> None:
    st.subheader("Resume Match")

    st.markdown("### Resume Input")

    uploaded = st.file_uploader(
        "Upload resume",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=False,
        help="Supported: TXT, PDF, DOCX",
    )

    if uploaded is not None:
        if st.button("Parse uploaded file", use_container_width=True):
            parsed = parse_resume_file(uploaded.name, uploaded.getvalue())
            st.session_state.resume_text = parsed.get("resume_text", "")
            st.session_state.resume_uploaded_name = uploaded.name
            st.session_state.resume_parse_meta = parsed
            st.success(f"Parsed {uploaded.name} successfully.")
            st.rerun()

    resume_text = st.text_area(
        "Resume text",
        value=st.session_state.resume_text,
        height=220,
        placeholder="Paste your resume text here, or upload a file above.",
    )

    top_k = st.slider(
        "Top K matches",
        min_value=1,
        max_value=10,
        value=st.session_state.resume_top_k,
    )

    if st.button("Run Resume Match", use_container_width=True):
        cleaned_resume_text = (resume_text or "").strip()

        if len(cleaned_resume_text) < 20:
            st.error("Resume text is too short. Please paste more content or parse a valid file first.")
        else:
            data = match_resume(resume_text=cleaned_resume_text, top_k=top_k)
            st.session_state.resume_text = cleaned_resume_text
            st.session_state.resume_top_k = top_k
            st.session_state.resume_match_result = data
            st.session_state.resume_expanded_job_id = None
            st.session_state.resume_expanded_job_detail = None
            st.session_state.resume_expanded_similar_jobs = []
            st.rerun()

    parse_meta = st.session_state.resume_parse_meta
    if parse_meta:
        with st.expander("Parsed Resume", expanded=False):
            st.write(f"**File:** {parse_meta.get('filename')}")
            st.write(f"**Characters:** {parse_meta.get('chars')}")
            st.caption(parse_meta.get("text_preview") or "")

    result = st.session_state.resume_match_result
    if not result:
        st.info("Upload or paste a resume, then run matching.")
        return

    profile = result.get("resume_profile") or {}
    matches = result.get("matches") or []

    with st.expander("Extracted Resume Profile", expanded=True):
        st.write("**Skills:**")
        st.write(profile.get("skills") or [])

    st.markdown("### Top Matched Jobs")

    for m in matches:
        show_details, show_similar = render_match_card(m)
        job_id = str(m.get("job_id"))

        if show_details:
            detail = get_job(job_id)
            st.session_state.resume_expanded_job_id = job_id
            st.session_state.resume_expanded_job_detail = detail
            st.session_state.resume_expanded_similar_jobs = []
            st.rerun()

        if show_similar:
            detail = get_job(job_id)
            similar = get_similar_jobs(job_id, top_k=5)
            st.session_state.resume_expanded_job_id = job_id
            st.session_state.resume_expanded_job_detail = detail
            st.session_state.resume_expanded_similar_jobs = similar.get("results", [])
            st.rerun()

        if st.session_state.resume_expanded_job_id == job_id:
            st.markdown("#### Matched Job Details")
            render_inline_job_detail(st.session_state.resume_expanded_job_detail)

            st.markdown("#### Similar Jobs")
            selected_similar_job_id = render_inline_similar_jobs(
                st.session_state.resume_expanded_similar_jobs,
                key_prefix=f"resume_similar_{job_id}",
            )

            if selected_similar_job_id:
                detail = get_job(selected_similar_job_id)
                st.session_state.resume_expanded_job_id = selected_similar_job_id
                st.session_state.resume_expanded_job_detail = detail
                st.session_state.resume_expanded_similar_jobs = []
                st.rerun()

            st.markdown("---")
