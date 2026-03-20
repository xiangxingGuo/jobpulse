from __future__ import annotations

import streamlit as st
from src.ui.api_client import analyze_skill_gap_serverless


def render_lex_career_coach_page() -> None:
    st.subheader("AWS Career Coach (Lex + Lambda)")

    st.markdown(
        "This assistant is powered by an AWS-native conversational workflow "
        "using Amazon Lex, Lambda, API Gateway, and DynamoDB."
    )

    target_role = st.text_input(
        "Target Role",
        value=st.session_state.get("lex_target_role", "AI Engineer"),
    )

    experience_level = st.selectbox(
        "Experience Level",
        options=["student", "intern", "new grad", "junior", "career switcher"],
        index=2,
    )

    candidate_background = st.text_area(
        "Your Background",
        value=st.session_state.get(
            "lex_background",
            "I have Python, PyTorch, FastAPI, RAG project experience, and deployed an AI project on AWS EC2 with Docker."
        ),
        height=150,
    )

    if st.button("Analyze Skill Gap (AWS)", use_container_width=True):
        if len((candidate_background or "").strip()) < 10:
            st.error("Please provide a more detailed background.")
            return

        st.session_state.lex_target_role = target_role
        st.session_state.lex_background = candidate_background

        with st.spinner("Calling AWS Lambda via API Gateway..."):
            try:
                out = analyze_skill_gap_serverless(
                    target_role=target_role,
                    experience_level=experience_level,
                    candidate_background=candidate_background,
                )
                st.session_state.lex_result = out
            except Exception as e:
                st.error(f"AWS career coach failed: {e}")
                return

    result = st.session_state.get("lex_result")
    if not result:
        st.info("Fill in your profile and run the analysis.")
        return

    message = result.get("message") or ""
    sources = result.get("sources") or []

    st.markdown("### Career Recommendation")
    st.write(message)

    if sources:
        st.markdown("### Market Examples")
        for src in sources:
            title = src.get("title") or "(untitled)"
            company = src.get("company") or ""
            st.markdown(f"- **{title}**{' · ' + company if company else ''}")