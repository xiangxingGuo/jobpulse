from __future__ import annotations

import streamlit as st

from src.ui.api_client import job_market_chat


def render_job_market_chat_page() -> None:
    st.subheader("Job Market Chat")

    st.markdown(
        "Ask grounded questions about job requirements, prioritization, market trends, "
        "or how your background aligns with a target role."
    )

    default_resume = st.session_state.get("resume_text", "")
    default_job_id = st.session_state.get("resume_fit_job_id", "")

    question = st.text_area(
        "Question",
        value=st.session_state.get(
            "job_market_chat_question",
            "Based on my background, which kinds of ML/AI roles should I prioritize?"
        ),
        height=120,
        placeholder="Ask about role fit, market trends, target jobs, or next steps.",
    )

    c1, c2, c3 = st.columns(3)

    top_k = c1.slider(
        "Top K retrieved jobs",
        min_value=1,
        max_value=10,
        value=st.session_state.get("job_market_chat_top_k", 5),
    )

    provider = c2.selectbox(
        "LLM provider",
        options=["openai", "nvidia"],
        index=0 if st.session_state.get("job_market_chat_provider", "openai") == "openai" else 1,
    )

    model = c3.text_input(
        "Model override (optional)",
        value=st.session_state.get("job_market_chat_model", ""),
    )

    use_resume = st.checkbox(
        "Use current resume context",
        value=True,
        help="Uses resume text from the Resume Match page if available.",
    )

    use_target_job = st.checkbox(
        "Use selected target job",
        value=bool(default_job_id),
        help="Uses the currently selected/analyzed job if available.",
    )

    resume_text = default_resume if use_resume else ""
    job_id = default_job_id if use_target_job else ""

    with st.expander("Context Preview", expanded=False):
        st.write("**Resume available:**", bool((resume_text or "").strip()))
        st.write("**Target job id:**", job_id or "(none)")
        if resume_text:
            st.caption((resume_text[:500] + "...") if len(resume_text) > 500 else resume_text)

    if st.button("Run Job Market Chat", use_container_width=True):
        cleaned_question = (question or "").strip()
        if len(cleaned_question) < 3:
            st.error("Please enter a longer question.")
            return

        st.session_state.job_market_chat_question = cleaned_question
        st.session_state.job_market_chat_top_k = top_k
        st.session_state.job_market_chat_provider = provider
        st.session_state.job_market_chat_model = model

        with st.spinner("Generating grounded answer..."):
            try:
                out = job_market_chat(
                    question=cleaned_question,
                    top_k=top_k,
                    resume_text=(resume_text or None),
                    job_id=(job_id or None),
                    provider=provider,
                    model=(model.strip() or None),
                )
                st.session_state.job_market_chat_result = out
            except Exception as e:
                st.error(f"Job market chat failed: {e}")
                return

    result = st.session_state.get("job_market_chat_result")
    if not result:
        st.info("Ask a question to start.")
        return

    answer = result.get("answer") or ""
    sources = result.get("sources") or []
    meta = result.get("meta") or {}

    st.markdown("### Answer")
    st.write(answer)

    if sources:
        st.markdown("### Sources")
        for src in sources:
            title = src.get("title") or "(untitled)"
            company = src.get("company") or ""
            reason = src.get("reason") or ""
            job_id_val = src.get("job_id") or ""
            st.markdown(
                f"- **{title}**"
                f"{' · ' + company if company else ''}"
                f"{' · job_id=' + job_id_val if job_id_val else ''}"
                f"{' — ' + reason if reason else ''}"
            )

    if meta:
        with st.expander("Chat Meta", expanded=False):
            st.json(meta)