from __future__ import annotations

import streamlit as st

from src.ui.api_client import (
    analyze_resume_fit,
    get_job,
    get_similar_jobs,
    match_resume,
    parse_resume_file,
)
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
        value=st.session_state.get("resume_text", ""),
        height=220,
        placeholder="Paste your resume text here, or upload a file above.",
    )

    top_k = st.slider(
        "Top K matches",
        min_value=1,
        max_value=10,
        value=st.session_state.get("resume_top_k", 5),
    )

    st.markdown("### Analysis Settings")

    c1, c2, c3 = st.columns(3)

    analysis_mode = c1.selectbox(
        "Analysis mode",
        options=["baseline", "hybrid"],
        index=0 if st.session_state.get("resume_analysis_mode", "baseline") == "baseline" else 1,
        help="Baseline uses deterministic overlap logic. Hybrid adds LLM reasoning on top.",
    )

    provider = c2.selectbox(
        "LLM provider",
        options=["openai", "nvidia"],
        index=0 if st.session_state.get("resume_analysis_provider", "openai") == "openai" else 1,
        help="Used only when analysis mode is hybrid or report generation needs an LLM.",
    )

    model = c3.text_input(
        "Model override (optional)",
        value=st.session_state.get("resume_analysis_model", ""),
        help="Leave blank to use the provider default model.",
    )

    include_report = st.checkbox(
        "Generate markdown report",
        value=st.session_state.get("resume_include_report", True),
        help="If disabled, only structured analysis is returned.",
    )

    if st.button("Run Resume Match", use_container_width=True):
        cleaned_resume_text = (resume_text or "").strip()

        if len(cleaned_resume_text) < 20:
            st.error("Resume text is too short. Please paste more content or parse a valid file first.")
        else:
            data = match_resume(resume_text=cleaned_resume_text, top_k=top_k)
            st.session_state.resume_text = cleaned_resume_text
            st.session_state.resume_top_k = top_k
            st.session_state.resume_analysis_mode = analysis_mode
            st.session_state.resume_analysis_provider = provider
            st.session_state.resume_analysis_model = model
            st.session_state.resume_include_report = include_report
            st.session_state.resume_match_result = data
            st.session_state.resume_expanded_job_id = None
            st.session_state.resume_expanded_job_detail = None
            st.session_state.resume_expanded_similar_jobs = []
            st.session_state.resume_fit_analysis = None
            st.session_state.resume_fit_job_id = None
            st.rerun()

    parse_meta = st.session_state.get("resume_parse_meta")
    if parse_meta:
        with st.expander("Parsed Resume", expanded=False):
            st.write(f"**File:** {parse_meta.get('filename')}")
            st.write(f"**Characters:** {parse_meta.get('chars')}")
            st.caption(parse_meta.get("text_preview") or "")

    result = st.session_state.get("resume_match_result")
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

        analyze_clicked = st.button(
            f"Analyze Fit for {job_id}",
            key=f"analyze_fit_{job_id}",
            use_container_width=True,
        )

        if analyze_clicked:
            cleaned_resume_text = (st.session_state.get("resume_text") or "").strip()
            if len(cleaned_resume_text) < 20:
                st.error("Resume text is too short. Please paste more content or parse a valid file first.")
            else:
                with st.spinner("Running skill-gap analysis..."):
                    try:
                        analysis = analyze_resume_fit(
                            resume_text=cleaned_resume_text,
                            job_id=job_id,
                            include_market_context=True,
                            market_top_k=5,
                            include_report=include_report,
                            analysis_mode=analysis_mode,
                            provider=provider,
                            model=(model.strip() or None),
                        )
                        st.session_state.resume_fit_analysis = analysis
                        st.session_state.resume_fit_job_id = job_id
                    except Exception as e:
                        st.error(f"Analyze fit failed: {e}")

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

        if st.session_state.get("resume_expanded_job_id") == job_id:
            st.markdown("#### Matched Job Details")
            render_inline_job_detail(st.session_state.get("resume_expanded_job_detail"))

            st.markdown("#### Similar Jobs")
            selected_similar_job_id = render_inline_similar_jobs(
                st.session_state.get("resume_expanded_similar_jobs", []),
                key_prefix=f"resume_similar_{job_id}",
            )

            if selected_similar_job_id:
                detail = get_job(selected_similar_job_id)
                st.session_state.resume_expanded_job_id = selected_similar_job_id
                st.session_state.resume_expanded_job_detail = detail
                st.session_state.resume_expanded_similar_jobs = []
                st.rerun()

            st.markdown("---")

        if st.session_state.get("resume_fit_job_id") == job_id:
            analysis = st.session_state.get("resume_fit_analysis") or {}
            _render_skill_gap_analysis(analysis)


def _render_skill_gap_analysis(analysis: dict) -> None:
    skill_gap = analysis.get("skill_gap") or {}
    resume_profile = analysis.get("resume_profile") or {}
    report_md = analysis.get("report_md") or ""
    meta = analysis.get("meta") or {}
    llm_meta = ((skill_gap.get("meta") or {}).get("llm") or {})

    analysis_mode = meta.get("analysis_mode") or ((skill_gap.get("meta") or {}).get("analysis_mode"))
    provider = meta.get("provider")
    model = meta.get("model")

    st.markdown("#### Skill Gap Analysis")

    fit_score = skill_gap.get("fit_score")
    fit_band = skill_gap.get("fit_band")
    confidence = skill_gap.get("confidence")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fit Score", fit_score if fit_score is not None else "-")
    c2.metric("Fit Band", fit_band or "-")
    c3.metric("Confidence", confidence if confidence is not None else "-")

    badges = []
    if analysis_mode:
        badges.append(f"mode={analysis_mode}")
    if provider:
        badges.append(f"provider={provider}")
    if model:
        badges.append(f"model={model}")

    if badges:
        st.caption(" | ".join(badges))
    
    summary = skill_gap.get("summary")
    if summary:
        st.info(summary)

    with st.expander("Resume Profile Used for Analysis", expanded=False):
        st.write("**Explicit Skills**")
        st.write(resume_profile.get("explicit_skills") or [])
        st.write("**ML Domains**")
        st.write(resume_profile.get("ml_domains") or [])
        st.write("**Deployment Signals**")
        st.write(resume_profile.get("deployment_signals") or [])

    strengths = skill_gap.get("strengths") or []
    if strengths:
        st.markdown("**Evidence-backed Strengths**")
        for item in strengths:
            st.markdown(f"- **{item.get('skill')}** ({item.get('support')}) — {item.get('rationale')}")
            for ev in item.get("evidence") or []:
                st.caption(f"[{ev.get('source')}] {ev.get('snippet')}")

    gaps = skill_gap.get("gaps") or []
    if gaps:
        st.markdown("**Main Gaps**")
        for item in gaps:
            st.markdown(
                f"- **{item.get('skill')}** "
                f"({item.get('category')}, severity={item.get('severity')}) — "
                f"{item.get('rationale')}"
            )
            for ev in item.get("evidence") or []:
                st.caption(f"[{ev.get('source')}] {ev.get('snippet')}")

    transferable = skill_gap.get("transferable_signals") or []
    if transferable:
        st.markdown("**Transferable Signals**")
        for item in transferable:
            st.markdown(f"- **{item.get('skill')}** — {item.get('rationale')}")
            for ev in item.get("evidence") or []:
                st.caption(f"[{ev.get('source')}] {ev.get('snippet')}")

    suggestions = skill_gap.get("resume_suggestions") or []
    if suggestions:
        st.markdown("**Resume Tailoring Suggestions**")
        for item in suggestions:
            st.markdown(
                f"- **{item.get('type')}** · {item.get('target')} — {item.get('rationale')}"
            )

    action_7d = skill_gap.get("action_plan_7d") or []
    action_30d = skill_gap.get("action_plan_30d") or []

    if action_7d:
        st.markdown("**7-Day Action Plan**")
        for step in action_7d:
            st.markdown(f"- {step}")

    if action_30d:
        st.markdown("**30-Day Action Plan**")
        for step in action_30d:
            st.markdown(f"- {step}")

    if report_md:
        with st.expander("Generated Markdown Report", expanded=True):
            st.markdown(report_md)

    if meta:
        with st.expander("Analysis Meta", expanded=False):
            st.json(meta)
    
    if llm_meta:
        with st.expander("LLM Analysis Info", expanded=False):
            st.json(llm_meta)
    
    st.markdown("---")