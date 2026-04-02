from __future__ import annotations

import streamlit as st


def render_header() -> None:
    st.title("JobPulse")
    st.caption("Semantic job search, structured extraction, and pipeline observability")


def render_metric_cards(
    indexed_jobs=None,
    runs_considered=None,
    avg_elapsed_sec=None,
    dq_slo_pass_rate=None,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indexed Jobs", indexed_jobs if indexed_jobs is not None else "—")
    c2.metric("Runs", runs_considered if runs_considered is not None else "—")
    c3.metric(
        "Avg Elapsed (s)",
        f"{avg_elapsed_sec:.2f}" if isinstance(avg_elapsed_sec, (int, float)) else "—",
    )
    c4.metric(
        "DQ SLO Pass",
        f"{dq_slo_pass_rate:.1%}" if isinstance(dq_slo_pass_rate, (int, float)) else "—",
    )


def render_result_card(job: dict) -> tuple[bool, bool]:
    job_id = str(job.get("job_id"))
    with st.container(border=True):
        st.markdown(f"### {job.get('title') or '(Untitled)'}")
        st.caption(f"{job.get('company') or 'N/A'} · {job.get('location') or 'N/A'}")

        score = job.get("score", 0.0)
        st.write(f"Semantic score: {score:.4f}")

        if job.get("url"):
            st.markdown(f"[Open job posting]({job['url']})")

        c1, c2 = st.columns(2)
        show_details = c1.button("Details", key=f"detail_{job_id}", use_container_width=True)
        show_similar = c2.button("Similar", key=f"similar_{job_id}", use_container_width=True)
        return show_details, show_similar


def render_job_detail(detail: dict | None) -> None:
    st.subheader("Job Details")

    if not detail:
        st.info("Select a job to view details.")
        return

    c1, c2 = st.columns(2)
    c1.write(f"**Title:** {detail.get('title') or 'N/A'}")
    c2.write(f"**Company:** {detail.get('company') or 'N/A'}")

    c3, c4 = st.columns(2)
    c3.write(f"**Location:** {detail.get('location_text') or 'N/A'}")
    c4.write(f"**Scrape Status:** {detail.get('scrape_status') or 'N/A'}")

    if detail.get("url"):
        st.markdown(f"[Open job posting]({detail['url']})")

    tabs = st.tabs(["Structured", "Skills", "Description"])

    with tabs[0]:
        st.json(detail.get("structured") or {})

    with tabs[1]:
        skills = detail.get("skills") or []
        if skills:
            st.write(skills)
        else:
            st.info("No skills found.")

    with tabs[2]:
        st.text_area(
            "Description",
            value=detail.get("description") or "",
            height=320,
            disabled=True,
            label_visibility="collapsed",
        )


def render_similar_jobs(similar_jobs: list[dict]) -> None:
    st.subheader("Similar Jobs")
    if not similar_jobs:
        st.info("No similar jobs loaded.")
        return

    for job in similar_jobs:
        with st.container(border=True):
            st.markdown(f"**{job.get('title') or '(Untitled)'}**")
            st.caption(f"{job.get('company') or 'N/A'} · {job.get('location') or 'N/A'}")
            st.write(f"Score: {job.get('score', 0.0):.4f}")
            if job.get("url"):
                st.markdown(f"[Open job posting]({job['url']})")


def render_match_card(job: dict) -> tuple[bool, bool]:
    job_id = str(job.get("job_id"))
    with st.container(border=True):
        st.markdown(f"### {job.get('title') or '(Untitled)'}")
        st.caption(f"{job.get('company') or 'N/A'} · {job.get('location') or 'N/A'}")
        st.write(f"Semantic score: {job.get('semantic_score', 0.0):.4f}")

        shared = job.get("shared_skills") or []
        missing = job.get("missing_skills") or []

        st.write("**Shared skills:**")
        st.write(shared if shared else ["—"])

        st.write("**Missing skills:**")
        st.write(missing if missing else ["—"])

        reasons = job.get("match_reasons") or []
        if reasons:
            st.write("**Why matched:**")
            for reason in reasons:
                st.write(f"- {reason}")

        c1, c2 = st.columns(2)
        show_details = c1.button(
            "View Job", key=f"resume_detail_{job_id}", use_container_width=True
        )
        show_similar = c2.button(
            "Similar Jobs", key=f"resume_similar_{job_id}", use_container_width=True
        )
        return show_details, show_similar


def render_clickable_similar_jobs(similar_jobs: list[dict], key_prefix: str = "sim") -> str | None:
    st.subheader("Similar Jobs")

    if not similar_jobs:
        st.info("No similar jobs loaded.")
        return None

    selected_job_id = None

    for job in similar_jobs:
        job_id = str(job.get("job_id"))
        with st.container(border=True):
            st.markdown(f"**{job.get('title') or '(Untitled)'}**")
            st.caption(f"{job.get('company') or 'N/A'} · {job.get('location') or 'N/A'}")
            st.write(f"Score: {job.get('score', 0.0):.4f}")

            c1, c2 = st.columns([1, 1])
            if c1.button(
                "Open Details", key=f"{key_prefix}_open_{job_id}", use_container_width=True
            ):
                selected_job_id = job_id
            if job.get("url"):
                c2.markdown(f"[Open posting]({job['url']})")

    return selected_job_id


def render_inline_job_detail(detail: dict | None) -> None:
    if not detail:
        st.info("No job detail loaded.")
        return

    c1, c2 = st.columns(2)
    c1.write(f"**Title:** {detail.get('title') or 'N/A'}")
    c2.write(f"**Company:** {detail.get('company') or 'N/A'}")

    c3, c4 = st.columns(2)
    c3.write(f"**Location:** {detail.get('location_text') or 'N/A'}")
    c4.write(f"**Scrape Status:** {detail.get('scrape_status') or 'N/A'}")

    if detail.get("url"):
        st.markdown(f"[Open job posting]({detail['url']})")

    tabs = st.tabs(["Structured", "Skills", "Description"])

    with tabs[0]:
        st.json(detail.get("structured") or {})

    with tabs[1]:
        skills = detail.get("skills") or []
        if skills:
            st.write(skills)
        else:
            st.info("No skills found.")

    with tabs[2]:
        st.text_area(
            "Description",
            value=detail.get("description") or "",
            height=260,
            disabled=True,
            label_visibility="collapsed",
            key=f"desc_{detail.get('job_id')}",
        )


def render_inline_similar_jobs(similar_jobs: list[dict], key_prefix: str = "sim") -> str | None:
    if not similar_jobs:
        st.info("No similar jobs loaded.")
        return None

    selected_job_id = None

    for job in similar_jobs:
        job_id = str(job.get("job_id"))
        with st.container(border=True):
            st.markdown(f"**{job.get('title') or '(Untitled)'}**")
            st.caption(f"{job.get('company') or 'N/A'} · {job.get('location') or 'N/A'}")
            st.write(f"Score: {job.get('score', 0.0):.4f}")

            c1, c2 = st.columns([1, 1])
            if c1.button(
                "Open Details", key=f"{key_prefix}_open_{job_id}", use_container_width=True
            ):
                selected_job_id = job_id
            if job.get("url"):
                c2.markdown(f"[Open posting]({job['url']})")

    return selected_job_id
