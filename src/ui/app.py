from __future__ import annotations

import os
import requests
import streamlit as st


API_BASE = os.getenv("JOBPULSE_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="JobPulse", layout="wide")
st.title("JobPulse")
st.caption("Semantic job search + pipeline observability")


tab1, tab2 = st.tabs(["Search", "Metrics"])


with tab1:
    st.subheader("Semantic Job Search")

    query = st.text_input(
        "Search query",
        value="entry level machine learning engineer pytorch mlops",
    )
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)

    if st.button("Search"):
        resp = requests.post(
            f"{API_BASE}/jobs/search",
            json={"query": query, "top_k": top_k},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        st.write(f"Results: {len(data['results'])}")
        for r in data["results"]:
            with st.container(border=True):
                st.markdown(f"### {r.get('title') or '(Untitled)'}")
                st.write(f"**Company:** {r.get('company') or 'N/A'}")
                st.write(f"**Location:** {r.get('location') or 'N/A'}")
                st.write(f"**Score:** {r.get('score'):.4f}")
                if r.get("url"):
                    st.markdown(f"[Open job posting]({r['url']})")

                job_id = r.get("job_id")
                if st.button(f"Show details: {job_id}", key=f"detail_{job_id}"):
                    dresp = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=60)
                    dresp.raise_for_status()
                    detail = dresp.json()

                    st.markdown("#### Skills")
                    st.write(detail.get("skills") or [])

                    st.markdown("#### Structured")
                    st.json(detail.get("structured") or {})

                    st.markdown("#### Description")
                    st.text(detail.get("description") or "")


with tab2:
    st.subheader("Pipeline Metrics")

    if st.button("Refresh metrics"):
        pass

    mresp = requests.get(f"{API_BASE}/metrics/summary?limit=20", timeout=60)
    mresp.raise_for_status()
    metrics = mresp.json()

    scrape = metrics.get("scrape") or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs considered", metrics.get("runs_considered"))
    c2.metric("Avg elapsed sec", None if scrape.get("avg_elapsed_sec") is None else round(scrape["avg_elapsed_sec"], 2))
    c3.metric("SLO pass rate", None if scrape.get("slo_pass_rate") is None else f"{scrape['slo_pass_rate']:.1%}")
    c4.metric("DQ SLO pass rate", None if scrape.get("dq_slo_pass_rate") is None else f"{scrape['dq_slo_pass_rate']:.1%}")

    st.markdown("### Aggregated Counts")
    st.json(metrics.get("counts") or {})

    st.markdown("### Recent Runs")
    st.dataframe(metrics.get("latest_runs") or [], use_container_width=True)