from __future__ import annotations

import streamlit as st

from src.ui.api_client import get_analytics_summary, get_metrics
from src.ui.components import render_metric_cards


def render_overview_page() -> None:
    st.subheader("Overview")

    metrics = get_metrics(limit=20)
    analytics = get_analytics_summary(limit=8)

    scrape = metrics.get("scrape") or {}
    counts = metrics.get("counts") or {}

    render_metric_cards(
        indexed_jobs=analytics.get("total_jobs"),
        runs_considered=metrics.get("runs_considered"),
        avg_elapsed_sec=scrape.get("avg_elapsed_sec"),
        dq_slo_pass_rate=scrape.get("dq_slo_pass_rate"),
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Top Skills")
        skills = analytics.get("top_skills") or []
        if skills:
            st.bar_chart(skills, x="name", y="count")
        else:
            st.info("No skill data available.")

        st.markdown("### Top Companies")
        companies = analytics.get("top_companies") or []
        if companies:
            st.bar_chart(companies, x="name", y="count")
        else:
            st.info("No company data available.")

    with right:
        st.markdown("### Recent Runs")
        st.dataframe(metrics.get("latest_runs") or [], use_container_width=True)

        st.markdown("### Aggregated Counts")
        st.json(counts)