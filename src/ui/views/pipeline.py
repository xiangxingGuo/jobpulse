from __future__ import annotations

import streamlit as st

from src.ui.api_client import get_metrics


def render_pipeline_page() -> None:
    st.subheader("Pipeline Observability")

    metrics = get_metrics(limit=20)
    scrape = metrics.get("scrape") or {}
    latest_runs = metrics.get("latest_runs") or []
    counts = metrics.get("counts") or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs considered", metrics.get("runs_considered"))
    c2.metric(
        "Avg elapsed sec",
        f"{scrape['avg_elapsed_sec']:.2f}" if scrape.get("avg_elapsed_sec") is not None else "—",
    )
    c3.metric(
        "SLO pass rate",
        f"{scrape['slo_pass_rate']:.1%}" if scrape.get("slo_pass_rate") is not None else "—",
    )
    c4.metric(
        "DQ SLO pass rate",
        f"{scrape['dq_slo_pass_rate']:.1%}" if scrape.get("dq_slo_pass_rate") is not None else "—",
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Recent Run Durations")
        run_chart = [
            {
                "run_id": r.get("run_id"),
                "elapsed_sec": r.get("elapsed_sec") or 0,
            }
            for r in latest_runs
        ]
        if run_chart:
            st.bar_chart(run_chart, x="run_id", y="elapsed_sec")
        else:
            st.info("No run duration data available.")

        st.markdown("### Latest Runs")
        st.dataframe(latest_runs, use_container_width=True)

    with right:
        st.markdown("### Aggregated Counts")
        count_rows = [{"name": k, "count": v} for k, v in counts.items()]
        if count_rows:
            st.bar_chart(count_rows, x="name", y="count")
            st.dataframe(count_rows, use_container_width=True)
        else:
            st.info("No aggregated counts available.")