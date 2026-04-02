from __future__ import annotations

import streamlit as st

from src.ui.api_client import get_analytics_summary


def _to_chart_rows(items: list[dict]) -> list[dict]:
    return [{"name": x["name"], "count": x["count"]} for x in items]


def render_analytics_page() -> None:
    st.subheader("Analytics")

    data = get_analytics_summary(limit=10)

    c1, c2 = st.columns(2)
    c1.metric("Total Jobs", data.get("total_jobs", 0))
    c2.metric("Top Skills Shown", len(data.get("top_skills", [])))

    left, right = st.columns(2)

    with left:
        st.markdown("### Top Skills")
        skills = _to_chart_rows(data.get("top_skills", []))
        if skills:
            st.bar_chart(skills, x="name", y="count")
            st.dataframe(skills, use_container_width=True)
        else:
            st.info("No skills data available.")

        st.markdown("### Top Titles")
        titles = _to_chart_rows(data.get("top_titles", []))
        if titles:
            st.bar_chart(titles, x="name", y="count")
            st.dataframe(titles, use_container_width=True)
        else:
            st.info("No title data available.")

    with right:
        st.markdown("### Top Companies")
        companies = _to_chart_rows(data.get("top_companies", []))
        if companies:
            st.bar_chart(companies, x="name", y="count")
            st.dataframe(companies, use_container_width=True)
        else:
            st.info("No company data available.")

        st.markdown("### Top Locations")
        locations = _to_chart_rows(data.get("top_locations", []))
        if locations:
            st.bar_chart(locations, x="name", y="count")
            st.dataframe(locations, use_container_width=True)
        else:
            st.info("No location data available.")
