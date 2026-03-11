from __future__ import annotations

import streamlit as st

from src.ui.api_client import match_resume, parse_resume_file


def render_resume_match_page() -> None:
    st.subheader("Resume Match")

    left, right = st.columns([1, 1.2])

    with left:
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
            height=320,
            placeholder="Paste your resume text here, or upload a file above.",
        )

        st.caption(f"Current resume text length: {len((resume_text or '').strip())}")

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
                st.rerun()

    with right:
        parse_meta = st.session_state.resume_parse_meta
        if parse_meta:
            st.markdown("### Parsed Resume")
            st.write(f"**File:** {parse_meta.get('filename')}")
            st.write(f"**Characters:** {parse_meta.get('chars')}")
            st.caption(parse_meta.get("text_preview") or "")

        result = st.session_state.resume_match_result
        if not result:
            st.info("Upload or paste a resume, then run matching.")
            return

        profile = result.get("resume_profile") or {}
        matches = result.get("matches") or []

        st.markdown("### Extracted Resume Profile")
        st.write("**Skills:**")
        st.write(profile.get("skills") or [])

        st.markdown("### Top Matched Jobs")
        for m in matches:
            with st.container(border=True):
                st.markdown(f"### {m.get('title') or '(Untitled)'}")
                st.caption(f"{m.get('company') or 'N/A'} · {m.get('location') or 'N/A'}")
                st.write(f"Semantic score: {m.get('semantic_score', 0.0):.4f}")

                st.write("**Shared skills:**")
                st.write(m.get("shared_skills") or [])

                st.write("**Missing skills:**")
                st.write(m.get("missing_skills") or [])

                st.write("**Why matched:**")
                for reason in m.get("match_reasons") or []:
                    st.write(f"- {reason}")

                if m.get("url"):
                    st.markdown(f"[Open job posting]({m['url']})")
