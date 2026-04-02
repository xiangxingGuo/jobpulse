from __future__ import annotations

import streamlit as st

from src.ui.api_client import (
    get_career_session,
    match_resume,
    parse_resume_file,
    send_career_message,
    start_career_session,
)


def _render_resume_match_results(result: dict) -> None:
    resume_profile = result.get("resume_profile", {}) or {}
    matches = result.get("matches", []) or []

    st.markdown("### Matching jobs from your resume")

    skills = resume_profile.get("skills", []) or []
    if skills:
        st.caption("Detected resume skills: " + ", ".join(skills[:15]))

    if not matches:
        st.info("No matching jobs found.")
        return

    for i, job in enumerate(matches, start=1):
        title = job.get("title") or "Untitled role"
        company = job.get("company") or "Unknown company"
        location = job.get("location") or "Unknown location"
        url = job.get("url")
        semantic_score = job.get("semantic_score")
        shared_skills = job.get("shared_skills", []) or []
        missing_skills = job.get("missing_skills", []) or []
        reasons = job.get("match_reasons", []) or []

        with st.container():
            st.markdown(f"**{i}. {title}**")
            st.write(f"**Company:** {company}")
            st.write(f"**Location:** {location}")

            if semantic_score is not None:
                st.write(f"**Match score:** {semantic_score:.3f}")

            if shared_skills:
                st.write("**Shared skills:** " + ", ".join(shared_skills[:10]))

            if missing_skills:
                st.write("**Missing skills:** " + ", ".join(missing_skills[:10]))

            if reasons:
                st.write("**Why this matches:**")
                for reason in reasons[:3]:
                    st.write(f"- {reason}")

            if url:
                st.markdown(f"[Open job posting]({url})")

            st.divider()


def render_lex_career_coach_page() -> None:
    st.subheader("AWS Career Coach")

    st.markdown(
        "This assistant uses a stateful backend workflow powered by "
        "API Gateway, Lambda, and DynamoDB."
    )

    st.markdown("### Optional: Upload Resume (for better analysis)")

    if "aws_resume_text" not in st.session_state:
        st.session_state["aws_resume_text"] = None

    if "aws_resume_filename" not in st.session_state:
        st.session_state["aws_resume_filename"] = None

    if "aws_resume_match_result" not in st.session_state:
        st.session_state["aws_resume_match_result"] = None

    if "aws_resume_match_error" not in st.session_state:
        st.session_state["aws_resume_match_error"] = None

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx", "txt"],
        help="Optional. The parsed resume text will be used as additional context during the conversation.",
    )

    if uploaded_file is not None:
        if st.session_state["aws_resume_filename"] != uploaded_file.name:
            with st.spinner("Parsing resume..."):
                try:
                    out = parse_resume_file(
                        filename=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                    )
                    resume_text = (out.get("resume_text") or "").strip()

                    st.session_state["aws_resume_text"] = resume_text or None
                    st.session_state["aws_resume_filename"] = uploaded_file.name
                    st.session_state["aws_resume_match_result"] = None
                    st.session_state["aws_resume_match_error"] = None

                    st.success("Resume uploaded and parsed successfully.")
                except Exception as e:
                    st.error(f"Failed to parse resume: {e}")

    if st.session_state["aws_resume_text"]:
        with st.expander("Resume Preview", expanded=False):
            preview = st.session_state["aws_resume_text"][:800]
            st.code(preview + ("..." if len(st.session_state["aws_resume_text"]) > 800 else ""))

    query_session_id = st.query_params.get("career_session_id", None)

    if "aws_coach_session_id" not in st.session_state:
        st.session_state["aws_coach_session_id"] = query_session_id

    if "aws_coach_messages" not in st.session_state:
        st.session_state["aws_coach_messages"] = []

    if "aws_coach_done" not in st.session_state:
        st.session_state["aws_coach_done"] = False

    st.query_params["page"] = "AWS Career Coach"
    if st.session_state["aws_coach_session_id"]:
        st.query_params["career_session_id"] = st.session_state["aws_coach_session_id"]

    if st.session_state["aws_coach_session_id"] and not st.session_state["aws_coach_messages"]:
        try:
            restored = get_career_session(st.session_state["aws_coach_session_id"])
            st.session_state["aws_coach_messages"] = restored.get("history", [])
            st.session_state["aws_coach_done"] = bool(restored.get("done", False))
        except Exception as e:
            st.warning(f"Could not restore previous session: {e}")

    c1, c2 = st.columns([1, 1])

    if c1.button("Start New Conversation", use_container_width=True):
        try:
            out = start_career_session()
            st.session_state["aws_coach_session_id"] = out["session_id"]
            st.session_state["aws_coach_messages"] = [
                {"role": "assistant", "content": out["reply"]}
            ]
            st.session_state["aws_coach_done"] = bool(out.get("done", False))
            st.session_state["aws_resume_match_result"] = None
            st.session_state["aws_resume_match_error"] = None
            st.query_params["page"] = "AWS Career Coach"
            st.query_params["career_session_id"] = out["session_id"]
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start conversation: {e}")
            return

    if c2.button("Reset Local Chat", use_container_width=True):
        st.session_state["aws_coach_session_id"] = None
        st.session_state["aws_coach_messages"] = []
        st.session_state["aws_coach_done"] = False
        st.session_state["aws_resume_match_result"] = None
        st.session_state["aws_resume_match_error"] = None
        st.query_params["page"] = "AWS Career Coach"
        st.query_params["career_session_id"] = ""
        st.rerun()

    if not st.session_state["aws_coach_session_id"]:
        st.info("Start a new conversation to begin.")
        return

    for msg in st.session_state["aws_coach_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state["aws_coach_done"]:
        st.success("Conversation completed.")

        resume_text = st.session_state.get("aws_resume_text")

        c1, c2 = st.columns([2, 1])

        with c1:
            if resume_text:
                if st.button("Find matching jobs from your resume", use_container_width=True):
                    with st.spinner("Finding matching jobs from your resume..."):
                        try:
                            result = match_resume(resume_text=resume_text, top_k=5)
                            st.session_state["aws_resume_match_result"] = result
                            st.session_state["aws_resume_match_error"] = None
                        except Exception as e:
                            st.session_state["aws_resume_match_result"] = None
                            st.session_state["aws_resume_match_error"] = str(e)
                    st.rerun()
            else:
                st.info("Upload a resume to enable job matching.")

        with c2:
            if st.button("Start New Conversation", use_container_width=True, key="done_start_new"):
                try:
                    out = start_career_session()
                    st.session_state["aws_coach_session_id"] = out["session_id"]
                    st.session_state["aws_coach_messages"] = [
                        {"role": "assistant", "content": out["reply"]}
                    ]
                    st.session_state["aws_coach_done"] = bool(out.get("done", False))
                    st.session_state["aws_resume_match_result"] = None
                    st.session_state["aws_resume_match_error"] = None
                    st.query_params["page"] = "AWS Career Coach"
                    st.query_params["career_session_id"] = out["session_id"]
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start conversation: {e}")
                    return

        if st.session_state.get("aws_resume_match_error"):
            st.error(f"Resume match failed: {st.session_state['aws_resume_match_error']}")

        if st.session_state.get("aws_resume_match_result"):
            _render_resume_match_results(st.session_state["aws_resume_match_result"])

        return

    user_input = st.chat_input("Type your answer...")
    if not user_input:
        return

    cleaned = user_input.strip()
    if not cleaned:
        st.warning("Please enter a message.")
        return

    st.session_state["aws_coach_messages"].append({"role": "user", "content": cleaned})

    with st.chat_message("user"):
        st.write(cleaned)

    with st.spinner("Waiting for AWS Career Coach..."):
        try:
            out = send_career_message(
                session_id=st.session_state["aws_coach_session_id"],
                message=cleaned,
                resume_text=st.session_state.get("aws_resume_text"),
            )
        except Exception as e:
            st.error(f"Conversation step failed: {e}")
            return

    reply = out.get("reply") or "Sorry, I did not receive a reply."
    st.session_state["aws_coach_messages"].append({"role": "assistant", "content": reply})
    st.session_state["aws_coach_done"] = bool(out.get("done", False))
    st.query_params["page"] = "AWS Career Coach"
    st.query_params["career_session_id"] = st.session_state["aws_coach_session_id"]

    st.rerun()
