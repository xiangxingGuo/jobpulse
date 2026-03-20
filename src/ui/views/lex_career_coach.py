from __future__ import annotations

import streamlit as st

from src.ui.api_client import (
    start_career_session,
    send_career_message,
    get_career_session,
)


def render_lex_career_coach_page() -> None:
    if (
        st.session_state.aws_coach_session_id
        and not st.session_state.aws_coach_messages
    ):
        try:
            restored = get_career_session(st.session_state.aws_coach_session_id)
            st.session_state.aws_coach_messages = restored.get("history", [])
            st.session_state.aws_coach_done = bool(restored.get("done", False))
        except Exception as e:
            st.warning(f"Could not restore previous session: {e}")


    st.subheader("AWS Career Coach")

    st.markdown(
        "This assistant uses a stateful backend workflow powered by "
        "API Gateway, Lambda, and DynamoDB."
    )

    if "aws_coach_session_id" not in st.session_state:
        st.session_state.aws_coach_session_id = None
    if "aws_coach_messages" not in st.session_state:
        st.session_state.aws_coach_messages = []
    if "aws_coach_done" not in st.session_state:
        st.session_state.aws_coach_done = False

    c1, c2 = st.columns([1, 1])

    if c1.button("Start New Conversation", use_container_width=True):
        try:
            out = start_career_session()
            st.session_state.aws_coach_session_id = out["session_id"]
            st.session_state.aws_coach_messages = [
                {"role": "assistant", "content": out["reply"]}
            ]
            st.session_state.aws_coach_done = bool(out.get("done", False))
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start conversation: {e}")
            return

    if c2.button("Reset Local Chat", use_container_width=True):
        st.session_state.aws_coach_session_id = None
        st.session_state.aws_coach_messages = []
        st.session_state.aws_coach_done = False
        st.rerun()

    if not st.session_state.aws_coach_session_id:
        st.info("Start a new conversation to begin.")
        return

    for msg in st.session_state.aws_coach_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.aws_coach_done:
        st.success("Conversation completed. Start a new conversation to run another analysis.")
        return

    user_input = st.chat_input("Type your answer...")
    if not user_input:
        return

    cleaned = user_input.strip()
    if not cleaned:
        st.warning("Please enter a message.")
        return

    st.session_state.aws_coach_messages.append(
        {"role": "user", "content": cleaned}
    )

    with st.chat_message("user"):
        st.write(cleaned)

    with st.spinner("Waiting for AWS Career Coach..."):
        try:
            out = send_career_message(
                session_id=st.session_state.aws_coach_session_id,
                message=cleaned,
            )
        except Exception as e:
            st.error(f"Conversation step failed: {e}")
            return

    reply = out.get("reply") or "Sorry, I did not receive a reply."
    st.session_state.aws_coach_messages.append(
        {"role": "assistant", "content": reply}
    )
    st.session_state.aws_coach_done = bool(out.get("done", False))

    st.rerun()