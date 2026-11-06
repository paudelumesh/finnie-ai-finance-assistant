"""Finnie Streamlit app.  Run:  streamlit run src/web_app/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `src` importable

import streamlit as st  # noqa: E402

from src.web_app import chat_tab, goals_tab, knowledge_tab, markets_tab, portfolio_tab  # noqa: E402
from src.web_app.components import disclaimer_banner  # noqa: E402
from src.web_app.runtime import Runtime, create_runtime, default_session  # noqa: E402

RISK = ["low", "moderate", "high"]
LEVELS = ["beginner", "intermediate", "advanced"]


@st.cache_resource(show_spinner="Starting Finnie…")
def get_runtime() -> Runtime:
    return create_runtime()


def _startup_error(exc: Exception) -> None:
    st.error(f"Finnie couldn't start: {exc}")
    if isinstance(exc, FileNotFoundError):
        st.code("python -m scripts.build_index", language="bash")
    st.stop()


def _sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.header("About you")
        st.caption("Finnie tailors explanations and goal assumptions to this.")
        ss.profile["risk_tolerance"] = st.selectbox(
            "Comfort with risk", RISK, index=RISK.index(ss.profile["risk_tolerance"]))
        ss.profile["knowledge_level"] = st.selectbox(
            "Investing knowledge", LEVELS, index=LEVELS.index(ss.profile["knowledge_level"]))
        st.divider()
        if st.button("🗑️ New conversation", width="stretch"):
            ss.chat, ss.thread_id = [], default_session()["thread_id"]
            st.rerun()
        st.caption("Finnie is a learning tool. It explains concepts and never tells you what to buy or sell.")


def main() -> None:
    st.set_page_config(page_title="Finnie · AI Finance Assistant", page_icon="💬", layout="wide")
    for key, value in default_session().items():
        st.session_state.setdefault(key, value)
    try:
        rt = get_runtime()
    except Exception as exc:  # missing index / API key: show a helpful message, not a traceback
        _startup_error(exc)
        return

    _sidebar()
    st.title("💬 Finnie")
    st.caption("Your friendly guide to investing, with plain-English answers.")
    disclaimer_banner()

    chat, portfolio, markets, goals, knowledge = st.tabs(["Chat", "Portfolio", "Markets", "Goals", "Knowledge"])
    with chat:
        chat_tab.render(rt)
    with portfolio:
        portfolio_tab.render(rt)
    with markets:
        markets_tab.render(rt)
    with goals:
        goals_tab.render(rt)
    with knowledge:
        knowledge_tab.render(rt)


main()
