"""Chat tab: the conversational front door to all six agents."""
from __future__ import annotations

import streamlit as st

from src.web_app.components import ask, render_message
from src.web_app.runtime import Runtime

EXAMPLES = [
    "What's the difference between an ETF and a mutual fund?",
    "How is NVDA doing today?",
    "I want $60,000 in 8 years and have $5,000 saved. How much should I save monthly?",
    "How is a Roth IRA different from a traditional IRA?",
    "What's the latest financial news on interest rates?",
    "Analyze my portfolio and explain what diversification means",
]


def render(rt: Runtime) -> None:
    ss = st.session_state
    if not ss.chat:
        st.markdown(
            "### 👋 Hi, I'm Finnie\n"
            "I explain investing in plain language. Ask me about concepts, live prices, your "
            "portfolio, savings goals, taxes or the news. Try one of these:"
        )
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, key=f"ex{i}", width="stretch"):
                ss.pending_prompt = example
                st.rerun()

    for m in ss.chat:
        render_message(m)

    prompt = st.chat_input("Ask Finnie about investing, markets, taxes or your goals…")
    prompt = prompt or ss.pop("pending_prompt", None)
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"), st.spinner("Finnie is thinking…"):
            ask(rt, prompt)
        st.rerun()
