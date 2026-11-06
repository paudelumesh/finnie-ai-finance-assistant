"""'About Finnie' menu: each item opens in its own modal dialog."""
from __future__ import annotations

from typing import Callable

import streamlit as st

from src.web_app import about_content as c


def _diagram(dot: str) -> None:
    st.graphviz_chart(dot, width="stretch")


@st.dialog("Architecture", width="large")
def architecture() -> None:
    st.markdown(
        "Finnie is a **multi-agent system**: one chat box in front, six specialist agents behind it. "
        "A router decides who should answer, the agents fetch data or retrieve knowledge, and their "
        "answers are merged and checked before you see them."
    )
    _diagram(c.SYSTEM_DIAGRAM)
    st.markdown(
        "**Layers**\n"
        "- **Presentation**: Streamlit tabs for chat, portfolio, markets, goals, knowledge and this About menu.\n"
        "- **Orchestration**: a LangGraph workflow (router, parallel agents, synthesizer, guardrail).\n"
        "- **Services**: the RAG retriever, market data, news search and pure-code calculators.\n"
        "- **External**: OpenAI for language, yFinance for prices, Tavily for news."
    )
    st.caption("Dashed arrows are LLM calls. Every external dependency has a cache and a labelled fallback.")


@st.dialog("High-level design", width="large")
def high_level_design() -> None:
    st.markdown("### What Finnie does\n"
                "Most beginners are blocked by jargon and fear of mistakes. Finnie explains investing in plain "
                "language, grounds answers in a curated knowledge base, shows live data and always stays on the "
                "education side of the line.")
    st.markdown("### One question, end to end")
    _diagram(c.FLOW_DIAGRAM)
    st.markdown("### The six agents")
    st.table({
        "Agent": [a[0] for a in c.AGENTS], "Handles": [a[1] for a in c.AGENTS],
        "Data source": [a[2] for a in c.AGENTS], "How it stays reliable": [a[3] for a in c.AGENTS],
    })
    st.markdown("### Design principles")
    for title, text in c.PRINCIPLES:
        st.markdown(f"**{title}.** {text}")


@st.dialog("Technical architecture in detail", width="large")
def technical_architecture() -> None:
    st.markdown("### 1. Workflow graph (LangGraph)")
    _diagram(c.GRAPH_DIAGRAM)
    st.markdown(
        "- **State** is a typed dict of JSON-friendly values: `query`, `turn`, `profile`, `context`, `messages` "
        "(append-only history), `route`, `responses` (append-only, tagged with the turn number) and `final`. "
        "Keeping it JSON-safe means any checkpointer can store it.\n"
        "- **Fan-out** uses `Send`: the router returns one `Send` per selected intent, so agents run concurrently "
        "and write to `responses` through an append reducer. The synthesizer only reads this turn's responses.\n"
        "- **Memory**: an in-memory checkpointer keyed by `thread_id` keeps history and the user profile across "
        "turns, so follow-ups like 'what about 12 years?' work. Threads are isolated from each other."
    )

    st.markdown("### 2. Routing")
    st.markdown(
        "1. An LLM classifier returns JSON `{intents, confidence, category}`, given the last four messages so it "
        "can resolve follow-ups.\n"
        "2. If the reply is invalid or the call fails or times out (15 s), **keyword rules** pick the intents.\n"
        "3. If confidence is below **0.6**, the LLM intents are **unioned** with keyword matches, so a vague "
        "multi-part question reaches every relevant agent.\n"
        "4. Intents are resolved against the registered agents. Off-topic or unavailable requests go to a "
        "`refuse` node with a clear message instead of an agent.\n"
        "5. The router's knowledge-base `category` is only a **hint** for retrieval, so a wrong guess can't hide "
        "the best article."
    )

    st.markdown("### 3. Agent contract and isolation")
    st.markdown(
        "Every agent extends `BaseAgent` and implements one method, `_handle(query, context) → AgentResponse` "
        "(content, sources, data, error). `run()` wraps it with logging and converts LLM errors and unexpected "
        "exceptions into a friendly fallback, and the graph adds a **60 s timeout** per agent. If some agents "
        "fail, the rest still answer and the reply notes what was skipped. Agents read context (profile, "
        "history, portfolio, goal) but never touch graph state directly."
    )

    st.markdown("### 4. Retrieval-augmented generation")
    _diagram(c.RAG_DIAGRAM)
    st.markdown(
        "Search is exact (every chunk is scored) because the corpus is small, which also makes category filtering "
        "exact. **Strict** filtering (Knowledge tab) restricts results to a category. **Soft** preference (router "
        "or the Tax agent) boosts a category but never hides a clearly better match. A `min_score` of 0.25 stops "
        "weak matches; with no match, the Q&A agent says so and gives a labelled general answer, or declines if "
        "the topic isn't finance."
    )

    st.markdown("### 5. Resilient external data")
    _diagram(c.RESILIENCE_DIAGRAM)
    st.markdown(
        "Quotes, price history and news share one pattern: a 30-minute TTL cache, retries with exponential "
        "backoff, then stale data, then clearly labelled sample data. Every result carries its source, which the "
        "UI shows as Live, Cached, Delayed or Sample. Sample prices exist only for known tickers; unknown symbols "
        "raise an error rather than invent a price. The 25-requests-a-day free tier of Alpha Vantage is avoided by "
        "using yFinance as the primary source."
    )

    st.markdown("### 6. Computation vs language")
    st.markdown(
        "- **Portfolio**: value, weights, allocation by asset class and sector, diversification score "
        "(effective number of holdings = 1/HHI, scaled so 1 holding = 0 and 10 or more = 100), weighted expense "
        "ratio from a bundled ETF table, a rule-based risk rating, gain/loss and concentration warnings.\n"
        "- **Goals**: monthly-compounding future value and its inverse, with conservative, expected and "
        "optimistic annual returns per risk profile. These are illustrative assumptions, not forecasts.\n"
        "- The LLM receives these results as facts and only explains them."
    )

    st.markdown("### 7. Guardrail")
    st.markdown(
        "After merging, one node checks the **question** for advice-seeking patterns (adds a framing note) and the "
        "**answer** for recommendation language (rewrites it with the LLM, falling back to dropping those "
        "sentences). It then guarantees exactly one disclaimer and stores the guarded text as the chat memory."
    )

    st.markdown("### 8. Access control")
    st.markdown(
        "The whole UI sits behind a single-owner login that runs before the model or any service starts. "
        "Credentials come from `AUTH_EMAIL` and `AUTH_PASSWORD_HASH` (a salted scrypt hash, never the password); "
        "if they are missing or malformed the app stays locked. Five failed attempts lock sign-in for 15 minutes "
        "process-wide, wrong email and wrong password look identical, and sessions expire after 8 hours. "
        "There is deliberately no sign-up."
    )

    st.markdown("### 9. Runtime, configuration and quality")
    st.markdown(
        "- **Async**: the workflow is async end to end. Streamlit reruns synchronously, so a single background "
        "event loop runs every coroutine and async clients always live on one loop. Blocking libraries (yFinance, "
        "embeddings) use worker threads.\n"
        "- **Configuration**: tunables (models, TTLs, chunking, thresholds, timeouts) are in `config.yaml`; secrets "
        "come from `.env` or `st.secrets` and are never committed.\n"
        "- **Testing**: all LLM and network calls sit behind small interfaces and are faked in tests, including "
        "headless UI tests with Streamlit's AppTest."
    )


@st.dialog("Technologies used", width="large")
def technologies() -> None:
    st.markdown("Every choice below is a real dependency in this project.")
    st.table({
        "Area": [t[0] for t in c.TECHNOLOGIES],
        "Technology": [t[1] for t in c.TECHNOLOGIES],
        "Role in Finnie": [t[2] for t in c.TECHNOLOGIES],
    })


MENU: dict[str, Callable[[], None]] = {
    "🏗️ Architecture": architecture,
    "🧭 High-level design": high_level_design,
    "🔬 Technical architecture in detail": technical_architecture,
    "🧰 Technologies used": technologies,
}


def render_menu() -> None:
    """Sidebar menu. The dialog opens after the popover so it is a top-level modal."""
    chosen = None
    with st.popover("ℹ️ About Finnie", width="stretch"):
        for label in MENU:
            if st.button(label, key=f"about_{label}", width="stretch"):
                chosen = label
    if chosen:
        MENU[chosen]()
