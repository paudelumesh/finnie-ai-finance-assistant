"""Small shared UI pieces and the chat helper used by every tab."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from src.core.base_agent import DISCLAIMER
from src.data.market_data import Quote
from src.web_app.runtime import Runtime

# Label + icon together, so freshness never relies on colour alone.
FRESHNESS = {
    "live": ("🟢", "Live"),
    "cache": ("🔵", "Cached (under 30 min old)"),
    "stale": ("🟠", "Delayed: live feed unavailable, showing last known"),
    "mock": ("⚪", "Sample data: live feed unavailable"),
}
_SEVERITY = ["live", "cache", "stale", "mock"]


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def disclaimer_banner() -> None:
    st.caption(f"⚠️ {DISCLAIMER}")


def worst_source(quotes: list[Quote]) -> str:
    return max((q.source for q in quotes), key=_SEVERITY.index, default="live")


def freshness_note(source: str, as_of: datetime | None = None) -> str:
    icon, label = FRESHNESS.get(source, ("⚪", source))
    stamp = f" · as of {as_of:%Y-%m-%d %H:%M} UTC" if as_of else ""
    return f"{icon} {label}{stamp}"


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            label = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
            st.markdown(f"- {label}" + (f" · _{s['category']}_" if s.get("category") else ""))


def ask(rt: Runtime, prompt: str, context: dict[str, Any] | None = None) -> dict:
    """Send `prompt` through the workflow and record both sides in the shared chat history."""
    ss = st.session_state
    ctx = dict(context or {})
    if ss.holdings and "portfolio" not in ctx:
        ctx["portfolio"] = ss.holdings
    ss.chat.append({"role": "user", "content": prompt})
    result = rt.run(rt.services.workflow.chat(prompt, ss.thread_id, ss.profile, ctx))
    entry = {
        "role": "assistant",
        "content": result.content,
        "sources": result.sources,
        "agents": result.agents,
        "guardrail": result.guardrail,
    }
    ss.chat.append(entry)
    return entry


def render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("agents"):
                st.caption("Answered by: " + ", ".join(a.replace("_", " ") for a in m["agents"]))
            render_sources(m.get("sources", []))
