"""Bridges Streamlit (sync, rerun-per-interaction) and the async workflow.

One long-lived background event loop runs every coroutine, so async clients (httpx, LangGraph)
are always used from the loop they were created on.
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, TypeVar

from src.workflow.factory import Services, build_services

T = TypeVar("T")


class AsyncRunner:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="finnie-loop", daemon=True)
        self._thread.start()

    def run(self, coro: Awaitable[T], timeout: float = 120) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)  # type: ignore[arg-type]

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)


@dataclass
class Runtime:
    services: Services
    runner: AsyncRunner

    def run(self, coro: Awaitable[T], timeout: float = 120) -> T:
        return self.runner.run(coro, timeout)


def create_runtime(services: Services | None = None) -> Runtime:
    return Runtime(services or build_services(), AsyncRunner())


def default_session() -> dict[str, Any]:
    """Initial per-browser-session state (kept in st.session_state)."""
    return {
        "thread_id": uuid.uuid4().hex,
        "chat": [],  # [{role, content, sources, agents, guardrail}]
        "profile": {"risk_tolerance": "moderate", "knowledge_level": "beginner"},
        "holdings": [],  # [{symbol, shares, cost_basis}]
        "analysis": None,  # last PortfolioAnalysis
        "quotes": {},
        "plan": None,  # last GoalPlan
    }
