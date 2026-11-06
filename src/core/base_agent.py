"""Base class shared by all six Finnie agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.core.llm import LLMClient, LLMError
from src.utils.logging import get_logger

DISCLAIMER = (
    "This is educational information only, not financial, tax or investment advice. "
    "Consider speaking with a licensed professional before making decisions."
)


@dataclass
class Source:
    title: str
    url: str | None = None
    category: str | None = None


@dataclass
class AgentResponse:
    agent: str
    content: str
    sources: list[Source] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class BaseAgent(ABC):
    """Subclasses implement `_handle`; `run` adds logging and uniform error fallback."""

    name: str = "base"
    description: str = ""
    fallback_message = (
        "Sorry, I couldn't complete that request right now. Please try again in a moment."
    )

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.log = get_logger(f"agent.{self.name}")

    @abstractmethod
    async def _handle(self, query: str, context: dict[str, Any]) -> AgentResponse:
        """Produce a response for `query`. `context` carries profile, history, etc."""

    async def run(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        context = context or {}
        if not query or not query.strip():
            return AgentResponse(self.name, "Please ask me a question.", error="empty_query")
        self.log.info("handling query (%d chars)", len(query))
        try:
            return await self._handle(query.strip(), context)
        except LLMError as exc:
            self.log.warning("LLM failure: %s", exc)
            return AgentResponse(self.name, self.fallback_message, error=f"llm_error: {exc}")
        except Exception as exc:  # last-resort: agents must never crash the workflow
            self.log.exception("unexpected agent failure")
            return AgentResponse(self.name, self.fallback_message, error=f"unexpected: {exc}")
