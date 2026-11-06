"""LLM abstraction so agents never depend on a concrete provider (and tests can fake it)."""
from __future__ import annotations

from typing import Protocol

from src.core.settings import Settings


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a response."""


class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


class OpenAIClient:
    """Thin async wrapper over langchain-openai's ChatOpenAI."""

    def __init__(self, settings: Settings) -> None:
        key = settings.secrets.openai_api_key
        if not key:
            raise LLMError("OPENAI_API_KEY is not set. Copy .env.example to .env and add it.")
        from langchain_openai import ChatOpenAI

        self._model = ChatOpenAI(
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            timeout=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
            api_key=key,
        )

    async def complete(self, system: str, user: str) -> str:
        try:
            result = await self._model.ainvoke([("system", system), ("human", user)])
        except Exception as exc:  # provider errors vary widely
            raise LLMError(f"LLM call failed: {exc}") from exc
        return str(result.content)


class FakeLLM:
    """Deterministic LLM for tests and offline demos."""

    def __init__(self, reply: str = "fake reply") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm.provider == "openai":
        return OpenAIClient(settings)
    raise LLMError(f"Unsupported LLM provider: {settings.llm.provider}")
