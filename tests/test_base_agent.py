import pytest

from src.core.base_agent import AgentResponse, BaseAgent
from src.core.llm import FakeLLM, LLMError, build_llm
from src.core.settings import load_settings


class EchoAgent(BaseAgent):
    name = "echo"

    async def _handle(self, query, context):
        return AgentResponse(self.name, await self.llm.complete("sys", query))


class LLMFailAgent(BaseAgent):
    name = "llmfail"

    async def _handle(self, query, context):
        raise LLMError("boom")


class CrashAgent(BaseAgent):
    name = "crash"

    async def _handle(self, query, context):
        raise ValueError("bad")


async def test_run_returns_llm_reply():
    resp = await EchoAgent(FakeLLM("hi")).run("  hello ")
    assert resp.ok and resp.content == "hi"


async def test_empty_query_short_circuits():
    llm = FakeLLM()
    resp = await EchoAgent(llm).run("   ")
    assert resp.error == "empty_query" and llm.calls == []


async def test_llm_error_falls_back():
    resp = await LLMFailAgent(FakeLLM()).run("q")
    assert not resp.ok and resp.error.startswith("llm_error")
    assert resp.content == LLMFailAgent.fallback_message


async def test_unexpected_error_is_contained():
    resp = await CrashAgent(FakeLLM()).run("q")
    assert resp.error.startswith("unexpected")


def test_build_llm_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = load_settings()
    settings.secrets.openai_api_key = None
    with pytest.raises(LLMError):
        build_llm(settings)


def test_build_llm_rejects_unknown_provider():
    settings = load_settings()
    settings.llm.provider = "nope"
    with pytest.raises(LLMError):
        build_llm(settings)
