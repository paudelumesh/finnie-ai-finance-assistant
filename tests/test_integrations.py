"""Tests for the thin adapters over real libraries (OpenAI, yfinance, sentence-transformers,
factory wiring, startup errors) using fakes underneath, so no network or keys are needed."""
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.llm import FakeLLM, LLMError, OpenAIClient
from src.core.settings import load_settings
from src.data.market_data import MarketDataError, YFinanceProvider
from src.data.news import SampleProvider, TavilyProvider
from src.rag.chunker import chunk_articles
from src.rag.embeddings import HashEmbedder, SentenceTransformerEmbedder
from src.rag.vector_store import VectorStore
from src.utils.logging import get_logger, setup_logging
from src.web_app.portfolio_tab import holdings_from_frame
from src.core.portfolio import PortfolioError
from tests.test_rag import ARTICLES


# ---- OpenAI client ------------------------------------------------------------
class _Msg:
    content = "hello"


def _install_fake_openai(monkeypatch, fail=False):
    seen = {}

    class FakeChat:
        def __init__(self, **kw):
            seen.update(kw)

        async def ainvoke(self, messages):
            if fail:
                raise RuntimeError("401")
            seen["messages"] = messages
            return _Msg()

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=FakeChat))
    return seen


async def test_openai_client_passes_settings_and_returns_text(monkeypatch):
    seen = _install_fake_openai(monkeypatch)
    s = load_settings()
    s.secrets.openai_api_key = "sk-x"
    out = await OpenAIClient(s).complete("sys", "usr")
    assert out == "hello" and seen["model"] == s.llm.model and seen["api_key"] == "sk-x"
    assert seen["messages"] == [("system", "sys"), ("human", "usr")]


async def test_openai_client_wraps_provider_errors(monkeypatch):
    _install_fake_openai(monkeypatch, fail=True)
    s = load_settings()
    s.secrets.openai_api_key = "sk-x"
    with pytest.raises(LLMError, match="LLM call failed"):
        await OpenAIClient(s).complete("a", "b")


def test_build_llm_returns_openai_client_when_configured(monkeypatch):
    from src.core.llm import build_llm
    _install_fake_openai(monkeypatch)
    s = load_settings()
    s.secrets.openai_api_key = "sk-x"
    assert isinstance(build_llm(s), OpenAIClient)


# ---- yfinance provider -----------------------------------------------------------
def _install_fake_yf(monkeypatch, closes=None, boom=False):
    class Ticker:
        def __init__(self, symbol):
            pass

        def history(self, period):
            if boom:
                raise RuntimeError("rate limited")
            idx = pd.date_range("2026-01-01", periods=len(closes or []))
            return pd.DataFrame({"Close": closes or []}, index=idx) if closes else pd.DataFrame()

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))


async def test_yfinance_quote_and_history(monkeypatch):
    _install_fake_yf(monkeypatch, closes=[100.0, 102.0, 105.0])
    p = YFinanceProvider()
    q = await p.get_quote("aapl")
    assert (q.symbol, q.price, q.previous_close, q.source) == ("AAPL", 105.0, 102.0, "live")
    hist = await p.get_history("AAPL", 30)
    assert [h.close for h in hist] == [100.0, 102.0, 105.0] and hist[0].day == "2026-01-01"


async def test_yfinance_single_point_uses_same_previous_close(monkeypatch):
    _install_fake_yf(monkeypatch, closes=[50.0])
    q = await YFinanceProvider().get_quote("X")
    assert q.previous_close == 50.0


async def test_yfinance_empty_and_error_become_market_data_errors(monkeypatch):
    _install_fake_yf(monkeypatch, closes=None)
    with pytest.raises(MarketDataError, match="no price data"):
        await YFinanceProvider().get_quote("ZZ")
    with pytest.raises(MarketDataError, match="no history"):
        await YFinanceProvider().get_history("ZZ", 10)
    _install_fake_yf(monkeypatch, boom=True)
    with pytest.raises(MarketDataError, match="yfinance failed"):
        await YFinanceProvider().get_quote("ZZ")
    with pytest.raises(MarketDataError, match="yfinance failed"):
        await YFinanceProvider().get_history("ZZ", 10)


# ---- embeddings ---------------------------------------------------------------------
def test_sentence_transformer_embedder_normalises(monkeypatch):
    class FakeST:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, show_progress_bar=False):
            return np.array([[3.0, 4.0]] * len(texts))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    out = SentenceTransformerEmbedder("m").embed(["a", "b"])
    assert out.dtype == np.float32 and np.allclose(out, [[0.6, 0.8]] * 2)


def test_hash_embedder_handles_empty_text():
    out = HashEmbedder(8).embed([""])
    assert out.shape == (1, 8) and not out.any()


def test_vector_store_detects_out_of_sync_files(tmp_path):
    store = VectorStore.build(chunk_articles(ARTICLES, 400, 50), HashEmbedder())
    store.save(tmp_path)
    import json
    chunks = json.loads((tmp_path / "chunks.json").read_text())[:-1]
    (tmp_path / "chunks.json").write_text(json.dumps(chunks))
    with pytest.raises(ValueError, match="out of sync"):
        VectorStore.load(HashEmbedder(), tmp_path)


# ---- factory wiring -------------------------------------------------------------------
def test_default_news_provider_depends_on_key():
    from src.workflow.factory import _default_news_provider
    s = load_settings()
    s.secrets.tavily_api_key = None
    assert isinstance(_default_news_provider(s), SampleProvider)
    s.secrets.tavily_api_key = "tvly-x"
    assert isinstance(_default_news_provider(s), TavilyProvider)


async def test_build_services_wires_all_six_agents(monkeypatch):
    from src.data.market_data import MockProvider
    from src.workflow import factory

    store = VectorStore.build(chunk_articles(ARTICLES, 400, 50), HashEmbedder())
    monkeypatch.setattr(factory.VectorStore, "load", classmethod(lambda cls, emb, *a, **k: store))
    s = load_settings()
    svc = factory.build_services(s, llm=FakeLLM('{"intents": ["finance_qa"], "confidence": 0.9}'),
                                 provider=MockProvider(), embedder=HashEmbedder(), news_provider=SampleProvider())
    assert set(svc.workflow.agents) == {"finance_qa", "portfolio_analysis", "market_analysis",
                                        "goal_planning", "tax_education", "news_synthesizer"}
    assert svc.retriever.store is store and svc.settings is s
    result = await svc.workflow.chat("what is compound interest?", "wire")
    assert result.agents == ["finance_qa"]
    assert factory.build_workflow(s, llm=FakeLLM(), provider=MockProvider(), embedder=HashEmbedder(),
                                  news_provider=SampleProvider()).agents


# ---- logging / runtime / UI helpers --------------------------------------------------------
def test_logging_setup_passes_level_to_basicconfig(monkeypatch):
    calls = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: calls.append(kw))
    setup_logging("debug")
    setup_logging("not-a-level")
    assert [c["level"] for c in calls] == [logging.DEBUG, logging.INFO]
    assert get_logger("x").name == "finnie.x"


def test_async_runner_runs_coroutines_and_closes():
    from src.web_app.runtime import AsyncRunner

    async def add(a, b):
        return a + b

    runner = AsyncRunner()
    assert runner.run(add(2, 3)) == 5
    runner.close()


def test_blank_rows_with_nan_symbols_are_ignored_not_turned_into_nan_ticker():
    df = pd.DataFrame({"symbol": ["VOO", None, float("nan")], "shares": [1.0, None, None],
                       "cost_basis": [None, None, None]})
    assert [h.symbol for h in holdings_from_frame(df)] == ["VOO"]


def test_holdings_from_frame_validation():
    df = pd.DataFrame({"symbol": ["voo", "", "bnd"], "shares": [2.0, None, 3.0], "cost_basis": [400.0, None, None]})
    h = holdings_from_frame(df)
    assert [(x.symbol, x.shares, x.cost_basis) for x in h] == [("VOO", 2.0, 400.0), ("BND", 3.0, None)]
    with pytest.raises(PortfolioError, match="at least one"):
        holdings_from_frame(pd.DataFrame({"symbol": ["", None], "shares": [None, None], "cost_basis": [None, None]}))
    with pytest.raises(PortfolioError, match="number of shares"):
        holdings_from_frame(pd.DataFrame({"symbol": ["VOO"], "shares": [None], "cost_basis": [None]}))
    with pytest.raises(PortfolioError, match="positive"):
        holdings_from_frame(pd.DataFrame({"symbol": ["VOO"], "shares": [-1.0], "cost_basis": [None]}))


def test_app_shows_helpful_message_when_startup_fails(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from src.web_app import runtime

    st.cache_resource.clear()

    def boom(services=None):
        raise FileNotFoundError("No index in /x. Run `python -m scripts.build_index` first.")

    monkeypatch.setattr(runtime, "create_runtime", boom)
    app_path = str(Path(__file__).resolve().parents[1] / "src" / "web_app" / "app.py")
    from src.core.auth import new_session
    at = AppTest.from_file(app_path, default_timeout=30)
    at.session_state["auth"] = new_session("owner@example.com")
    at.run()
    assert not at.exception
    assert any("couldn't start" in e.value for e in at.error)
    assert any("scripts.build_index" in c.value for c in at.code)
