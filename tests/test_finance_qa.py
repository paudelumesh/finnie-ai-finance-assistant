import pytest

from src.agents.finance_qa import NO_KB_NOTE, FinanceQAAgent
from src.core.base_agent import DISCLAIMER
from src.core.llm import FakeLLM
from src.rag.chunker import chunk_articles
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from tests.test_rag import ARTICLES


@pytest.fixture
def retriever():
    store = VectorStore.build(chunk_articles(ARTICLES, 400, 50), HashEmbedder())
    return Retriever(store, top_k=2, min_score=0.2)


async def test_grounded_answer_cites_sources_and_disclaims(retriever):
    llm = FakeLLM("Interest on interest [1].")
    resp = await FinanceQAAgent(llm, retriever).run("what is compound interest?")
    assert resp.ok and resp.data["grounded"] is True
    assert resp.sources[0].title == "Compound interest"
    assert DISCLAIMER in resp.content
    system, user = llm.calls[0]
    assert "[1]" in user and "Passages:" in user


async def test_ungrounded_answer_is_flagged(retriever):
    llm = FakeLLM("General answer.")
    resp = await FinanceQAAgent(llm, retriever).run("quantum zebra chromodynamics")
    assert resp.data["grounded"] is False and resp.sources == []
    assert NO_KB_NOTE in resp.content and DISCLAIMER in resp.content


async def test_strict_category_restricts_retrieval(retriever):
    resp = await FinanceQAAgent(FakeLLM("ok"), retriever).run(
        "exchange traded fund basket of stocks", {"category": "funds", "category_strict": True}
    )
    assert resp.sources and all(s.category == "funds" for s in resp.sources)


async def test_router_category_is_only_a_hint(retriever):
    resp = await FinanceQAAgent(FakeLLM("ok"), retriever).run(
        "what is compound interest?", {"category": "funds"}
    )
    assert resp.sources[0].title == "Compound interest"


async def test_knowledge_level_reaches_prompt(retriever):
    llm = FakeLLM("ok")
    await FinanceQAAgent(llm, retriever).run(
        "roth ira", {"profile": {"knowledge_level": "beginner"}}
    )
    assert "beginner" in llm.calls[0][1]


async def test_ungrounded_prompt_restricts_scope(retriever):
    llm = FakeLLM("ok")
    await FinanceQAAgent(llm, retriever).run("quantum zebra chromodynamics")
    assert "only cover finance" in llm.calls[0][0]
