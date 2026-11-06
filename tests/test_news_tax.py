import pytest

from src.agents.news_synthesizer import NewsSynthesizerAgent, build_search_query
from src.agents.tax_education import NO_KB, TaxEducationAgent
from src.core.base_agent import DISCLAIMER
from src.core.llm import FakeLLM
from src.data.news import (
    NewsError, NewsItem, NewsService, SAMPLE_ITEMS, SampleProvider, TavilyProvider,
)
from src.rag.chunker import chunk_articles
from src.rag.embeddings import HashEmbedder
from src.rag.loader import Article
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore

ITEMS = [NewsItem("Fed holds rates", "https://example.com/a", "The Fed held rates steady.", "Mon")]


class Provider:
    def __init__(self, fail=0):
        self.fail, self.calls = fail, 0

    async def search(self, query, max_results):
        self.calls += 1
        if self.calls <= self.fail:
            raise NewsError("rate limited")
        return ITEMS


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


async def no_sleep(_):
    pass


def svc(provider, clock=None, **kw):
    return NewsService(provider, clock=clock or Clock(), sleep=no_sleep, **kw)


# ---- NewsService ----------------------------------------------------------
async def test_news_cache_then_expiry():
    clock, prov = Clock(), Provider()
    s = svc(prov, clock, ttl_seconds=100)
    assert (await s.search("Q")).source == "live"
    assert (await s.search("q")).source == "cache" and prov.calls == 1
    clock.t = 101
    assert (await s.search("q")).source == "live" and prov.calls == 2


async def test_news_retries_then_succeeds():
    s = svc(Provider(fail=1), max_retries=1)
    assert (await s.search("q")).source == "live"


async def test_news_stale_then_sample_then_raise():
    clock, prov = Clock(), Provider()
    s = svc(prov, clock, ttl_seconds=10, max_retries=0)
    await s.search("q")
    clock.t, prov.fail = 100, 10**6
    assert (await s.search("q")).source == "stale"
    sample = await svc(Provider(fail=10**6), max_retries=0).search("fresh")
    assert sample.source == "sample" and sample.items == SAMPLE_ITEMS[:5]
    with pytest.raises(NewsError):
        await svc(Provider(fail=10**6), max_retries=0, use_sample_on_failure=False).search("q")


async def test_sample_provider_limits_results():
    assert len(await SampleProvider().search("x", 2)) == 2


def test_tavily_requires_key():
    with pytest.raises(NewsError, match="TAVILY_API_KEY"):
        TavilyProvider(None)


async def test_tavily_parses_results_and_wraps_errors():
    prov = TavilyProvider("k")

    class Client:
        async def search(self, *a, **k):
            return {"results": [
                {"title": "T", "url": "u", "content": "a  b\nc", "published_date": "d"},
                {"title": "", "url": "x"},
            ]}

    prov._client = Client()
    items = await prov.search("q", 3)
    assert items == [NewsItem("T", "u", "a b c", "d")]

    class Empty:
        async def search(self, *a, **k):
            return {"results": []}

    prov._client = Empty()
    with pytest.raises(NewsError, match="no news"):
        await prov.search("q", 3)

    class Boom:
        async def search(self, *a, **k):
            raise RuntimeError("401")

    prov._client = Boom()
    with pytest.raises(NewsError, match="failed"):
        await prov.search("q", 3)


# ---- News agent -------------------------------------------------------------
def test_search_query_building():
    assert build_search_query("What's the latest news on Nvidia?") == "Nvidia financial news"
    assert build_search_query("news") == "stock market financial news today"


async def test_news_agent_summarises_with_sources():
    llm = FakeLLM("Fed held rates [1].")
    resp = await NewsSynthesizerAgent(llm, svc(Provider())).run("latest news on interest rates")
    assert resp.ok and resp.sources[0].url == "https://example.com/a" and DISCLAIMER in resp.content
    assert "[1] Fed holds rates" in llm.calls[0][1] and resp.data["freshness"] == "live"


async def test_news_agent_flags_sample_data_and_failure():
    resp = await NewsSynthesizerAgent(FakeLLM("x"), svc(Provider(fail=99), max_retries=0)).run("news")
    assert "sample items, not current news" in resp.content and resp.data["freshness"] == "sample"
    dead = svc(Provider(fail=99), max_retries=0, use_sample_on_failure=False)
    bad = await NewsSynthesizerAgent(FakeLLM("x"), dead).run("news")
    assert bad.error.startswith("news_unavailable")


async def test_news_agent_flags_stale():
    clock, prov = Clock(), Provider()
    s = svc(prov, clock, ttl_seconds=1, max_retries=0)
    agent = NewsSynthesizerAgent(FakeLLM("x"), s)
    await agent.run("news")
    clock.t, prov.fail = 50, 99
    assert "earlier results" in (await agent.run("news")).content


# ---- Tax agent ---------------------------------------------------------------
def tax_retriever():
    arts = [
        Article("roth", "Roth IRA", "retirement", "IRS", "https://www.irs.gov",
                "## Basics\nA Roth IRA is funded with after tax money and qualified withdrawals are tax free."),
        Article("gains", "Capital gains", "tax", "IRS", "https://www.irs.gov",
                "## Basics\nCapital gains tax applies when you sell an investment for a profit."),
    ]
    return Retriever(VectorStore.build(chunk_articles(arts, 400, 50), HashEmbedder()), 2, 0.1)


async def test_tax_agent_grounded_with_sources():
    llm = FakeLLM("Roth withdrawals can be tax free [1].")
    resp = await TaxEducationAgent(llm, tax_retriever()).run("how is a roth ira taxed")
    assert resp.ok and resp.data["grounded"] and DISCLAIMER in resp.content
    assert {s.title for s in resp.sources} <= {"Roth IRA", "Capital gains"}
    assert "NEVER tell the user what they personally owe" in llm.calls[0][0]


async def test_tax_agent_without_material_points_to_irs():
    llm = FakeLLM("unused")
    resp = await TaxEducationAgent(llm, tax_retriever()).run("quantum zebra chromodynamics")
    assert NO_KB in resp.content and resp.data["grounded"] is False and llm.calls == []


def test_retrieve_preferring_boosts_preferred_categories():
    r = tax_retriever()
    hits = r.retrieve_preferring("withdrawals capital gains tax", ("tax",), bonus=1.0).hits
    assert hits[0].chunk.category == "tax"
