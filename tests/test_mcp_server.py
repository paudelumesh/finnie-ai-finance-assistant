import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from src.data.market_data import MarketDataService, MockProvider
from src.mcp_server.server import Backends, build_server
from src.rag.chunker import chunk_articles
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.workflow.graph import ChatResult
from tests.test_rag import ARTICLES


class FakeWorkflow:
    async def chat(self, question, thread_id):
        return ChatResult("Answer", [{"title": "T"}], ["finance_qa"], {})


@pytest.fixture
def server():
    store = VectorStore.build(chunk_articles(ARTICLES, 400, 50), HashEmbedder())
    return build_server(Backends(MarketDataService(MockProvider()), Retriever(store, 3, 0.1), FakeWorkflow()))


async def call(server, name, **args):
    result = await server.call_tool(name, args)
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def test_lists_all_tools(server):
    names = {t.name for t in await server.list_tools()}
    assert names == {"get_stock_quote", "get_price_trend", "analyze_portfolio_csv",
                     "plan_savings_goal", "search_knowledge_base", "ask_finnie"}
    assert all(t.description for t in await server.list_tools())


async def test_quote_and_trend(server):
    q = await call(server, "get_stock_quote", symbol="$aapl")
    assert q["symbol"] == "AAPL" and q["data_source"] == "mock" and "disclaimer" in q
    t = await call(server, "get_price_trend", symbol="VOO", days=10)
    assert t["days"] == 10 and t["direction"] == "up" and len(t["closes"]) == 10


async def test_quote_errors_are_tool_errors(server):
    with pytest.raises(ToolError, match="No market data"):
        await server.call_tool("get_stock_quote", {"symbol": "ZZZZ"})
    with pytest.raises(ToolError, match="between 5 and 365"):
        await server.call_tool("get_price_trend", {"symbol": "VOO", "days": 1})
    with pytest.raises(ToolError, match="No market data"):
        await server.call_tool("get_price_trend", {"symbol": "ZZZZ"})


async def test_portfolio_tool(server):
    out = await call(server, "analyze_portfolio_csv", holdings_csv="symbol,shares,cost\nVOO,10,400\nBND,20,70\n")
    assert out["total_value"] > 0 and out["risk_rating"] and len(out["positions"]) == 2
    assert out["data_sources"] == ["mock"] and out["total_gain_loss"] is not None
    with pytest.raises(ToolError, match="needs 'symbol'"):
        await server.call_tool("analyze_portfolio_csv", {"holdings_csv": "a,b\n1,2"})


async def test_goal_tool(server):
    out = await call(server, "plan_savings_goal", target=60000, years=8, current=5000, risk="moderate")
    assert out["on_track"] is True and len(out["scenarios"]) == 3
    assert out["scenarios"][0]["monthly_needed"] > out["scenarios"][2]["monthly_needed"]
    with pytest.raises(ToolError, match="unknown risk"):
        await server.call_tool("plan_savings_goal", {"target": 1000, "years": 2, "risk": "yolo"})


async def test_knowledge_search_tool(server):
    out = await call(server, "search_knowledge_base", query="compound interest earns interest on interest")
    assert out["results"][0]["title"] == "Compound interest" and "basics" in out["categories"]
    funds = await call(server, "search_knowledge_base", query="etf expense ratios", category="funds")
    assert all(r["category"] == "funds" for r in funds["results"])
    with pytest.raises(ToolError, match="Unknown category"):
        await server.call_tool("search_knowledge_base", {"query": "x", "category": "nope"})


async def test_ask_finnie_tool(server):
    out = await call(server, "ask_finnie", question="what is an ETF?")
    assert out == {"answer": "Answer", "agents": ["finance_qa"], "sources": [{"title": "T"}]}


def test_backends_build_lazily_and_accept_injection(monkeypatch):
    b = Backends()
    assert b._market is None and b._retriever is None and b._workflow is None
    market = b.market  # builds the real service object without any network call
    assert market.ttl == 1800 and b.market is market
