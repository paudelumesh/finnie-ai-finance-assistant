"""Finnie as a Model Context Protocol (MCP) server, for Claude Desktop and other MCP clients.

Five tools are deterministic (no LLM): quotes, trends, portfolio analysis, goal planning and
knowledge-base search. `ask_finnie` runs the full multi-agent workflow and needs an OpenAI key.
Services load lazily so the server starts instantly and tests need no network.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from src.core.base_agent import DISCLAIMER
from src.core.goals import GoalError, plan_goal
from src.core.portfolio import PortfolioError, analyze_portfolio, parse_holdings_csv
from src.data.market_data import MarketDataError, MarketDataService, YFinanceProvider
from src.workflow.graph import FinnieWorkflow

INSTRUCTIONS = (
    "Finnie is an educational personal-finance assistant. Tools return facts and calculations for "
    "learning; they are not investment, tax or financial advice. Always relay the disclaimer."
)


class Backends:
    """Lazily-built services; any piece can be injected (tests) or built on first use."""

    def __init__(self, market: MarketDataService | None = None, retriever: Any = None,
                 workflow: FinnieWorkflow | None = None) -> None:
        self._market, self._retriever, self._workflow = market, retriever, workflow

    @property
    def market(self) -> MarketDataService:
        if self._market is None:
            from src.core.settings import get_settings

            cfg = get_settings().market_data
            self._market = MarketDataService(YFinanceProvider(), cfg.cache_ttl_seconds,
                                             use_mock_on_failure=cfg.use_mock_on_failure)
        return self._market

    @property
    def retriever(self):
        if self._retriever is None:
            from src.core.settings import get_settings
            from src.rag.embeddings import SentenceTransformerEmbedder
            from src.rag.retriever import Retriever
            from src.rag.vector_store import VectorStore

            rag = get_settings().rag
            store = VectorStore.load(SentenceTransformerEmbedder(rag.embedding_model))
            self._retriever = Retriever(store, rag.top_k, rag.min_score)
        return self._retriever

    @property
    def workflow(self) -> FinnieWorkflow:
        if self._workflow is None:
            from src.workflow.factory import build_workflow

            self._workflow = build_workflow()
        return self._workflow


def build_server(backends: Backends | None = None) -> MCPServer:
    b = backends or Backends()
    server = MCPServer("finnie", instructions=INSTRUCTIONS)

    @server.tool()
    async def get_stock_quote(symbol: str) -> dict[str, Any]:
        """Latest price for a stock or ETF ticker, with change vs previous close and data freshness."""
        try:
            q = await b.market.get_quote(symbol.strip().lstrip("$"))
        except MarketDataError as exc:
            raise ToolError(f"No market data for {symbol}: {exc}") from exc
        return {"symbol": q.symbol, "price": round(q.price, 2), "change_pct": round(q.change_pct, 2),
                "currency": q.currency, "as_of": q.as_of.isoformat(), "data_source": q.source,
                "note": "data_source: live | cache | stale | mock (sample data)", "disclaimer": DISCLAIMER}

    @server.tool()
    async def get_price_trend(symbol: str, days: int = 30) -> dict[str, Any]:
        """Recent closing prices (trading days) with percentage change and direction."""
        if not 5 <= days <= 365:
            raise ToolError("days must be between 5 and 365")
        try:
            t = await b.market.get_trend(symbol.strip().lstrip("$"), days)
        except MarketDataError as exc:
            raise ToolError(f"No market data for {symbol}: {exc}") from exc
        return {"symbol": t.symbol, "days": len(t.points), "change_pct": round(t.change_pct, 2),
                "direction": t.direction, "data_source": t.source,
                "closes": [{"date": p.day, "close": round(p.close, 2)} for p in t.points]}

    @server.tool()
    async def analyze_portfolio_csv(holdings_csv: str) -> dict[str, Any]:
        """Analyse holdings given as CSV text (columns: symbol, shares, optional cost_basis).

        Returns total value, allocation by asset class and sector, a 0-100 diversification score,
        weighted expense ratio, a risk rating and warnings.
        """
        try:
            holdings = parse_holdings_csv(holdings_csv)
            quotes = await b.market.get_quotes([h.symbol for h in holdings])
            a = analyze_portfolio(holdings, quotes)
        except (PortfolioError, MarketDataError) as exc:
            raise ToolError(str(exc)) from exc
        return {
            "total_value": a.total_value, "risk_rating": a.risk_rating,
            "diversification_score": a.diversification_score,
            "weighted_expense_ratio_pct": round(a.weighted_expense_ratio * 100, 4),
            "annual_fee_estimate": a.annual_fee_estimate, "total_gain_loss": a.total_gain_loss,
            "allocation_by_class": {k: round(v, 4) for k, v in a.allocation_by_class.items()},
            "allocation_by_sector": {k: round(v, 4) for k, v in a.allocation_by_sector.items()},
            "positions": [{"symbol": p.symbol, "value": round(p.value, 2), "weight": round(p.weight, 4)}
                          for p in a.positions],
            "warnings": a.warnings,
            "data_sources": sorted({q.source for q in quotes.values()}), "disclaimer": DISCLAIMER,
        }

    @server.tool()
    async def plan_savings_goal(target: float, years: float, current: float = 0.0,
                                risk: str = "moderate", monthly: float | None = None) -> dict[str, Any]:
        """Project a savings goal under conservative, expected and optimistic return assumptions.

        risk is low, moderate or high. Leave monthly empty to compute what the expected scenario needs.
        Returns are illustrative planning assumptions, not forecasts.
        """
        try:
            plan = plan_goal(target, years, current, risk, monthly)
        except GoalError as exc:
            raise ToolError(str(exc)) from exc
        out = asdict(plan)
        out["disclaimer"] = DISCLAIMER
        return out

    @server.tool()
    async def search_knowledge_base(query: str, category: str | None = None, limit: int = 4) -> dict[str, Any]:
        """Search Finnie's curated finance-education articles; optionally restrict to one category."""
        retriever = b.retriever
        if category and category not in retriever.store.categories:
            raise ToolError(f"Unknown category '{category}'. Choose from: {', '.join(retriever.store.categories)}")
        hits = retriever.store.search(query, max(1, min(limit, 10)), category, retriever.min_score)
        return {"results": [
            {"title": h.chunk.title, "category": h.chunk.category, "score": round(h.score, 3),
             "text": h.chunk.text, "source": h.chunk.source, "source_url": h.chunk.source_url}
            for h in hits], "categories": retriever.store.categories}

    @server.tool()
    async def ask_finnie(question: str, thread_id: str = "mcp") -> dict[str, Any]:
        """Ask Finnie anything: routes to the right specialist agents and returns a guarded, cited answer.

        Uses the OpenAI API, so it needs OPENAI_API_KEY in Finnie's .env.
        """
        result = await b.workflow.chat(question, thread_id)
        return {"answer": result.content, "agents": result.agents, "sources": result.sources}

    return server
