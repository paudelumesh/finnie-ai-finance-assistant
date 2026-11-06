"""Market Analysis agent: live quotes and short-term trends, explained for beginners."""
from __future__ import annotations

import asyncio
import re
from typing import Any

from src.core.base_agent import DISCLAIMER, AgentResponse, BaseAgent
from src.core.llm import LLMClient
from src.data.asset_info import ASSETS, NAME_ALIASES
from src.data.market_data import MarketDataError, MarketDataService, Quote, Trend

OVERVIEW_SYMBOLS = ["SPY", "QQQ", "IWM", "BND"]
MAX_SYMBOLS = 5
_STOPWORDS = {
    "I", "A", "AN", "THE", "AND", "OR", "IS", "IT", "ETF", "ETFS", "IRA", "USD", "US", "USA",
    "AI", "PE", "ROTH", "HOW", "WHY", "WHAT", "DO", "DOES", "ON", "IN", "OF", "TO", "FOR",
    "MY", "ME", "CEO", "IPO", "GDP", "CPI", "FED", "SEC", "IRS", "EPS", "ROI", "NOW", "ARE",
}

SYSTEM = (
    "You are Finnie, a friendly market explainer for beginners. Use ONLY the figures provided; "
    "never invent prices or news. Explain what the numbers mean in plain language, mention that "
    "short-term moves are normal noise, note the data freshness, and never predict prices or "
    "tell the user to buy or sell. Keep it under 200 words."
)


def extract_symbols(query: str) -> list[str]:
    found: list[str] = []

    def add(sym: str) -> None:
        if sym not in found:
            found.append(sym)

    lowered = query.lower()
    for alias, sym in NAME_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            add(sym)
    for m in re.finditer(r"(\$)?\b([A-Z]{1,5}(?:-USD)?)\b", query):
        sym, dollar = m.group(2), bool(m.group(1))
        if sym in _STOPWORDS:
            continue
        if dollar or sym in ASSETS:  # unknown tickers must be written like $PLTR
            add(sym)
    return found[:MAX_SYMBOLS]


def describe(quote: Quote, trend: Trend | None) -> str:
    arrow = "up" if quote.change >= 0 else "down"
    line = (f"{quote.symbol}: ${quote.price:,.2f} ({arrow} {abs(quote.change_pct):.2f}% vs previous close; "
            f"data: {quote.source}, as of {quote.as_of:%Y-%m-%d %H:%M} UTC)")
    if trend and trend.points:
        line += f"; last {len(trend.points)} trading days: {trend.direction} ({trend.change_pct:+.1f}%)"
    return line


class MarketAnalysisAgent(BaseAgent):
    name = "market_analysis"
    description = "Looks up live stock/ETF quotes and explains recent price moves."

    def __init__(self, llm: LLMClient, market: MarketDataService) -> None:
        super().__init__(llm)
        self.market = market

    async def snapshot(self, symbols: list[str]) -> list[tuple[Quote, Trend]]:
        quotes, trends = await asyncio.gather(
            asyncio.gather(*(self.market.get_quote(s) for s in symbols)),
            asyncio.gather(*(self.market.get_trend(s, 30) for s in symbols)),
        )
        return list(zip(quotes, trends))

    async def _handle(self, query: str, context: dict[str, Any]) -> AgentResponse:
        symbols = extract_symbols(query)
        overview = not symbols
        try:
            rows = await self.snapshot(symbols or OVERVIEW_SYMBOLS)
        except MarketDataError as exc:
            self.log.warning("no market data: %s", exc)
            return AgentResponse(
                self.name,
                f"I couldn't find market data for {', '.join(symbols) or 'that request'}. "
                "Check the ticker symbol (write less common ones like $PLTR) and try again.",
                error="no_market_data",
            )
        facts = "\n".join(describe(q, t) for q, t in rows)
        stale = any(q.source in {"mock", "stale"} for q, _ in rows)
        user = f"Question: {query}\n\nData:\n{facts}"
        if overview:
            user += "\n\n(No specific ticker was asked about; give a brief market overview from these index funds.)"
        answer = await self.llm.complete(SYSTEM, user)
        if stale:
            answer += "\n\n_Heads up: some figures are sample or delayed data because the live feed was unavailable._"
        return AgentResponse(
            agent=self.name,
            content=f"{answer}\n\n_{DISCLAIMER}_",
            data={
                "symbols": [q.symbol for q, _ in rows],
                "quotes": [q for q, _ in rows],
                "trends": [t for _, t in rows],
                "overview": overview,
                "degraded": stale,
            },
        )
