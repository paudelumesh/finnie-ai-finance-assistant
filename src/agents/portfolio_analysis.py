"""Portfolio Analysis agent: deterministic metrics + a plain-language explanation."""
from __future__ import annotations

from typing import Any

from src.core.base_agent import DISCLAIMER, AgentResponse, BaseAgent
from src.core.llm import LLMClient
from src.core.portfolio import (
    CLASS_LABELS, Holding, PortfolioAnalysis, PortfolioError, analyze_portfolio,
)
from src.data.market_data import MarketDataError, MarketDataService

SYSTEM = (
    "You are Finnie, a friendly portfolio educator. Explain the portfolio metrics provided, "
    "using ONLY those numbers (they were computed by code; do not recompute or invent any). "
    "Cover: allocation, diversification, fees, risk and the warnings. Describe general "
    "principles a learner could consider, but never tell the user to buy, sell or hold "
    "specific securities. Keep it under 250 words."
)

NO_PORTFOLIO = (
    "I don't have a portfolio for you yet. Upload a CSV (columns: symbol, shares, optional "
    "cost_basis) or enter your holdings in the Portfolio tab, then ask me again."
)


def to_holdings(raw: list[Any]) -> list[Holding]:
    out = []
    for item in raw:
        out.append(item if isinstance(item, Holding) else Holding(
            str(item["symbol"]).upper(), float(item["shares"]), item.get("cost_basis")))
    return out


def summarise(a: PortfolioAnalysis) -> str:
    lines = [
        f"Total value: ${a.total_value:,.2f}",
        "Allocation by asset class: " + ", ".join(
            f"{CLASS_LABELS.get(k, k)} {v:.0%}" for k, v in a.allocation_by_class.items()),
        "Allocation by sector: " + ", ".join(f"{k} {v:.0%}" for k, v in a.allocation_by_sector.items()),
        f"Diversification score: {a.diversification_score}/100 (based on effective number of holdings)",
        f"Weighted expense ratio: {a.weighted_expense_ratio:.3%} (about ${a.annual_fee_estimate:,.2f}/year)",
        f"Risk rating: {a.risk_rating}",
        "Top holdings: " + ", ".join(f"{p.symbol} {p.weight:.0%}" for p in a.positions[:5]),
    ]
    if a.total_gain_loss is not None:
        lines.append(f"Total gain/loss vs cost basis: ${a.total_gain_loss:,.2f}")
    if a.warnings:
        lines.append("Warnings: " + " | ".join(a.warnings))
    return "\n".join(lines)


class PortfolioAnalysisAgent(BaseAgent):
    name = "portfolio_analysis"
    description = "Analyses uploaded holdings for allocation, diversification, fees and risk."

    def __init__(self, llm: LLMClient, market: MarketDataService) -> None:
        super().__init__(llm)
        self.market = market

    async def _handle(self, query: str, context: dict[str, Any]) -> AgentResponse:
        raw = context.get("portfolio")
        if not raw:
            return AgentResponse(self.name, NO_PORTFOLIO, data={"needs_portfolio": True})
        try:
            holdings = to_holdings(raw)
            quotes = await self.market.get_quotes([h.symbol for h in holdings])
            analysis = analyze_portfolio(holdings, quotes)
        except (PortfolioError, MarketDataError, KeyError, TypeError, ValueError) as exc:
            return AgentResponse(
                self.name, f"I couldn't analyse that portfolio: {exc}", error="invalid_portfolio"
            )
        degraded = any(q.source in {"mock", "stale"} for q in quotes.values())
        answer = await self.llm.complete(
            SYSTEM, f"User question: {query}\n\nPortfolio metrics:\n{summarise(analysis)}"
        )
        if degraded:
            answer += "\n\n_Heads up: some prices are sample or delayed data because the live feed was unavailable._"
        return AgentResponse(
            agent=self.name,
            content=f"{answer}\n\n_{DISCLAIMER}_",
            data={"analysis": analysis, "quotes": quotes, "degraded": degraded},
        )
