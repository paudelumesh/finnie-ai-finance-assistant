"""Query router: LLM intent classifier (structured JSON) with a keyword fallback.

A query can carry several intents ("analyse my portfolio and explain ETFs"); the graph fans
out to every selected agent in parallel and merges the answers.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass, field

from src.agents.market_analysis import extract_symbols
from src.core.llm import LLMClient
from src.utils.json_utils import extract_json
from src.utils.logging import get_logger

log = get_logger("router")

AGENT_INTENTS = (
    "finance_qa",
    "portfolio_analysis",
    "market_analysis",
    "goal_planning",
    "tax_education",
    "news_synthesizer",
)
OFF_TOPIC = "off_topic"
VALID_INTENTS = {*AGENT_INTENTS, OFF_TOPIC}

KEYWORDS: dict[str, re.Pattern[str]] = {
    "portfolio_analysis": re.compile(
        r"\b(my (portfolio|holdings|investments|allocation)|analy[sz]e my|rebalanc\w* my)\b", re.I),
    "market_analysis": re.compile(
        r"(\bprice of\b|\bquote\b|\bstock price\b|\btrading at\b|\bmarket today\b|"
        r"\bhow is \w+ doing\b|\bup today\b|\bdown today\b|\$[A-Za-z]{1,5}\b)", re.I),
    "goal_planning": re.compile(
        r"\b(how much (do i need|should i save)|monthly (saving|contribution)s?|"
        r"save \$?[\d,]+|reach \$?[\d,]+|savings? goal|financial goal|retire in)\b", re.I),
    "tax_education": re.compile(
        r"\b(tax\w*|401\(?k\)?|roth|529|hsa|capital gains|deduct\w*)\b|\bira\b", re.I),
    "news_synthesizer": re.compile(r"\b(news|headlines?|latest on|what happened)\b", re.I),
}

SYSTEM = f"""You are the router for Finnie, a beginner finance education assistant.
Classify the user's LATEST message into one or more intents:
- finance_qa: general finance/investing concepts and definitions
- portfolio_analysis: analysing the user's own holdings (diversification, allocation, risk)
- market_analysis: live prices, quotes, market moves, index performance
- goal_planning: how much to save, projections toward a target amount/date
- tax_education: tax concepts and account types (401(k), IRA, Roth, capital gains)
- news_synthesizer: summarising current financial news
- {OFF_TOPIC}: not about personal finance, investing, taxes or the economy
Use earlier conversation to resolve follow-ups (e.g. "what about 15 years?" continues a goal plan).
Select every intent the message needs. Reply with ONLY JSON:
{{"intents": ["..."], "confidence": 0.0-1.0, "category": "<optional knowledge-base category or null>"}}"""


@dataclass
class RouteDecision:
    intents: list[str]
    confidence: float
    method: str  # llm | llm+keyword | keyword
    category: str | None = None
    unavailable: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def keyword_intents(query: str) -> list[str]:
    found = [name for name, pat in KEYWORDS.items() if pat.search(query)]
    if extract_symbols(query) and "market_analysis" not in found:
        found.append("market_analysis")
    return found


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


class Router:
    def __init__(
        self,
        llm: LLMClient,
        confidence_threshold: float = 0.6,
        timeout: float = 15,
        categories: set[str] | None = None,
    ) -> None:
        self.llm = llm
        self.threshold = confidence_threshold
        self.timeout = timeout
        self.categories = categories or set()

    async def _classify(self, query: str, history: list[dict]) -> dict | None:
        recent = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
        prompt = (f"Conversation so far:\n{recent}\n\n" if recent else "") + f"Latest message: {query}"
        try:
            raw = await asyncio.wait_for(self.llm.complete(SYSTEM, prompt), self.timeout)
        except Exception as exc:  # timeout, provider error: fall back to keywords
            log.warning("classifier failed (%s); using keyword routing", exc)
            return None
        return extract_json(raw)

    async def route(self, query: str, history: list[dict] | None = None) -> RouteDecision:
        history = history or []
        keywords = keyword_intents(query)
        data = await self._classify(query, history)

        intents: list[str] = []
        confidence = 0.0
        category = None
        if data:
            raw_intents = data.get("intents")
            if isinstance(raw_intents, str):
                raw_intents = [raw_intents]
            intents = _dedupe([i for i in (raw_intents or []) if i in VALID_INTENTS])
            try:
                confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            cat = data.get("category")
            category = cat if isinstance(cat, str) and cat in self.categories else None

        if not intents:
            return RouteDecision(keywords or ["finance_qa"], 0.5, "keyword")

        method = "llm"
        if OFF_TOPIC in intents and len(intents) > 1:
            intents.remove(OFF_TOPIC)
        if confidence < self.threshold and intents != [OFF_TOPIC]:
            intents = _dedupe(intents + keywords)
            method = "llm+keyword"
        return RouteDecision(intents, confidence, method, category)
