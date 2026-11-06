"""News Synthesizer agent: searches current financial news and summarises it with sources."""
from __future__ import annotations

import re
from typing import Any

from src.core.base_agent import DISCLAIMER, AgentResponse, BaseAgent, Source
from src.core.llm import LLMClient
from src.data.news import NewsError, NewsService

SYSTEM = (
    "You are Finnie, a financial news explainer for beginners. Summarise the numbered articles "
    "provided, using ONLY their content; never add facts, figures or predictions of your own. "
    "Cite articles inline like [1]. Group related stories, explain any jargon simply, note when "
    "articles disagree, and mention that headlines are not a reason to trade. "
    "Never recommend buying or selling anything. Keep it under 220 words."
)

_FILLER = re.compile(
    r"\b(please|can you|could you|tell me|what('s| is| are)|the|latest|current|recent|news|"
    r"headlines?|about|on|any|summari[sz]e|me|give|show|today)\b", re.I)


def build_search_query(query: str) -> str:
    """Keep the user's topic words and anchor the search on financial news."""
    topic = " ".join(_FILLER.sub(" ", query).replace("?", " ").split())
    return f"{topic} financial news" if topic else "stock market financial news today"


class NewsSynthesizerAgent(BaseAgent):
    name = "news_synthesizer"
    description = "Summarises current financial news with source links."

    def __init__(self, llm: LLMClient, news: NewsService, max_results: int = 5) -> None:
        super().__init__(llm)
        self.news = news
        self.max_results = max_results

    async def _handle(self, query: str, context: dict[str, Any]) -> AgentResponse:
        try:
            result = await self.news.search(build_search_query(query), self.max_results)
        except NewsError as exc:
            return AgentResponse(
                self.name, "I couldn't reach the news service right now. Please try again shortly.",
                error=f"news_unavailable: {exc}",
            )
        articles = "\n\n".join(
            f"[{i}] {n.title} ({n.published or 'date unknown'})\n{n.snippet}"
            for i, n in enumerate(result.items, start=1)
        )
        answer = await self.llm.complete(SYSTEM, f"Question: {query}\n\nArticles:\n{articles}")
        note = {
            "sample": "\n\n_Heads up: the live news feed was unavailable, so these are generic sample items, not current news._",
            "stale": "\n\n_Heads up: the live news feed was unavailable, so this uses earlier results._",
        }.get(result.source, "")
        return AgentResponse(
            agent=self.name,
            content=f"{answer}{note}\n\n_{DISCLAIMER}_",
            sources=[Source(n.title, n.url, "news") for n in result.items],
            data={"count": len(result.items), "freshness": result.source},
        )
