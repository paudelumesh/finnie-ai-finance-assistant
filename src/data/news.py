"""News layer: Tavily search with TTL cache, retries and a clearly-labelled sample fallback."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from src.utils.logging import get_logger

log = get_logger("news")


class NewsError(RuntimeError):
    """The news provider could not return results."""


@dataclass(frozen=True)
class NewsItem:
    title: str
    url: str
    snippet: str
    published: str | None = None


@dataclass
class NewsResult:
    items: list[NewsItem]
    source: str = "live"  # live | cache | stale | sample


class NewsProvider(Protocol):
    async def search(self, query: str, max_results: int) -> list[NewsItem]: ...


class TavilyProvider:
    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise NewsError("TAVILY_API_KEY is not set")
        from tavily import AsyncTavilyClient

        self._client = AsyncTavilyClient(api_key=api_key)

    async def search(self, query: str, max_results: int) -> list[NewsItem]:
        try:
            resp = await self._client.search(query, topic="news", max_results=max_results, days=7)
        except Exception as exc:
            raise NewsError(f"Tavily search failed: {exc}") from exc
        items = [
            NewsItem(
                title=(r.get("title") or "").strip(),
                url=r.get("url") or "",
                snippet=" ".join((r.get("content") or "").split())[:500],
                published=r.get("published_date"),
            )
            for r in resp.get("results", [])
            if r.get("title") and r.get("url")
        ]
        if not items:
            raise NewsError("no news results")
        return items


SAMPLE_ITEMS = [
    NewsItem(
        "Sample: Markets weigh interest-rate expectations",
        "https://www.investor.gov",
        "Sample headline for offline demos. Investors often watch central bank signals because "
        "interest rates influence borrowing costs, bond yields and stock valuations.",
    ),
    NewsItem(
        "Sample: Earnings season shapes company stock moves",
        "https://www.investor.gov",
        "Sample headline for offline demos. Quarterly earnings reports update investors on company "
        "profit and outlook, which can move individual stock prices.",
    ),
    NewsItem(
        "Sample: Diversified index funds remain a common core holding",
        "https://www.investor.gov",
        "Sample headline for offline demos. Broad index funds spread risk across many companies and "
        "are widely discussed as a low-cost way to invest.",
    ),
]


class SampleProvider:
    async def search(self, query: str, max_results: int) -> list[NewsItem]:
        return SAMPLE_ITEMS[:max_results]


class NewsService:
    """Cache (TTL) -> provider with retries -> stale cache -> sample items."""

    def __init__(
        self,
        provider: NewsProvider,
        ttl_seconds: int = 1800,
        max_retries: int = 1,
        use_sample_on_failure: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.provider = provider
        self.ttl = ttl_seconds
        self.max_retries = max_retries
        self.use_sample = use_sample_on_failure
        self._clock, self._sleep = clock, sleep
        self._cache: dict[tuple[str, int], tuple[float, list[NewsItem]]] = {}

    async def search(self, query: str, max_results: int = 5) -> NewsResult:
        key, now = (query.strip().lower(), max_results), self._clock()
        cached = self._cache.get(key)
        if cached and now - cached[0] < self.ttl:
            return NewsResult(cached[1], "cache")
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                items = await self.provider.search(query, max_results)
                self._cache[key] = (now, items)
                return NewsResult(items, "live")
            except Exception as exc:
                last = exc
                log.warning("news attempt %d failed: %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    await self._sleep(2**attempt)
        if cached:
            return NewsResult(cached[1], "stale")
        if self.use_sample:
            return NewsResult(await SampleProvider().search(query, max_results), "sample")
        raise NewsError(str(last))
