"""Retriever: applies settings (top_k, min_score) and shapes hits into prompt context + sources."""
from __future__ import annotations

from dataclasses import dataclass

from src.core.base_agent import Source
from src.rag.vector_store import SearchHit, VectorStore


@dataclass
class RetrievalResult:
    hits: list[SearchHit]

    @property
    def found(self) -> bool:
        return bool(self.hits)

    def context_block(self) -> str:
        """Numbered passages the LLM can cite as [1], [2], ..."""
        return "\n\n".join(
            f"[{i}] ({h.chunk.title}) {h.chunk.text}" for i, h in enumerate(self.hits, start=1)
        )

    def sources(self) -> list[Source]:
        seen: set[str] = set()
        out: list[Source] = []
        for h in self.hits:
            if h.chunk.slug in seen:
                continue
            seen.add(h.chunk.slug)
            out.append(Source(h.chunk.title, h.chunk.source_url, h.chunk.category))
        return out


class Retriever:
    def __init__(self, store: VectorStore, top_k: int = 4, min_score: float = 0.25) -> None:
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(
        self, query: str, category: str | None = None, strict: bool = False, margin: float = 0.05
    ) -> RetrievalResult:
        """`strict` filters hard (user-chosen category). Otherwise `category` is only a hint
        (e.g. guessed by the router) and is used only if it doesn't hide a clearly better match."""
        if category and strict:
            return RetrievalResult(self.store.search(query, self.top_k, category, self.min_score))
        hits = self.store.search(query, self.top_k, None, self.min_score)
        if category:
            preferred = self.store.search(query, self.top_k, category, self.min_score)
            if preferred and (not hits or preferred[0].score >= hits[0].score - margin):
                hits = preferred
        return RetrievalResult(hits)

    def retrieve_preferring(
        self, query: str, categories: tuple[str, ...], bonus: float = 0.06
    ) -> RetrievalResult:
        """Rank unfiltered hits with a small score bonus for preferred categories."""
        pool = self.store.search(query, self.top_k * 3, None, self.min_score)
        ranked = sorted(
            pool, key=lambda h: h.score + (bonus if h.chunk.category in categories else 0), reverse=True
        )
        return RetrievalResult(ranked[: self.top_k])
