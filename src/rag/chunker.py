"""Split articles into overlapping, section-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.rag.loader import Article


@dataclass(frozen=True)
class Chunk:
    text: str
    title: str
    category: str
    source: str
    source_url: str | None
    slug: str
    index: int


def _split_sections(body: str) -> list[str]:
    parts = re.split(r"(?m)^(?=## )", body)
    return [p.strip() for p in parts if p.strip()]


def _window(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    step = max(size - overlap, 1)
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # prefer to break on whitespace
            ws = text.rfind(" ", start + step // 2, end)
            end = ws if ws > start else end
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def chunk_article(article: Article, size: int = 800, overlap: int = 120) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks: list[Chunk] = []
    for section in _split_sections(article.body):
        for piece in _window(section, size, overlap):
            chunks.append(
                Chunk(
                    text=piece,
                    title=article.title,
                    category=article.category,
                    source=article.source,
                    source_url=article.source_url,
                    slug=article.slug,
                    index=len(chunks),
                )
            )
    return chunks


def chunk_articles(articles: list[Article], size: int = 800, overlap: int = 120) -> list[Chunk]:
    return [c for a in articles for c in chunk_article(a, size, overlap)]
