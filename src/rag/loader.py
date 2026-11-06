"""Load Markdown articles (YAML front-matter + body) from the knowledge base."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.core.settings import ROOT_DIR

ARTICLES_DIR = ROOT_DIR / "src" / "data" / "articles"


@dataclass(frozen=True)
class Article:
    slug: str
    title: str
    category: str
    source: str
    source_url: str | None
    body: str
    reviewed: bool = False


def parse_article(path: Path) -> Article:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path.name}: missing front-matter")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front) or {}
    for key in ("title", "category"):
        if key not in meta:
            raise ValueError(f"{path.name}: front-matter missing '{key}'")
    return Article(
        slug=path.stem,
        title=str(meta["title"]),
        category=str(meta["category"]),
        source=str(meta.get("source", "Unknown")),
        source_url=meta.get("source_url"),
        body=body.strip(),
        reviewed=bool(meta.get("reviewed", False)),
    )


def load_articles(directory: Path = ARTICLES_DIR) -> list[Article]:
    return [parse_article(p) for p in sorted(Path(directory).rglob("*.md"))]
