"""Draft knowledge-base articles from src/data/topics.yaml using the configured LLM.

Usage:  python -m scripts.generate_articles [--limit N] [--force]

Articles are LLM-drafted educational summaries. Each carries `reviewed: false` in its
front-matter until a human has checked it against the reference sources below.
"""
from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

import yaml

from src.core.llm import build_llm
from src.core.settings import ROOT_DIR, get_settings

TOPICS_PATH = ROOT_DIR / "src" / "data" / "topics.yaml"
ARTICLES_DIR = ROOT_DIR / "src" / "data" / "articles"

# Reference publishers used for attribution (site-level URLs only; no deep links are invented).
REFERENCES = {
    "investopedia": ("Investopedia", "https://www.investopedia.com"),
    "investor_gov": ("SEC Investor.gov", "https://www.investor.gov"),
    "bogleheads": ("Bogleheads Wiki", "https://www.bogleheads.org/wiki"),
    "irs": ("IRS", "https://www.irs.gov"),
}
CATEGORY_REFERENCE = {
    "tax": "irs",
    "retirement": "investor_gov",
    "portfolio": "bogleheads",
    "funds": "bogleheads",
    "risks": "investor_gov",
}

SYSTEM = (
    "You write accurate, jargon-free personal-finance education for beginners. "
    "Use widely accepted principles and US context. Never give personalised advice or "
    "recommend specific securities. Define terms on first use. Use concrete numeric examples."
)

PROMPT = """Write a Markdown educational article titled "{title}" (category: {category}).
Requirements:
- 350-550 words, beginner friendly, neutral tone.
- Structure: a short intro, 3-5 sections with '##' headings, a 'Key takeaways' bullet list.
- Include at least one simple numeric example where relevant.
- Do not include a front-matter block or the title as an H1; output the body only.
- Do not invent statistics, laws or limits you are unsure about; keep figures general
  (say 'annual contribution limits change yearly' rather than quoting a number).
"""


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def strip_title(title: str, body: str) -> str:
    """Drop a leading line that just repeats the article title (with or without '#')."""
    lines = body.strip().splitlines()
    if lines and lines[0].lstrip("# ").strip().lower() == title.lower():
        lines = lines[1:]
    return "\n".join(lines).strip()


def render(title: str, category: str, body: str) -> str:
    body = strip_title(title, body)
    ref_key = CATEGORY_REFERENCE.get(category, "investopedia")
    source, url = REFERENCES[ref_key]
    front = {
        "title": title,
        "category": category,
        "source": source,
        "source_url": url,
        "reviewed": False,
    }
    return f"---\n{yaml.safe_dump(front, sort_keys=False).strip()}\n---\n\n{body.strip()}\n"


async def generate(limit: int | None, force: bool, concurrency: int = 5) -> int:
    llm = build_llm(get_settings())
    topics = yaml.safe_load(TOPICS_PATH.read_text())
    jobs = [(cat, t) for cat, items in topics.items() for t in items]
    if limit:
        jobs = jobs[:limit]
    sem = asyncio.Semaphore(concurrency)
    written = 0

    async def one(category: str, title: str) -> None:
        nonlocal written
        path = ARTICLES_DIR / category / f"{slugify(title)}.md"
        if path.exists() and not force:
            return
        async with sem:
            body = await llm.complete(SYSTEM, PROMPT.format(title=title, category=category))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(title, category, body))
        written += 1
        print(f"wrote {path.relative_to(ROOT_DIR)}")

    results = await asyncio.gather(*(one(c, t) for c, t in jobs), return_exceptions=True)
    for (cat, title), res in zip(jobs, results):
        if isinstance(res, Exception):
            print(f"FAILED {cat}/{title}: {res}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(f"done: {asyncio.run(generate(args.limit, args.force))} articles written")
