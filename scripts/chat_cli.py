"""Terminal chat with the full Finnie workflow.

Usage:
  python -m scripts.chat_cli                       # interactive
  python -m scripts.chat_cli "question 1" "question 2"   # scripted multi-turn
Options: --risk low|moderate|high   --portfolio holdings.csv
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.core.portfolio import parse_holdings_csv
from src.core.settings import get_settings
from src.utils.logging import setup_logging
from src.workflow.factory import build_workflow


async def main(queries: list[str], risk: str | None, portfolio_csv: str | None) -> None:
    setup_logging("WARNING")
    wf = build_workflow(get_settings())
    profile = {"risk_tolerance": risk} if risk else None
    context = {}
    if portfolio_csv:
        context["portfolio"] = parse_holdings_csv(Path(portfolio_csv).read_text())

    async def ask(q: str) -> None:
        r = await wf.chat(q, "cli", profile=profile, context=context)
        print(f"\n[route: {r.route['intents']} via {r.route['method']}, conf {r.route['confidence']:.2f}]")
        print(r.content)
        if r.sources:
            print("Sources:", "; ".join(s["title"] for s in r.sources))

    if queries:
        for q in queries:
            print(f"\n>>> {q}")
            await ask(q)
        return
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            await ask(q)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*")
    ap.add_argument("--risk")
    ap.add_argument("--portfolio")
    a = ap.parse_args()
    asyncio.run(main(a.queries, a.risk, a.portfolio))
