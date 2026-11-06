"""Tax Education agent: explains tax concepts and account types (never personal tax advice)."""
from __future__ import annotations

from typing import Any

from src.core.base_agent import DISCLAIMER, AgentResponse, BaseAgent
from src.core.llm import LLMClient
from src.rag.retriever import Retriever

PREFERRED = ("tax", "retirement")

SYSTEM = (
    "You are Finnie, a tax-education tutor for beginners. Answer ONLY from the numbered passages, "
    "citing them like [1]. Explain how account types and tax concepts work in general terms "
    "(US context). Contribution limits, income thresholds and rules change yearly, so do not quote "
    "specific figures unless a passage states them, and say where to check current numbers "
    "(irs.gov). NEVER tell the user what they personally owe, which account they should use, or how "
    "to file; explain the trade-offs instead and suggest a tax professional for personal decisions. "
    "If the passages do not cover the question, say so plainly. Keep it under 250 words."
)

NO_KB = (
    "I couldn't find tax material on that in my knowledge base. For specific tax questions, the "
    "IRS website (irs.gov) or a licensed tax professional is the right place to check."
)


class TaxEducationAgent(BaseAgent):
    name = "tax_education"
    description = "Explains tax concepts and account types (401(k), IRA, Roth, capital gains)."

    def __init__(self, llm: LLMClient, retriever: Retriever) -> None:
        super().__init__(llm)
        self.retriever = retriever

    async def _handle(self, query: str, context: dict[str, Any]) -> AgentResponse:
        result = self.retriever.retrieve_preferring(query, PREFERRED)
        if not result.found:
            return AgentResponse(self.name, f"{NO_KB}\n\n_{DISCLAIMER}_", data={"grounded": False})
        answer = await self.llm.complete(
            SYSTEM, f"Passages:\n{result.context_block()}\n\nQuestion: {query}"
        )
        return AgentResponse(
            agent=self.name,
            content=f"{answer}\n\n_{DISCLAIMER}_",
            sources=result.sources(),
            data={"grounded": True, "scores": [round(h.score, 3) for h in result.hits]},
        )
