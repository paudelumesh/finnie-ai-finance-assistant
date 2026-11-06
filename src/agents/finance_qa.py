"""Finance Q&A agent: answers beginner questions grounded in the RAG knowledge base."""
from __future__ import annotations

from typing import Any

from src.core.base_agent import DISCLAIMER, AgentResponse, BaseAgent
from src.core.llm import LLMClient
from src.rag.retriever import Retriever

GROUNDED_SYSTEM = (
    "You are Finnie, a friendly financial education tutor for beginners. "
    "Answer ONLY using the numbered passages provided. Cite passages inline like [1]. "
    "Explain jargon simply and keep answers under 250 words. "
    "Never give personalised advice or recommend buying or selling specific securities; "
    "if asked, explain the concepts to consider instead. "
    "If the passages do not contain the answer, say so plainly."
)

UNGROUNDED_SYSTEM = (
    "You are Finnie, a friendly financial education tutor for beginners. "
    "The knowledge base had no matching article. If the question is NOT about personal "
    "finance, investing, taxes or the economy, politely say you only cover finance topics "
    "and answer nothing else. Otherwise give a short, general, widely accepted explanation "
    "(under 150 words), start by saying this answer is not from the knowledge base, and "
    "never give personalised advice or recommend specific securities."
)

NO_KB_NOTE = "_Note: no matching knowledge-base article was found, so this is general information._"


class FinanceQAAgent(BaseAgent):
    name = "finance_qa"
    description = "Explains general finance and investing concepts using the knowledge base."

    def __init__(self, llm: LLMClient, retriever: Retriever) -> None:
        super().__init__(llm)
        self.retriever = retriever

    async def _handle(self, query: str, context: dict[str, Any]) -> AgentResponse:
        result = self.retriever.retrieve(query, category=context.get("category"))
        level = context.get("profile", {}).get("knowledge_level")
        level_hint = f"\nThe learner's level is: {level}." if level else ""

        if result.found:
            user = f"Passages:\n{result.context_block()}\n\nQuestion: {query}{level_hint}"
            answer = await self.llm.complete(GROUNDED_SYSTEM, user)
            sources = result.sources()
        else:
            answer = await self.llm.complete(UNGROUNDED_SYSTEM, f"Question: {query}{level_hint}")
            answer = f"{answer}\n\n{NO_KB_NOTE}"
            sources = []

        return AgentResponse(
            agent=self.name,
            content=f"{answer}\n\n_{DISCLAIMER}_",
            sources=sources,
            data={"grounded": result.found, "scores": [round(h.score, 3) for h in result.hits]},
        )
