"""Wire the real components (LLM, market data, RAG, agents, router) into a workflow."""
from __future__ import annotations

from typing import Any

from src.agents.finance_qa import FinanceQAAgent
from src.agents.goal_planning import GoalPlanningAgent
from src.agents.market_analysis import MarketAnalysisAgent
from src.agents.portfolio_analysis import PortfolioAnalysisAgent
from src.core.llm import LLMClient, build_llm
from src.core.settings import Settings, get_settings
from src.data.market_data import MarketDataService, QuoteProvider, YFinanceProvider
from src.rag.embeddings import Embedder, SentenceTransformerEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.workflow.graph import FinnieWorkflow
from src.workflow.router import Router


def build_workflow(
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    provider: QuoteProvider | None = None,
    embedder: Embedder | None = None,
    checkpointer: Any = None,
) -> FinnieWorkflow:
    settings = settings or get_settings()
    llm = llm or build_llm(settings)
    market = MarketDataService(
        provider or YFinanceProvider(),
        ttl_seconds=settings.market_data.cache_ttl_seconds,
        use_mock_on_failure=settings.market_data.use_mock_on_failure,
    )
    store = VectorStore.load(embedder or SentenceTransformerEmbedder(settings.rag.embedding_model))
    retriever = Retriever(store, settings.rag.top_k, settings.rag.min_score)

    agents = {
        "finance_qa": FinanceQAAgent(llm, retriever),
        "portfolio_analysis": PortfolioAnalysisAgent(llm, market),
        "market_analysis": MarketAnalysisAgent(llm, market),
        "goal_planning": GoalPlanningAgent(llm),
    }
    router = Router(
        llm,
        settings.router.confidence_threshold,
        settings.router.classifier_timeout_seconds,
        set(store.categories),
    )
    return FinnieWorkflow(llm, agents, router, settings.router.agent_timeout_seconds, checkpointer)
