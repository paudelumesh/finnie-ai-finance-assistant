import asyncio

import pytest

from src.agents.finance_qa import FinanceQAAgent
from src.agents.goal_planning import GoalPlanningAgent
from src.agents.market_analysis import MarketAnalysisAgent
from src.agents.portfolio_analysis import PortfolioAnalysisAgent
from src.core.base_agent import DISCLAIMER, AgentResponse, BaseAgent
from src.core.llm import FakeLLM
from src.data.market_data import MarketDataService, MockProvider
from src.rag.chunker import chunk_articles
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.workflow.graph import OFF_TOPIC_MESSAGE, FinnieWorkflow
from src.workflow.router import Router
from tests.test_rag import ARTICLES


class SmartLLM:
    """Plays every LLM role by inspecting the system prompt."""

    def __init__(self, route, goal='{"target": 20000, "years": 5}'):
        self.route, self.goal, self.calls = route, goal, []

    async def complete(self, system, user):
        self.calls.append((system, user))
        if "router for Finnie" in system:
            return self.route
        if "Extract financial goal parameters" in system:
            return self.goal
        if "Combine the specialist answers" in system:
            return "MERGED ANSWER"
        return "agent answer"


async def no_sleep(_):
    pass


def build(route, **kw):
    llm = SmartLLM(route)
    store = VectorStore.build(chunk_articles(ARTICLES, 400, 50), HashEmbedder())
    market = MarketDataService(MockProvider(), sleep=no_sleep)
    agents = {
        "finance_qa": FinanceQAAgent(llm, Retriever(store, 3, 0.1)),
        "portfolio_analysis": PortfolioAnalysisAgent(llm, market),
        "market_analysis": MarketAnalysisAgent(llm, market),
        "goal_planning": GoalPlanningAgent(llm),
    }
    agents.update(kw.pop("extra_agents", {}))
    router = Router(llm, categories=set(store.categories))
    return FinnieWorkflow(llm, agents, router, **kw), llm


def route_json(*intents, conf=0.95, category=None):
    import json
    return json.dumps({"intents": list(intents), "confidence": conf, "category": category})


async def test_single_intent_goes_to_one_agent_and_keeps_disclaimer_once():
    wf, llm = build(route_json("finance_qa"))
    r = await wf.chat("what is compound interest?", "t1")
    assert r.agents == ["finance_qa"] and r.content.count(DISCLAIMER) == 1
    assert "agent answer" in r.content and r.sources[0]["title"] == "Compound interest"
    assert not any("Combine the specialist" in s for s, _ in llm.calls)


async def test_multi_intent_fans_out_and_merges():
    wf, llm = build(route_json("finance_qa", "market_analysis"))
    r = await wf.chat("explain ETFs and how is VOO doing", "t2")
    assert r.agents == ["finance_qa", "market_analysis"]
    assert r.content.startswith("MERGED ANSWER") and r.content.count(DISCLAIMER) == 1
    merge_prompt = next(u for s, u in llm.calls if "Combine the specialist" in s)
    assert "### Finance Qa" in merge_prompt and DISCLAIMER not in merge_prompt


async def test_off_topic_is_refused_without_agents():
    wf, llm = build(route_json("off_topic", conf=0.9))
    r = await wf.chat("who won the world cup", "t3")
    assert r.content == OFF_TOPIC_MESSAGE and r.agents == ["router"]
    assert all("router for Finnie" in s for s, _ in llm.calls)


async def test_tax_is_remapped_to_qa_with_category_filter():
    wf, llm = build(route_json("tax_education"))
    r = await wf.chat("explain a roth ira", "t4")
    assert r.route["intents"] == ["finance_qa"] and r.route["category"] == "tax"
    assert r.agents == ["finance_qa"]


async def test_unavailable_only_intent_gets_clear_message():
    wf, _ = build(route_json("news_synthesizer"))
    r = await wf.chat("latest headlines", "t5")
    assert "isn't available yet" in r.content and "news synthesizer" in r.content
    assert r.route["unavailable"] == ["news_synthesizer"]


async def test_new_agent_registered_later_is_used():
    class NewsAgent(BaseAgent):
        name = "news_synthesizer"

        async def _handle(self, query, context):
            return AgentResponse(self.name, "NEWS SUMMARY")

    wf, _ = build(route_json("news_synthesizer"), extra_agents={"news_synthesizer": NewsAgent(FakeLLM())})
    r = await wf.chat("latest headlines", "t6")
    assert r.content.startswith("NEWS SUMMARY") and r.agents == ["news_synthesizer"]


async def test_memory_across_turns_and_isolated_threads():
    wf, llm = build(route_json("finance_qa"))
    await wf.chat("what is an ETF?", "A")
    await wf.chat("and a mutual fund?", "A")
    await wf.chat("something else", "B")
    assert [m["role"] for m in wf.history("A")] == ["user", "assistant", "user", "assistant"]
    assert len(wf.history("B")) == 2 and wf.history("nope") == []
    classify_prompts = [u for s, u in llm.calls if "router for Finnie" in s]
    assert "what is an ETF?" in classify_prompts[1]  # turn 2 sees turn 1
    assert "what is an ETF?" not in classify_prompts[2]  # thread B does not


async def test_only_current_turn_responses_are_reported():
    wf, _ = build(route_json("finance_qa"))
    await wf.chat("one", "T")
    r = await wf.chat("two", "T")
    assert r.turn == 2 and len(r.responses) == 1 and r.responses[0]["turn"] == 2


async def test_context_and_profile_reach_agents():
    wf, llm = build(route_json("portfolio_analysis"))
    r = await wf.chat("analyse my portfolio", "t7",
                      profile={"risk_tolerance": "low"},
                      context={"portfolio": [{"symbol": "VOO", "shares": 2}]})
    assert r.responses[0]["data"]["analysis"]["total_value"] > 0
    wf2, _ = build(route_json("goal_planning"))
    r2 = await wf2.chat("plan", "t8", profile={"risk_tolerance": "low"}, context={"goal": {"target": 10000, "years": 4}})
    assert r2.responses[0]["data"]["plan"]["risk"] == "low"


async def test_profile_persists_across_turns_when_not_resent():
    wf, _ = build(route_json("goal_planning"))
    await wf.chat("plan", "P", profile={"risk_tolerance": "high"})
    r = await wf.chat("plan again", "P")
    assert r.responses[0]["data"]["plan"]["risk"] == "high"


async def test_one_failing_agent_does_not_sink_the_answer():
    class Broken(BaseAgent):
        name = "market_analysis"

        async def _handle(self, query, context):
            raise ValueError("kaboom")

    wf, _ = build(route_json("finance_qa", "market_analysis"), extra_agents={"market_analysis": Broken(FakeLLM())})
    r = await wf.chat("explain ETFs and price of VOO", "t10")
    assert "agent answer" in r.content and "couldn't complete part" in r.content and "market_analysis" in r.content


async def test_all_agents_failing_returns_friendly_fallback():
    class Broken(BaseAgent):
        name = "finance_qa"

        async def _handle(self, query, context):
            raise ValueError("kaboom")

    wf, _ = build(route_json("finance_qa"), extra_agents={"finance_qa": Broken(FakeLLM())})
    r = await wf.chat("anything", "t11")
    assert r.content.startswith(BaseAgent.fallback_message)


async def test_slow_agent_times_out():
    class Slow(BaseAgent):
        name = "finance_qa"

        async def _handle(self, query, context):
            await asyncio.sleep(5)

    wf, _ = build(route_json("finance_qa"), extra_agents={"finance_qa": Slow(FakeLLM())}, agent_timeout=0.05)
    r = await wf.chat("anything", "t12")
    assert "too long" in r.content and r.responses[0]["error"] == "timeout"


async def test_merge_failure_falls_back_to_concatenation():
    wf, llm = build(route_json("finance_qa", "market_analysis"))

    async def boom(system, user):
        if "Combine the specialist" in system:
            raise RuntimeError("merge down")
        return await SmartLLM.complete(llm, system, user)

    wf.llm = type("L", (), {"complete": staticmethod(boom)})()
    r = await wf.chat("explain ETFs and price of VOO", "t13")
    assert "### Finance Qa" in r.content and "### Market Analysis" in r.content and DISCLAIMER in r.content


async def test_non_json_context_objects_are_made_checkpoint_safe():
    from src.core.portfolio import Holding
    wf, _ = build(route_json("portfolio_analysis"))
    r = await wf.chat("analyse my portfolio", "t14", context={"portfolio": [Holding("VOO", 2)]})
    assert r.responses[0]["data"]["analysis"]["total_value"] > 0


async def test_guardrail_runs_in_graph_and_memory_stores_guarded_text():
    wf, _ = build(route_json("finance_qa"))
    r = await wf.chat("Should I buy AAPL?", "g1")
    assert r.guardrail["flagged"] and r.content.startswith("**I can't tell you what to buy")
    assert wf.history("g1")[-1]["content"] == r.content


async def test_clean_answer_not_flagged_and_off_topic_has_no_disclaimer():
    wf, _ = build(route_json("finance_qa"))
    assert not (await wf.chat("what is an ETF?", "g2")).guardrail["flagged"]
    wf2, _ = build(route_json("off_topic", conf=0.9))
    r = await wf2.chat("who won the game", "g3")
    assert r.content == OFF_TOPIC_MESSAGE and DISCLAIMER not in r.content
