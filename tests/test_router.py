import pytest

from src.core.llm import FakeLLM
from src.workflow.router import Router, keyword_intents

CATS = {"tax", "basics", "funds"}


class BoomLLM:
    async def complete(self, system, user):
        raise RuntimeError("down")


def router(reply, **kw):
    return Router(FakeLLM(reply), categories=CATS, **kw)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("analyze my portfolio", {"portfolio_analysis"}),
        ("what's the price of AAPL", {"market_analysis"}),
        ("how much should I save to reach $50,000", {"goal_planning"}),
        ("explain a Roth IRA", {"tax_education"}),
        ("latest news on tech", {"news_synthesizer"}),
        ("how is VOO doing and analyze my portfolio", {"market_analysis", "portfolio_analysis"}),
        ("what is compound interest", set()),
    ],
)
def test_keyword_intents(query, expected):
    assert set(keyword_intents(query)) == expected


async def test_llm_route_high_confidence_is_trusted():
    d = await router('{"intents": ["finance_qa"], "confidence": 0.9, "category": "funds"}').route("what is an ETF")
    assert d.intents == ["finance_qa"] and d.method == "llm" and d.category == "funds"


async def test_low_confidence_unions_keyword_intents():
    d = await router('{"intents": ["finance_qa"], "confidence": 0.3}').route("what is my portfolio worth, price of AAPL")
    assert d.method == "llm+keyword"
    assert set(d.intents) == {"finance_qa", "portfolio_analysis", "market_analysis"}


async def test_multi_intent_from_llm_and_dedupe():
    d = await router('{"intents": ["portfolio_analysis", "finance_qa", "finance_qa"], "confidence": 0.95}').route("x")
    assert d.intents == ["portfolio_analysis", "finance_qa"]


async def test_off_topic_alone_and_dropped_when_mixed():
    only = await router('{"intents": ["off_topic"], "confidence": 0.2}').route("who won the game")
    assert only.intents == ["off_topic"]
    mixed = await router('{"intents": ["off_topic", "finance_qa"], "confidence": 0.9}').route("x")
    assert mixed.intents == ["finance_qa"]


async def test_invalid_values_are_sanitised():
    d = await router('{"intents": ["bogus", "goal_planning"], "confidence": "high", "category": "nonsense"}').route("x")
    assert d.intents[0] == "goal_planning" and d.confidence == 0.0 and d.category is None
    s = await router('{"intents": "market_analysis", "confidence": 7}').route("x")
    assert s.intents == ["market_analysis"] and s.confidence == 1.0


async def test_garbage_or_failure_falls_back_to_keywords():
    d = await router("not json at all").route("price of AAPL")
    assert d.intents == ["market_analysis"] and d.method == "keyword"
    d2 = await Router(BoomLLM()).route("explain a roth")
    assert d2.intents == ["tax_education"] and d2.method == "keyword"
    d3 = await Router(BoomLLM()).route("hello there")
    assert d3.intents == ["finance_qa"]


async def test_history_is_included_in_prompt():
    llm = FakeLLM('{"intents": ["goal_planning"], "confidence": 0.9}')
    history = [{"role": "user", "content": "I want 50k in 10 years"}, {"role": "assistant", "content": "plan..."}]
    await Router(llm).route("what about 15 years?", history)
    assert "50k in 10 years" in llm.calls[0][1] and "what about 15 years?" in llm.calls[0][1]
