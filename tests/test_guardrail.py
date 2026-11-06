import pytest

from src.core.base_agent import DISCLAIMER
from src.core.llm import FakeLLM
from src.workflow.guardrail import (
    EDUCATION_NOTE, apply_guardrail, asks_for_advice, ensure_disclaimer, sounds_like_advice,
)


@pytest.mark.parametrize("q", [
    "Should I buy AAPL?", "should i sell my bonds", "What stocks should I buy?",
    "Is Tesla a good investment?", "is a roth ira a good investment", "Where should I invest $5000?",
    "Do you recommend VOO?", "tell me what to invest in",
])
def test_advice_queries_detected(q):
    assert asks_for_advice(q)


@pytest.mark.parametrize("q", [
    "What is an ETF?", "How does compound interest work?", "Explain a Roth IRA",
    "How is AAPL doing today?", "what is diversification",
])
def test_educational_queries_not_flagged(q):
    assert not asks_for_advice(q)


@pytest.mark.parametrize("t", [
    "You should buy more bonds.", "I recommend buying VOO.", "Sell it now.", "Your best option is gold.",
])
def test_advice_answers_detected(t):
    assert sounds_like_advice(t)


def test_neutral_answers_not_flagged():
    assert not sounds_like_advice("Investors often weigh fees, risk and time horizon [1].")


def test_ensure_disclaimer_is_idempotent():
    once = ensure_disclaimer("hello")
    assert once.count(DISCLAIMER) == 1 and ensure_disclaimer(once) == once


async def test_clean_answer_only_gets_disclaimer():
    r = await apply_guardrail("what is an ETF", "An ETF is a fund [1].", FakeLLM("unused"))
    assert not r.flagged and r.content.count(DISCLAIMER) == 1


async def test_advice_request_prepends_educational_note():
    r = await apply_guardrail("Should I buy AAPL?", "Apple is a large company.", FakeLLM("unused"))
    assert r.flagged and r.reasons == ["user_requested_advice"]
    assert r.content.startswith(EDUCATION_NOTE) and r.content.count(DISCLAIMER) == 1


async def test_advice_in_answer_is_rewritten_by_llm():
    llm = FakeLLM("Investors weigh fees and risk before choosing bonds.")
    r = await apply_guardrail("explain bonds", "You should buy more bonds.", llm)
    assert "answer_contained_recommendation" in r.reasons
    assert "should buy" not in r.content and "weigh fees" in r.content
    assert "You should buy" in llm.calls[0][1]  # rewrite saw the original


async def test_bad_rewrite_falls_back_to_dropping_sentences():
    llm = FakeLLM("You should buy bonds anyway.")  # rewrite still gives advice
    r = await apply_guardrail("x", f"Bonds pay interest. You should buy bonds. _{DISCLAIMER}_", llm)
    assert "Bonds pay interest." in r.content and "should buy" not in r.content


async def test_empty_or_failing_rewrite_is_handled():
    class Boom:
        async def complete(self, *a):
            raise RuntimeError("down")

    r = await apply_guardrail("x", "You should buy gold.", Boom())
    assert "should buy" not in r.content and r.content.count(DISCLAIMER) == 1
    only = await apply_guardrail("x", "You should buy gold.", FakeLLM(""))
    assert "personal recommendation" in only.content


async def test_disclaimer_can_be_skipped():
    r = await apply_guardrail("who won", "I only cover finance.", FakeLLM(), add_disclaimer=False)
    assert DISCLAIMER not in r.content
