"""Output guardrail: keeps answers educational (not advice) and makes the disclaimer certain."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.base_agent import DISCLAIMER

# The user is asking for a personal recommendation.
ADVICE_QUERY = re.compile(
    r"\b(should i (buy|sell|invest|put|hold|switch|move|pick|choose)|"
    r"(what|which) (stocks?|etfs?|funds?|crypto\w*|investments?) (should|do you recommend|would you)|"
    r"(is|are) \S+( \S+){0,2} (a )?(good|bad|safe|smart) (buy|investment|stock|etf|bet|time to)|"
    r"(tell|advise|recommend) me (what|which|where)|"
    r"(do you|would you) (recommend|suggest)|"
    r"where should i (put|invest)|best (stock|etf|fund|investment)s? (to|for) (buy|me)|"
    r"(should|can) i (afford|go all in))\b",
    re.I,
)
# The generated answer itself sounds like a personal recommendation.
ADVICE_ANSWER = re.compile(
    r"\b(you should (buy|sell|invest in|hold|put|move|switch|pick|choose|open)|"
    r"i (recommend|suggest|advise) (that you |you )?(buy|sell|investing|buying|selling|holding|putting)|"
    r"(buy|sell) (it|this|that|now)\b|"
    r"is a (great|strong|good) (buy|investment)|"
    r"your best (option|move|choice) is)\b",
    re.I,
)

EDUCATION_NOTE = (
    "**I can't tell you what to buy, sell or hold.** That depends on your goals, timeline, "
    "taxes and risk tolerance, and I'm an educational tool, not an advisor. "
    "Here is how to think about it instead:\n\n"
)

REWRITE_SYSTEM = (
    "Rewrite the answer so it contains NO personal recommendation. Replace any 'you should "
    "buy/sell/hold' style statements with neutral education: the factors to weigh, trade-offs, "
    "and questions to ask a licensed advisor. Keep all numbers, citations like [1] and facts "
    "unchanged. Do not add a disclaimer. Return only the rewritten answer."
)


@dataclass
class GuardrailResult:
    content: str
    flagged: bool = False
    reasons: list[str] = field(default_factory=list)


def asks_for_advice(query: str) -> bool:
    return bool(ADVICE_QUERY.search(query))


def sounds_like_advice(text: str) -> bool:
    return bool(ADVICE_ANSWER.search(text))


def ensure_disclaimer(content: str) -> str:
    return content if DISCLAIMER in content else f"{content.rstrip()}\n\n_{DISCLAIMER}_"


async def apply_guardrail(query: str, content: str, llm, add_disclaimer: bool = True) -> GuardrailResult:
    reasons: list[str] = []
    body = content

    if sounds_like_advice(body):
        reasons.append("answer_contained_recommendation")
        stripped = body.replace(f"_{DISCLAIMER}_", "").strip()
        try:
            rewritten = (await llm.complete(REWRITE_SYSTEM, stripped)).strip()
        except Exception:
            rewritten = ""
        # Trust the rewrite only if it is non-empty and actually removed the advice.
        ok = rewritten and not sounds_like_advice(rewritten)
        body = rewritten if ok else _drop_advice_sentences(stripped)

    if asks_for_advice(query):
        reasons.append("user_requested_advice")
        body = EDUCATION_NOTE + body

    if add_disclaimer:
        body = ensure_disclaimer(body)
    return GuardrailResult(body, bool(reasons), reasons)


def _drop_advice_sentences(text: str) -> str:
    kept = [s for s in re.split(r"(?<=[.!?])\s+", text) if not ADVICE_ANSWER.search(s)]
    return " ".join(kept).strip() or "I can't make a personal recommendation."
