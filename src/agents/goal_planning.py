"""Goal Planning agent: extracts goal parameters, computes projections in code, narrates results."""
from __future__ import annotations

from typing import Any

from src.core.base_agent import DISCLAIMER, AgentResponse, BaseAgent
from src.core.goals import GoalError, GoalPlan, plan_goal
from src.core.llm import LLMClient
from src.utils.json_utils import extract_json

EXTRACT_SYSTEM = (
    "Extract financial goal parameters from the user's message. Reply with ONLY a JSON object "
    'with keys: "target" (number, USD), "years" (number), "current" (number, existing savings), '
    '"monthly" (number or null, planned monthly saving), "risk" ("low"|"moderate"|"high"|null). '
    "Use null for anything not stated. Do not guess. If earlier user messages are shown, this may "
    "be a follow-up: carry over any parameter the latest message does not change "
    '(e.g. "what about 12 years?" keeps the earlier target and savings).'
)

NARRATE_SYSTEM = (
    "You are Finnie, a friendly goal-planning educator. Explain the plan below using ONLY its "
    "numbers (computed by code; do not recompute). Mention the three return scenarios are "
    "illustrative assumptions, not forecasts or guarantees, and that real returns vary. "
    "Do not recommend specific investments. Keep it under 220 words."
)

FIELDS = ("target", "years", "current", "monthly", "risk")


def _num(value: Any) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def summarise(p: GoalPlan) -> str:
    lines = [
        f"Goal: ${p.target:,.0f} in {p.years:g} years; current savings ${p.current:,.0f}; "
        f"risk profile: {p.risk}",
        f"Monthly contribution used: ${p.contribution_used:,.2f} (total put in: ${p.total_contributed:,.0f})",
        f"On track with expected scenario: {'yes' if p.on_track else 'no'}",
    ]
    for s in p.scenarios:
        lines.append(
            f"- {s.name} ({s.annual_return:.0%}/yr): would need ${s.monthly_needed:,.2f}/month to hit "
            f"the target exactly; saving the used ${p.contribution_used:,.2f}/month instead "
            f"grows to ${s.projected_value:,.0f}"
        )
    return "\n".join(lines)


class GoalPlanningAgent(BaseAgent):
    name = "goal_planning"
    description = "Calculates how much to save monthly to reach a target, by risk profile."

    async def _extract(self, query: str, history: list[dict]) -> dict[str, Any]:
        earlier = [m["content"][:300] for m in history if m.get("role") == "user"][-3:]
        prompt = (
            ("Earlier user messages:\n" + "\n".join(f"- {m}" for m in earlier) + "\n\n") if earlier else ""
        ) + f"Latest message: {query}"
        parsed = extract_json(await self.llm.complete(EXTRACT_SYSTEM, prompt)) or {}
        return {k: parsed.get(k) for k in FIELDS}

    async def _handle(self, query: str, context: dict[str, Any]) -> AgentResponse:
        params = {k: v for k, v in (context.get("goal") or {}).items() if v is not None}
        if not {"target", "years"} <= params.keys():
            extracted = await self._extract(query, context.get("history") or [])
            params = {**{k: v for k, v in extracted.items() if v is not None}, **params}
        risk = params.get("risk") or context.get("profile", {}).get("risk_tolerance")

        target, years = _num(params.get("target")), _num(params.get("years"))
        missing = [n for n, v in (("target amount", target), ("time horizon in years", years)) if v is None]
        if missing:
            return AgentResponse(
                self.name,
                "To build a plan I need your " + " and ".join(missing)
                + ". Optionally tell me your current savings, monthly budget and comfort with risk.",
                data={"needs": missing},
            )
        try:
            plan = plan_goal(target, years, _num(params.get("current")) or 0.0, risk, _num(params.get("monthly")))
        except GoalError as exc:
            return AgentResponse(self.name, f"I couldn't build that plan: {exc}", error="invalid_goal")

        answer = await self.llm.complete(NARRATE_SYSTEM, f"User question: {query}\n\nPlan:\n{summarise(plan)}")
        return AgentResponse(
            agent=self.name, content=f"{answer}\n\n_{DISCLAIMER}_", data={"plan": plan}
        )
