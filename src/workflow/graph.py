"""LangGraph workflow: router -> parallel agent fan-out -> synthesizer.

    START -> router -(Send per intent)-> agent_<name> ... -> synthesize -> END
                    \\-(off-topic / nothing available)-> refuse -> synthesize
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.core.base_agent import DISCLAIMER, AgentResponse, BaseAgent
from src.core.llm import LLMClient
from src.utils.logging import get_logger
from src.workflow.guardrail import apply_guardrail
from src.workflow.router import AGENT_INTENTS, OFF_TOPIC, RouteDecision, Router
from src.workflow.state import FinnieState, jsonable, response_to_dict

log = get_logger("workflow")

# Intents without a dedicated agent yet are served by an existing one (agent, category).
REMAP: dict[str, tuple[str, str | None]] = {"tax_education": ("finance_qa", "tax")}
HISTORY_WINDOW = 6

OFF_TOPIC_MESSAGE = (
    "I'm Finnie, a finance education assistant, so I can only help with personal finance, "
    "investing, taxes and the economy. Try asking about concepts like ETFs, a Roth IRA, "
    "your portfolio, a stock quote, or a savings goal."
)
UNAVAILABLE_MESSAGE = "That part of your request needs a feature that isn't available yet: {names}."

MERGE_SYSTEM = (
    "You are Finnie, a friendly financial education tutor. Combine the specialist answers below "
    "into ONE cohesive reply to the user's question. Keep every number, citation like [1], and "
    "caveat intact; do not add facts. Use short headings where helpful and remove repetition. "
    "Do not add a disclaimer. Never recommend buying or selling specific securities."
)


def strip_disclaimer(text: str) -> str:
    return text.replace(f"_{DISCLAIMER}_", "").strip()


@dataclass
class ChatResult:
    content: str
    sources: list[dict]
    agents: list[str]
    route: dict[str, Any]
    guardrail: dict[str, Any] = field(default_factory=dict)
    responses: list[dict] = field(default_factory=list)
    thread_id: str = "default"
    turn: int = 0


class FinnieWorkflow:
    def __init__(
        self,
        llm: LLMClient,
        agents: dict[str, BaseAgent],
        router: Router,
        agent_timeout: float = 60,
        checkpointer: Any = None,
    ) -> None:
        self.llm = llm
        self.agents = agents
        self.router = router
        self.agent_timeout = agent_timeout
        self.graph = self._build(checkpointer or MemorySaver())

    # ---- graph construction -------------------------------------------------
    def _build(self, checkpointer: Any):
        g = StateGraph(FinnieState)
        g.add_node("router", self._router_node)
        g.add_node("refuse", self._refuse_node)
        g.add_node("synthesize", self._synthesize_node)
        g.add_node("guardrail", self._guardrail_node)
        destinations = ["refuse"]
        for name in self.agents:
            node = f"agent_{name}"
            g.add_node(node, self._make_agent_node(name))
            g.add_edge(node, "synthesize")
            destinations.append(node)
        g.add_edge(START, "router")
        g.add_conditional_edges("router", self._dispatch, destinations)
        g.add_edge("refuse", "synthesize")
        g.add_edge("synthesize", "guardrail")
        g.add_edge("guardrail", END)
        return g.compile(checkpointer=checkpointer)

    # ---- nodes ----------------------------------------------------------------
    def _resolve(self, decision: RouteDecision) -> RouteDecision:
        """Map intents onto agents that actually exist; record what can't be served."""
        intents: list[str] = []
        unavailable: list[str] = []
        category = decision.category
        remap_category: str | None = None
        for intent in decision.intents:
            if intent == OFF_TOPIC or intent in self.agents:
                intents.append(intent)
            elif intent in REMAP and REMAP[intent][0] in self.agents:
                target, cat = REMAP[intent]
                intents.append(target)
                if target not in decision.intents:
                    remap_category = cat
            else:
                unavailable.append(intent)
        intents = list(dict.fromkeys(intents))
        category = category or remap_category
        return RouteDecision(intents, decision.confidence, decision.method, category, unavailable)

    async def _router_node(self, state: FinnieState) -> dict:
        decision = self._resolve(await self.router.route(state["query"], state.get("messages", [])))
        log.info("route: %s (conf %.2f, %s)", decision.intents, decision.confidence, decision.method)
        return {"turn": state.get("turn", 0) + 1, "route": decision.to_dict()}

    def _dispatch(self, state: FinnieState) -> list[Send]:
        targets = [i for i in state["route"]["intents"] if i in self.agents]
        if not targets:
            return [Send("refuse", state)]
        return [Send(f"agent_{i}", {**state, "intent": i}) for i in targets]

    def _agent_context(self, state: FinnieState, name: str) -> dict[str, Any]:
        ctx = dict(state.get("context") or {})
        ctx["profile"] = state.get("profile") or {}
        ctx["history"] = (state.get("messages") or [])[-HISTORY_WINDOW:]
        if name == "finance_qa" and state["route"].get("category"):
            ctx["category"] = state["route"]["category"]
        return ctx

    def _make_agent_node(self, name: str):
        async def node(state: FinnieState) -> dict:
            turn = state["turn"]
            try:
                resp = await asyncio.wait_for(
                    self.agents[name].run(state["query"], self._agent_context(state, name)),
                    self.agent_timeout,
                )
            except asyncio.TimeoutError:
                log.warning("agent %s timed out", name)
                resp = AgentResponse(name, "That took too long. Please try again.", error="timeout")
            return {"responses": [response_to_dict(resp, turn)]}

        node.__name__ = f"agent_{name}"
        return node

    async def _refuse_node(self, state: FinnieState) -> dict:
        unavailable = state["route"].get("unavailable") or []
        text = (
            UNAVAILABLE_MESSAGE.format(names=", ".join(u.replace("_", " ") for u in unavailable))
            if unavailable else OFF_TOPIC_MESSAGE
        )
        resp = AgentResponse("router", text, error="unsupported" if unavailable else None)
        return {"responses": [response_to_dict(resp, state["turn"])]}

    async def _synthesize_node(self, state: FinnieState) -> dict:
        turn = state["turn"]
        order = {n: i for i, n in enumerate(AGENT_INTENTS)}
        mine = sorted((r for r in state.get("responses", []) if r["turn"] == turn),
                      key=lambda r: order.get(r["agent"], 99))
        good = [r for r in mine if not r["error"]]

        if len(good) == 1:
            content = good[0]["content"]
        elif not good:
            content = mine[0]["content"] if mine else "Sorry, I couldn't process that request."
        else:
            content = await self._merge(state["query"], good)
        failed = [r["agent"] for r in mine if r["error"] and good]
        if failed:
            content += f"\n\n_Note: I couldn't complete part of your request ({', '.join(failed)})._"

        sources: list[dict] = []
        seen: set[str] = set()
        for r in good or mine:
            for s in r["sources"]:
                if s["title"] not in seen:
                    seen.add(s["title"])
                    sources.append(s)

        return {"final": {"content": content, "sources": sources, "agents": [r["agent"] for r in mine]}}

    async def _guardrail_node(self, state: FinnieState) -> dict:
        """Education-not-advice check on the merged answer; also what we store as memory."""
        final = dict(state["final"])
        result = await apply_guardrail(
            state["query"], final["content"], self.llm, add_disclaimer=final["agents"] != ["router"]
        )
        final["content"] = result.content
        final["guardrail"] = {"flagged": result.flagged, "reasons": result.reasons}
        return {
            "final": final,
            "messages": [
                {"role": "user", "content": state["query"]},
                {"role": "assistant", "content": result.content},
            ],
        }

    async def _merge(self, query: str, responses: list[dict]) -> str:
        parts = "\n\n".join(
            f"### {r['agent'].replace('_', ' ').title()}\n{strip_disclaimer(r['content'])}"
            for r in responses
        )
        try:
            merged = await self.llm.complete(MERGE_SYSTEM, f"User question: {query}\n\nAnswers:\n{parts}")
        except Exception as exc:  # never lose the specialist answers over a merge failure
            log.warning("merge failed (%s); concatenating", exc)
            merged = parts
        return f"{merged.strip()}\n\n_{DISCLAIMER}_"

    # ---- public API -------------------------------------------------------------
    async def chat(
        self,
        query: str,
        thread_id: str = "default",
        profile: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ChatResult:
        inputs: dict[str, Any] = {"query": query.strip(), "context": jsonable(context or {})}
        if profile is not None:
            inputs["profile"] = jsonable(profile)
        out = await self.graph.ainvoke(inputs, config={"configurable": {"thread_id": thread_id}})
        final = out["final"]
        return ChatResult(
            content=final["content"],
            sources=final["sources"],
            agents=final["agents"],
            route=out["route"],
            guardrail=final.get("guardrail", {}),
            responses=[r for r in out["responses"] if r["turn"] == out["turn"]],
            thread_id=thread_id,
            turn=out["turn"],
        )

    def history(self, thread_id: str = "default") -> list[dict]:
        snap = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return list(snap.values.get("messages", [])) if snap and snap.values else []
