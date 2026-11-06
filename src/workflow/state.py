"""LangGraph state. Everything here is JSON-friendly so any checkpointer can persist it."""
from __future__ import annotations

import operator
from dataclasses import is_dataclass
from datetime import date, datetime
from typing import Annotated, Any, TypedDict

from src.core.base_agent import AgentResponse


class FinnieState(TypedDict, total=False):
    query: str
    turn: int
    profile: dict[str, Any]  # risk_tolerance, knowledge_level, goals (persists across turns)
    context: dict[str, Any]  # per-turn inputs: portfolio holdings, goal form values
    messages: Annotated[list[dict], operator.add]  # conversation history (memory)
    route: dict[str, Any]
    intent: str  # set on the payload sent to a single agent node
    responses: Annotated[list[dict], operator.add]  # per-agent results, tagged with `turn`
    final: dict[str, Any]


def jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-safe data; objects we can't represent are dropped."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): v for k, v in ((k, jsonable(x)) for k, x in value.items()) if v is not _DROP}
    if isinstance(value, (list, tuple)):
        return [v for v in (jsonable(x) for x in value) if v is not _DROP]
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable({k: getattr(value, k) for k in value.__dataclass_fields__})
    return _DROP


_DROP = object()


def response_to_dict(resp: AgentResponse, turn: int) -> dict[str, Any]:
    return {
        "turn": turn,
        "agent": resp.agent,
        "content": resp.content,
        "sources": [{"title": s.title, "url": s.url, "category": s.category} for s in resp.sources],
        "error": resp.error,
        "data": {k: v for k, v in ((k, jsonable(x)) for k, x in resp.data.items()) if v is not _DROP},
    }
