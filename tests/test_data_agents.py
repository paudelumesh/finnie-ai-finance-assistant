import pytest

from src.agents.goal_planning import GoalPlanningAgent
from src.agents.market_analysis import MarketAnalysisAgent, extract_symbols
from src.agents.portfolio_analysis import PortfolioAnalysisAgent, to_holdings
from src.core.base_agent import DISCLAIMER
from src.core.llm import FakeLLM
from src.core.portfolio import Holding
from src.data.market_data import MarketDataService, MockProvider
from src.utils.json_utils import extract_json


class DeadProvider:
    async def get_quote(self, symbol):
        raise RuntimeError("down")

    get_history = get_quote


async def no_sleep(_):
    pass


def mock_market():
    return MarketDataService(MockProvider(), sleep=no_sleep)


def dead_market():
    return MarketDataService(DeadProvider(), max_retries=0, sleep=no_sleep)


# ---- json utils -----------------------------------------------------------
def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('text {"a": 3} more') == {"a": 3}
    assert extract_json("no json") is None and extract_json("") is None
    assert extract_json("[1, 2]") is None


# ---- market ---------------------------------------------------------------
@pytest.mark.parametrize(
    "query, expected",
    [
        ("How is AAPL doing today?", ["AAPL"]),
        ("compare apple and microsoft", ["AAPL", "MSFT"]),
        ("what about $PLTR?", ["PLTR"]),
        ("What is a REIT and an ETF?", []),
        ("Is the S&P 500 up?", ["SPY"]),
        ("VOO vs VTI vs QQQ vs BND vs AGG vs GLD", ["VOO", "VTI", "QQQ", "BND", "AGG"]),
    ],
)
def test_extract_symbols(query, expected):
    assert extract_symbols(query) == expected


async def test_market_agent_quote_and_data():
    llm = FakeLLM("AAPL is up a little.")
    resp = await MarketAnalysisAgent(llm, mock_market()).run("how is AAPL doing")
    assert resp.ok and DISCLAIMER in resp.content
    assert resp.data["symbols"] == ["AAPL"] and not resp.data["overview"]
    assert "AAPL: $" in llm.calls[0][1] and "data: mock" in llm.calls[0][1]
    assert resp.data["degraded"] is True and "sample or delayed" in resp.content


async def test_market_agent_overview_when_no_ticker():
    resp = await MarketAnalysisAgent(FakeLLM("ok"), mock_market()).run("how is the market today")
    assert resp.data["overview"] and resp.data["symbols"] == ["SPY", "QQQ", "IWM", "BND"]


async def test_market_agent_unknown_ticker_gives_friendly_error():
    resp = await MarketAnalysisAgent(FakeLLM("ok"), dead_market()).run("price of $ZZZZ")
    assert resp.error == "no_market_data" and "ZZZZ" in resp.content


# ---- portfolio ------------------------------------------------------------
async def test_portfolio_agent_requires_portfolio():
    resp = await PortfolioAnalysisAgent(FakeLLM(), mock_market()).run("analyse my portfolio")
    assert resp.ok and resp.data["needs_portfolio"] and "Upload a CSV" in resp.content


async def test_portfolio_agent_analyses_and_passes_numbers_to_llm():
    llm = FakeLLM("Looks concentrated.")
    ctx = {"portfolio": [Holding("VOO", 10), {"symbol": "bnd", "shares": 20, "cost_basis": 70}]}
    resp = await PortfolioAnalysisAgent(llm, mock_market()).run("how diversified am I?", ctx)
    assert resp.ok and DISCLAIMER in resp.content
    prompt = llm.calls[0][1]
    assert "Total value: $" in prompt and "Risk rating:" in prompt and "Weighted expense ratio" in prompt
    assert "Total gain/loss" in prompt
    assert resp.data["analysis"].total_value > 0 and resp.data["degraded"]


async def test_portfolio_agent_bad_input_and_no_data():
    agent = PortfolioAnalysisAgent(FakeLLM(), mock_market())
    bad = await agent.run("analyse", {"portfolio": [{"symbol": "VOO", "shares": -1}]})
    assert bad.error == "invalid_portfolio"
    missing_key = await agent.run("analyse", {"portfolio": [{"shares": 1}]})
    assert missing_key.error == "invalid_portfolio"
    unknown = await PortfolioAnalysisAgent(FakeLLM(), dead_market()).run(
        "analyse", {"portfolio": [Holding("ZZZZ", 1)]})
    assert unknown.error == "invalid_portfolio"


def test_to_holdings_accepts_both_shapes():
    out = to_holdings([Holding("A", 1), {"symbol": "b", "shares": "2"}])
    assert [(h.symbol, h.shares) for h in out] == [("A", 1), ("B", 2.0)]


# ---- goals ----------------------------------------------------------------
class ScriptedLLM(FakeLLM):
    """First call returns the extraction JSON; later calls return the narration."""

    def __init__(self, extraction):
        super().__init__("Here's your plan.")
        self.extraction = extraction

    async def complete(self, system, user):
        await super().complete(system, user)
        return self.extraction if len(self.calls) == 1 else self.reply


async def test_goal_agent_extracts_then_plans():
    llm = ScriptedLLM('{"target": 50000, "years": 10, "current": 5000, "monthly": null, "risk": "high"}')
    resp = await GoalPlanningAgent(llm).run("I want $50k in 10 years, have 5k, aggressive")
    plan = resp.data["plan"]
    assert resp.ok and plan.risk == "high" and plan.target == 50000 and plan.current == 5000
    assert "would need $" in llm.calls[1][1] and DISCLAIMER in resp.content


async def test_goal_agent_asks_for_missing_parameters():
    llm = ScriptedLLM('{"target": null, "years": 5, "current": null, "monthly": null, "risk": null}')
    resp = await GoalPlanningAgent(llm).run("help me save")
    assert resp.data["needs"] == ["target amount"] and "target amount" in resp.content
    garbage = await GoalPlanningAgent(ScriptedLLM("not json")).run("help me save")
    assert garbage.data["needs"] == ["target amount", "time horizon in years"]


async def test_goal_agent_uses_ui_context_without_extraction():
    llm = FakeLLM("narrated")
    ctx = {"goal": {"target": 20000, "years": 5}, "profile": {"risk_tolerance": "low"}}
    resp = await GoalPlanningAgent(llm).run("plan it", ctx)
    assert resp.data["plan"].risk == "low" and len(llm.calls) == 1


async def test_goal_agent_invalid_values():
    llm = ScriptedLLM('{"target": -5, "years": 10, "current": null, "monthly": null, "risk": null}')
    resp = await GoalPlanningAgent(llm).run("save -5 dollars")
    assert resp.error == "invalid_goal"
    bad_risk = ScriptedLLM('{"target": 5000, "years": 2, "current": "abc", "monthly": true, "risk": "yolo"}')
    assert (await GoalPlanningAgent(bad_risk).run("x")).error == "invalid_goal"


async def test_goal_followup_uses_history_for_extraction():
    llm = ScriptedLLM('{"target": 60000, "years": 12, "current": 5000, "monthly": null, "risk": null}')
    history = [{"role": "user", "content": "save 60000 in 8 years, I have 5000"},
               {"role": "assistant", "content": "plan..."}]
    resp = await GoalPlanningAgent(llm).run("what about 12 years?", {"history": history})
    assert resp.data["plan"].years == 12 and resp.data["plan"].target == 60000
    system, prompt = llm.calls[0]
    assert "save 60000 in 8 years" in prompt and "Latest message: what about 12 years?" in prompt
    assert "plan..." not in prompt  # only user messages are replayed


async def test_goal_summary_does_not_mislabel_projected_values():
    llm = ScriptedLLM('{"target": 60000, "years": 8, "current": 5000, "monthly": null, "risk": null}')
    await GoalPlanningAgent(llm).run("plan")
    narrate_prompt = llm.calls[1][1]
    assert "to hit the target exactly" in narrate_prompt and "saving the used" in narrate_prompt
