"""Deterministic goal-planning maths (future value, required savings, scenarios)."""
from __future__ import annotations

from dataclasses import dataclass

# Annual return assumptions per risk profile: (conservative, expected, optimistic).
# Illustrative planning assumptions for education, not forecasts.
RETURN_ASSUMPTIONS: dict[str, tuple[float, float, float]] = {
    "low": (0.02, 0.04, 0.06),
    "moderate": (0.03, 0.06, 0.09),
    "high": (0.04, 0.08, 0.11),
}
SCENARIO_NAMES = ("conservative", "expected", "optimistic")
RISK_ALIASES = {
    "conservative": "low", "cautious": "low", "low": "low",
    "moderate": "moderate", "medium": "moderate", "balanced": "moderate",
    "aggressive": "high", "high": "high", "growth": "high",
}


class GoalError(ValueError):
    """Invalid goal inputs."""


def normalise_risk(risk: str | None) -> str:
    key = (risk or "moderate").strip().lower()
    if key not in RISK_ALIASES:
        raise GoalError(f"unknown risk level '{risk}' (use low, moderate or high)")
    return RISK_ALIASES[key]


def future_value(current: float, monthly: float, annual_rate: float, years: float) -> float:
    """FV with monthly compounding and end-of-month contributions."""
    n = round(years * 12)
    r = annual_rate / 12
    if r == 0:
        return current + monthly * n
    growth = (1 + r) ** n
    return current * growth + monthly * (growth - 1) / r


def required_monthly(target: float, current: float, annual_rate: float, years: float) -> float:
    """Monthly contribution needed to reach `target`; 0 if current savings already suffice."""
    n = round(years * 12)
    r = annual_rate / 12
    needed = target - current * ((1 + r) ** n if r else 1)
    if needed <= 0:
        return 0.0
    factor = ((1 + r) ** n - 1) / r if r else n
    return needed / factor


@dataclass
class Scenario:
    name: str
    annual_return: float
    monthly_needed: float
    projected_value: float  # value reached if the *expected* contribution is saved
    yearly_path: list[float]


@dataclass
class GoalPlan:
    target: float
    current: float
    years: float
    risk: str
    scenarios: list[Scenario]
    contribution_used: float  # the monthly amount used for projection paths
    total_contributed: float
    on_track: bool


def plan_goal(
    target: float,
    years: float,
    current: float = 0.0,
    risk: str | None = "moderate",
    monthly: float | None = None,
) -> GoalPlan:
    """If `monthly` is None, project using the contribution the expected scenario requires."""
    if target <= 0:
        raise GoalError("target amount must be positive")
    if years <= 0 or years > 80:
        raise GoalError("years must be between 0 and 80")
    if current < 0 or (monthly is not None and monthly < 0):
        raise GoalError("savings amounts cannot be negative")
    level = normalise_risk(risk)
    rates = RETURN_ASSUMPTIONS[level]
    expected_need = required_monthly(target, current, rates[1], years)
    used = expected_need if monthly is None else monthly

    scenarios = []
    for name, rate in zip(SCENARIO_NAMES, rates):
        path = [round(future_value(current, used, rate, y), 2) for y in range(0, int(years) + 1)]
        if years != int(years):
            path.append(round(future_value(current, used, rate, years), 2))
        scenarios.append(Scenario(name, rate, round(required_monthly(target, current, rate, years), 2),
                                  path[-1], path))
    expected_value = scenarios[1].projected_value
    return GoalPlan(
        target=target, current=current, years=years, risk=level, scenarios=scenarios,
        contribution_used=round(used, 2),
        total_contributed=round(current + used * round(years * 12), 2),
        on_track=expected_value >= target - 0.01,
    )
