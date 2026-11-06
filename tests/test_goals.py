import pytest

from src.core.goals import (
    GoalError, future_value, normalise_risk, plan_goal, required_monthly,
)


def test_future_value_known_values():
    assert future_value(1000, 0, 0.0, 5) == 1000
    assert future_value(0, 100, 0.0, 1) == 1200
    # $1000 at 12%/yr compounded monthly for 1 year = 1000 * 1.01^12
    assert future_value(1000, 0, 0.12, 1) == pytest.approx(1126.83, abs=0.01)


def test_required_monthly_inverts_future_value():
    m = required_monthly(50000, 5000, 0.06, 10)
    assert future_value(5000, m, 0.06, 10) == pytest.approx(50000, rel=1e-6)
    assert required_monthly(100, 1000, 0.05, 5) == 0  # already funded
    assert required_monthly(1200, 0, 0.0, 1) == pytest.approx(100)


def test_plan_goal_scenarios_ordered_and_on_track():
    p = plan_goal(100000, 20, current=10000, risk="moderate")
    cons, exp, opt = p.scenarios
    assert cons.monthly_needed > exp.monthly_needed > opt.monthly_needed
    assert cons.projected_value < exp.projected_value < opt.projected_value
    assert exp.projected_value == pytest.approx(100000, rel=1e-3) and p.on_track
    assert len(exp.yearly_path) == 21 and exp.yearly_path[0] == 10000


def test_plan_goal_with_fixed_monthly_can_miss_target():
    p = plan_goal(100000, 10, monthly=100, risk="low")
    assert not p.on_track and p.contribution_used == 100
    assert p.total_contributed == 100 * 120


def test_fractional_years_adds_final_point():
    assert len(plan_goal(10000, 2.5).scenarios[1].yearly_path) == 4


def test_higher_risk_needs_less_monthly():
    low = plan_goal(50000, 10, risk="low").scenarios[1].monthly_needed
    high = plan_goal(50000, 10, risk="aggressive").scenarios[1].monthly_needed
    assert high < low


def test_risk_normalisation():
    assert normalise_risk(None) == "moderate"
    assert normalise_risk("Conservative") == "low"
    with pytest.raises(GoalError):
        normalise_risk("yolo")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(target=0, years=5), dict(target=1000, years=0), dict(target=1000, years=100),
        dict(target=1000, years=5, current=-1), dict(target=1000, years=5, monthly=-1),
    ],
)
def test_plan_goal_validation(kwargs):
    with pytest.raises(GoalError):
        plan_goal(**kwargs)
