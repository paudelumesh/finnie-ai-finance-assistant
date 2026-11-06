"""Goals tab: deterministic savings projections with three return scenarios."""
from __future__ import annotations

import streamlit as st

from src.core.goals import RETURN_ASSUMPTIONS, GoalError, normalise_risk, plan_goal
from src.web_app import charts
from src.web_app.components import ask, theme_mode
from src.web_app.runtime import Runtime

RISK_LABELS = {"low": "Low (cautious)", "moderate": "Moderate (balanced)", "high": "High (growth)"}


def render(rt: Runtime) -> None:
    ss, mode = st.session_state, theme_mode()
    default_risk = normalise_risk(ss.profile.get("risk_tolerance"))
    with st.form("goal_form"):
        c1, c2, c3 = st.columns(3)
        target = c1.number_input("Target amount ($)", min_value=100.0, value=60000.0, step=1000.0)
        years = c2.number_input("Years to reach it", min_value=1.0, max_value=60.0, value=8.0, step=1.0)
        current = c3.number_input("Already saved ($)", min_value=0.0, value=5000.0, step=500.0)
        c4, c5 = st.columns(2)
        risk = c4.selectbox("Comfort with risk", list(RISK_LABELS), index=list(RISK_LABELS).index(default_risk),
                            format_func=RISK_LABELS.get)
        monthly = c5.number_input("Monthly saving ($, 0 = calculate it for me)", min_value=0.0, value=0.0, step=50.0)
        submitted = st.form_submit_button("Build my plan", type="primary")

    if submitted:
        try:
            ss.plan = plan_goal(target, years, current, risk, monthly or None)
        except GoalError as exc:
            st.error(str(exc))
            ss.plan = None

    plan = ss.plan
    if not plan:
        st.info("Fill in your goal and press **Build my plan**.")
        return

    expected = plan.scenarios[1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly saving used", charts.money(plan.contribution_used),
              help="If you left monthly saving at 0, this is what the expected scenario needs.")
    c2.metric("Total you contribute", charts.money(plan.total_contributed))
    c3.metric("Expected value at the end", charts.money(expected.projected_value),
              "On track" if plan.on_track else "Short of target", delta_color="normal" if plan.on_track else "inverse")

    st.plotly_chart(charts.projection_chart(plan, mode), width="stretch")

    st.subheader("What each scenario needs")
    st.table({
        "Scenario": [s.name.title() for s in plan.scenarios],
        "Assumed return / yr": [f"{s.annual_return:.0%}" for s in plan.scenarios],
        "Monthly needed to hit target": [charts.money(s.monthly_needed) for s in plan.scenarios],
        f"Value after {plan.years:g} yrs at ${plan.contribution_used:,.0f}/mo": [charts.money(s.projected_value) for s in plan.scenarios],
    })
    low, mid, high = RETURN_ASSUMPTIONS[plan.risk]
    st.caption(f"Return assumptions for a {plan.risk} risk profile ({low:.0%} / {mid:.0%} / {high:.0%}) are "
               "illustrative planning figures, not forecasts or guarantees. Real returns vary year to year.")

    if st.button("💬 Explain this plan with Finnie", type="primary"):
        goal = {"target": plan.target, "years": plan.years, "current": plan.current, "risk": plan.risk,
                "monthly": monthly or None}
        with st.spinner("Finnie is writing it up…"):
            entry = ask(rt, f"Explain my savings plan for ${plan.target:,.0f} in {plan.years:g} years.", {"goal": goal})
        st.markdown(entry["content"])
