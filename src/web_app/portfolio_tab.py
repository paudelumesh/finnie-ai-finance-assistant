"""Portfolio tab: input holdings, run the deterministic analysis, visualise it."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.core.portfolio import Holding, PortfolioError, analyze_portfolio, parse_holdings_csv, validate_holdings
from src.core.settings import ROOT_DIR
from src.data.market_data import MarketDataError
from src.web_app import charts
from src.web_app.components import ask, freshness_note, theme_mode, worst_source
from src.web_app.runtime import Runtime

SAMPLE_CSV = ROOT_DIR / "data_samples" / "sample_portfolio.csv"
TEMPLATE = pd.DataFrame({"symbol": ["VOO", "BND", ""], "shares": [10.0, 20.0, None], "cost_basis": [None, None, None]})


def _text(value) -> str:
    """Blank cells arrive as None or NaN (NaN is truthy), so test for missing explicitly."""
    return "" if value is None or pd.isna(value) else str(value).strip()


def holdings_from_frame(df: pd.DataFrame) -> list[Holding]:
    out = []
    for row in df.to_dict("records"):
        sym = _text(row.get("symbol")).upper()
        if not sym:
            continue
        shares = row.get("shares")
        if shares is None or pd.isna(shares):
            raise PortfolioError(f"{sym}: enter the number of shares")
        cost = row.get("cost_basis")
        out.append(Holding(sym, float(shares), None if cost is None or pd.isna(cost) else float(cost)))
    if not out:
        raise PortfolioError("add at least one holding")
    return validate_holdings(out)


def _holdings_to_state(holdings: list[Holding]) -> list[dict]:
    return [{"symbol": h.symbol, "shares": h.shares, "cost_basis": h.cost_basis} for h in holdings]


def _collect_holdings(source: str) -> list[Holding] | None:
    if source == "Upload CSV":
        up = st.file_uploader("CSV with columns: symbol, shares, optional cost_basis", type="csv")
        return parse_holdings_csv(up.getvalue().decode("utf-8")) if up else None
    if source == "Use sample portfolio":
        st.caption("A demo portfolio of AAPL, MSFT, VOO and NVDA.")
        return parse_holdings_csv(Path(SAMPLE_CSV).read_text())
    df = st.data_editor(TEMPLATE, num_rows="dynamic", width="stretch", key="holdings_editor",
                        column_config={"cost_basis": st.column_config.NumberColumn("cost_basis (per share, optional)")})
    return holdings_from_frame(df)


def _render_analysis(rt: Runtime, mode: str) -> None:
    ss = st.session_state
    a, quotes = ss.analysis, list(ss.quotes.values())
    st.caption(freshness_note(worst_source(quotes), max(q.as_of for q in quotes)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total value", charts.money(a.total_value))
    c2.metric("Risk rating", a.risk_rating)
    c3.metric("Diversification", f"{a.diversification_score:.0f} / 100",
              help="Based on the effective number of equally sized holdings: 1 holding = 0, 10 or more = 100.")
    c4.metric("Est. annual fees", charts.money(a.annual_fee_estimate), f"{a.weighted_expense_ratio:.3%} expense ratio",
              delta_color="off")

    left, right = st.columns(2)
    with left:
        st.subheader("By asset type")
        st.plotly_chart(charts.allocation_donut(a.allocation_by_class, mode), width="stretch")
    with right:
        st.subheader("By sector")
        st.plotly_chart(charts.sector_bars(a.allocation_by_sector, mode), width="stretch")

    if a.warnings:
        st.subheader("Things to look at")
        for w in a.warnings:
            st.warning(w, icon="⚠️")
    st.subheader("Positions")
    st.dataframe(
        charts.positions_frame(a), width="stretch", hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Value": st.column_config.NumberColumn(format="$%.2f"),
            "Gain/Loss": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    if st.button("💬 Explain my portfolio with Finnie", type="primary"):
        with st.spinner("Finnie is analysing…"):
            entry = ask(rt, "Analyze my portfolio and explain what the numbers mean.")
        st.markdown(entry["content"])


def render(rt: Runtime) -> None:
    ss = st.session_state
    mode = theme_mode()
    source = st.radio("How do you want to enter holdings?", ["Use sample portfolio", "Upload CSV", "Enter manually"],
                      horizontal=True)
    try:
        holdings = _collect_holdings(source)
    except PortfolioError as exc:
        st.error(str(exc))
        holdings = None

    if holdings and st.button("Analyze portfolio", type="primary"):
        try:
            with st.spinner("Fetching prices…"):
                quotes = rt.run(rt.services.market.get_quotes([h.symbol for h in holdings]))
            ss.analysis, ss.quotes = analyze_portfolio(holdings, quotes), quotes
            ss.holdings = _holdings_to_state(holdings)
        except (PortfolioError, MarketDataError) as exc:
            st.error(f"Couldn't analyse that portfolio: {exc}")

    if ss.analysis:
        _render_analysis(rt, mode)
    else:
        st.info("Choose how to enter your holdings, then press **Analyze portfolio**.")
