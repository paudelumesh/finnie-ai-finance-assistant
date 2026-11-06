"""Markets tab: index-fund overview, rebased comparison chart and single-ticker lookup."""
from __future__ import annotations

import asyncio

import streamlit as st

from src.agents.market_analysis import OVERVIEW_SYMBOLS
from src.data.market_data import MarketDataError, MarketDataService
from src.web_app import charts
from src.web_app.components import ask, freshness_note, theme_mode, worst_source
from src.web_app.runtime import Runtime

LABELS = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq 100 (QQQ)", "IWM": "Small caps (IWM)", "BND": "US bonds (BND)"}


async def load(market: MarketDataService, symbols: list[str], days: int):
    quotes = await asyncio.gather(*(market.get_quote(s) for s in symbols))
    trends = await asyncio.gather(*(market.get_trend(s, days) for s in symbols))
    return list(quotes), list(trends)


def render(rt: Runtime) -> None:
    market, mode = rt.services.market, theme_mode()
    top = st.columns([4, 1])
    days = top[0].radio("Window (trading days)", [30, 90], horizontal=True, key="mkt_days")
    if top[1].button("🔄 Refresh", help="Bypass the 30-minute cache"):
        market.clear_cache()

    try:
        with st.spinner("Loading market data…"):
            quotes, trends = rt.run(load(market, OVERVIEW_SYMBOLS, days))
    except MarketDataError as exc:
        st.error(f"Market data is unavailable right now: {exc}")
        return

    st.caption(freshness_note(worst_source(quotes), max(q.as_of for q in quotes)))
    cols = st.columns(len(quotes))
    for col, q in zip(cols, quotes):
        col.metric(LABELS.get(q.symbol, q.symbol), f"${q.price:,.2f}", f"{q.change_pct:+.2f}% today")

    st.subheader(f"Performance, last {days} trading days (indexed to 100)")
    st.plotly_chart(charts.compare_chart(trends, mode), width="stretch")
    if st.button("💬 Explain today's market"):
        with st.spinner("Finnie is reading the market…"):
            st.markdown(ask(rt, "How is the market doing today? Give me a beginner-friendly overview.")["content"])

    st.divider()
    st.subheader("Look up a stock or ETF")
    symbol = st.text_input("Ticker symbol", placeholder="e.g. AAPL, MSFT, VTI").strip().upper().lstrip("$")
    if symbol:
        try:
            qs, ts = rt.run(load(market, [symbol], days))
        except MarketDataError:
            st.error(f"No market data found for {symbol}. Check the ticker and try again.")
            return
        q, t = qs[0], ts[0]
        c1, c2, c3 = st.columns(3)
        c1.metric(symbol, f"${q.price:,.2f}", f"{q.change_pct:+.2f}% today")
        c2.metric(f"{days}-day change", f"{t.change_pct:+.1f}%")
        c3.metric("Trend", t.direction.title())
        st.caption(freshness_note(q.source, q.as_of))
        st.plotly_chart(charts.trend_chart(t, mode), width="stretch")
