from datetime import datetime, timezone

import pytest

from src.core.portfolio import (
    Holding, PortfolioError, analyze_portfolio, hhi_score, parse_holdings_csv,
)
from src.data.market_data import Quote


def quotes(**prices):
    now = datetime.now(timezone.utc)
    return {s: Quote(s, p, p, now) for s, p in prices.items()}


def test_hhi_score_bounds():
    assert hhi_score([1.0]) == 0
    assert hhi_score([]) == 0
    assert hhi_score([0.1] * 10) == 100
    assert hhi_score([0.05] * 20) == 100
    assert 0 < hhi_score([0.7, 0.3]) < hhi_score([0.5, 0.5]) < 100


def test_analysis_totals_weights_and_classes():
    a = analyze_portfolio(
        [Holding("VOO", 10), Holding("BND", 50, 70.0)], quotes(VOO=500.0, BND=100.0)
    )
    assert a.total_value == 10000
    assert [p.symbol for p in a.positions] == ["VOO", "BND"]
    assert a.positions[0].weight == pytest.approx(0.5)
    assert a.allocation_by_class == {"etf_equity": 0.5, "etf_bond": 0.5}
    assert a.positions[1].gain_loss == pytest.approx(1500)
    assert a.total_gain_loss == pytest.approx(1500)
    assert a.weighted_expense_ratio == pytest.approx(0.0003)
    assert a.annual_fee_estimate == pytest.approx(3.0)
    assert a.risk_rating == "Moderate"


def test_concentration_and_no_bond_warnings():
    a = analyze_portfolio([Holding("AAPL", 90), Holding("MSFT", 10)], quotes(AAPL=100, MSFT=100))
    text = " ".join(a.warnings)
    assert "AAPL is 90%" in text and "No bond exposure" in text and "Fewer than 5" in text
    assert a.risk_rating in {"High", "Very High"}


def test_single_asset_and_unknown_symbol_warnings():
    a = analyze_portfolio([Holding("ZZZZ", 1)], quotes(ZZZZ=10))
    text = " ".join(a.warnings)
    assert "single asset" in text and "No reference data for ZZZZ" in text
    assert a.total_gain_loss is None and a.diversification_score == 0


def test_crypto_warning_and_low_risk():
    crypto = analyze_portfolio([Holding("BTC-USD", 1), Holding("BND", 1)], quotes(**{"BTC-USD": 1000, "BND": 100}))
    assert any("Crypto" in w for w in crypto.warnings)
    low = analyze_portfolio([Holding("BND", 10)], quotes(BND=100))
    assert low.risk_rating == "Low"


def test_analysis_errors():
    with pytest.raises(PortfolioError, match="empty"):
        analyze_portfolio([], {})
    with pytest.raises(PortfolioError, match="no price"):
        analyze_portfolio([Holding("VOO", 1)], {})
    with pytest.raises(PortfolioError, match="positive"):
        analyze_portfolio([Holding("VOO", 0)], quotes(VOO=1))
    with pytest.raises(PortfolioError, match="zero"):
        analyze_portfolio([Holding("VOO", 1)], quotes(VOO=0))
    with pytest.raises(PortfolioError, match="negative"):
        analyze_portfolio([Holding("VOO", 1, -5)], quotes(VOO=1))


def test_csv_parsing_variants_and_merging():
    csv_text = "Ticker,Quantity,Avg_Cost\nvoo,10,400\nVOO,10,500\nbnd,5,\n"
    h = {x.symbol: x for x in parse_holdings_csv(csv_text)}
    assert h["VOO"].shares == 20 and h["VOO"].cost_basis == pytest.approx(450)
    assert h["BND"].cost_basis is None


def test_csv_merge_unknown_cost_drops_basis():
    h = parse_holdings_csv("symbol,shares,cost\nVOO,1,100\nVOO,1,\n")
    assert h[0].cost_basis is None


@pytest.mark.parametrize(
    "text, msg",
    [
        ("", "empty"),
        ("name,amount\nx,1", "needs 'symbol'"),
        ("symbol,shares\nVOO,abc", "invalid number"),
        ("symbol,shares\n,5", "no holdings"),
        ("symbol,shares\nVOO,-1", "positive"),
    ],
)
def test_csv_errors(text, msg):
    with pytest.raises(PortfolioError, match=msg):
        parse_holdings_csv(text)
