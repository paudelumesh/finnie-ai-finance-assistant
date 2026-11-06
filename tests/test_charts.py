import pandas as pd
import pytest

from src.core.goals import plan_goal
from src.core.portfolio import Holding, analyze_portfolio
from src.data.market_data import PricePoint, Quote, Trend
from src.web_app import charts
from datetime import datetime, timezone


def trend(symbol, start, end, n=10):
    pts = [PricePoint(f"2026-01-{i + 1:02d}", start + (end - start) * i / (n - 1)) for i in range(n)]
    return Trend(symbol, pts)


def analysis():
    now = datetime.now(timezone.utc)
    q = {s: Quote(s, p, p, now) for s, p in {"VOO": 500.0, "BND": 100.0, "AAPL": 200.0}.items()}
    return analyze_portfolio([Holding("VOO", 10, 400), Holding("BND", 50), Holding("AAPL", 5)], q)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_donut_uses_fixed_slot_colours_and_gap(mode):
    a = analysis()
    fig = charts.allocation_donut(a.allocation_by_class, mode)
    pie = fig.data[0]
    assert list(pie.marker.colors) == charts.series_colors(mode)[: len(a.allocation_by_class)]
    assert pie.marker.line.color == charts.tokens(mode)["surface"] and pie.marker.line.width == 2
    assert pie.sort is False and abs(sum(pie.values) - 1) < 1e-9
    assert pie.textposition == "outside"
    assert [n for n in pie.labels] == ["Stock funds", "Bond funds", "Individual stocks"]


def test_sector_bars_sorted_single_hue_one_axis():
    a = analysis()
    fig = charts.sector_bars(a.allocation_by_sector)
    bar = fig.data[0]
    assert list(bar.x) == sorted(bar.x) and bar.marker.color == charts.series_colors("light")[0]
    assert "yaxis2" not in fig.layout and fig.layout.showlegend is False


def test_positions_frame_columns_and_gain():
    df = charts.positions_frame(analysis())
    assert isinstance(df, pd.DataFrame) and df["Symbol"].tolist()[0] == "VOO"
    assert df.loc[df.Symbol == "VOO", "Gain/Loss"].iloc[0] == pytest.approx(1000)
    assert pd.isna(df.loc[df.Symbol == "BND", "Gain/Loss"].iloc[0])
    assert df["Weight %"].sum() == pytest.approx(100, abs=0.2)


def test_trend_chart_has_line_and_endpoint_marker():
    fig = charts.trend_chart(trend("AAPL", 100, 110))
    line, marker = fig.data
    assert line.mode == "lines" and line.line.width == 2 and len(line.x) == 10
    assert len(marker.x) == 1 and marker.marker.size >= 8 and marker.showlegend is False


def test_compare_chart_rebases_to_100_with_direct_labels():
    fig = charts.compare_chart([trend("SPY", 400, 440), trend("BND", 70, 70), Trend("EMPTY")])
    assert len(fig.data) == 2  # empty trend skipped
    assert fig.data[0].y[0] == pytest.approx(100) and fig.data[0].y[-1] == pytest.approx(110)
    assert fig.data[1].y[-1] == pytest.approx(100)
    assert [a.text for a in fig.layout.annotations][:2] == ["SPY 110.0", "BND 100.0"]
    assert fig.data[0].line.color != fig.data[1].line.color and "yaxis2" not in fig.layout


def test_projection_chart_three_scenarios_target_line_and_fixed_colours():
    plan = plan_goal(60000, 8, current=5000, risk="moderate")
    fig = charts.projection_chart(plan, "light")
    names = [t.name for t in fig.data]
    assert names[0].startswith("Conservative") and names[1].startswith("Expected") and names[2].startswith("Optimistic")
    cols = charts.series_colors("light")
    assert [t.line.color for t in fig.data] == [cols[1], cols[0], cols[2]]
    assert fig.data[1].line.width == 3 and len(fig.data[1].x) == 9
    assert any(abs(s.y0 - 60000) < 1e-6 for s in fig.layout.shapes)
    assert fig.data[1].y[-1] == pytest.approx(60000, rel=1e-3)


def test_projection_chart_fractional_years_ends_on_exact_year():
    fig = charts.projection_chart(plan_goal(10000, 2.5))
    assert fig.data[1].x[-1] == 2.5


def test_money_formatting_and_tokens():
    assert charts.money(1234567) == "$1,234,567" and charts.money(12.5) == "$12.50"
    assert charts.tokens("dark")["surface"] == "#1a1a19" and charts.tokens("anything")["surface"] == "#fcfcfb"
