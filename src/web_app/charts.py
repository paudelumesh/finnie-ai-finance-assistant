"""Plotly chart builders. Pure functions (data in, Figure out) so they are easy to test.

Colour follows the dataviz method: categorical hues in fixed slot order (validated palette),
one hue for magnitude, a neutral for reference lines, text in ink tokens (never series colour),
one y-axis per chart, thin marks, and a surface gap between touching fills.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.core.goals import GoalPlan
from src.core.portfolio import CLASS_LABELS, PortfolioAnalysis
from src.data.market_data import Trend

# Categorical slots 1-8 (blue, orange, aqua, yellow, magenta, green, violet, red).
_SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
_TOKENS = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e1"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#33332f"},
}
BLUE, ORANGE, AQUA = 0, 1, 2  # slot indices used for the three goal scenarios

SCENARIO_SLOT = {"conservative": ORANGE, "expected": BLUE, "optimistic": AQUA}


def tokens(mode: str) -> dict[str, str]:
    return _TOKENS["dark" if mode == "dark" else "light"]


def series_colors(mode: str) -> list[str]:
    return _SERIES["dark" if mode == "dark" else "light"]


def money(value: float) -> str:
    return f"${value:,.0f}" if abs(value) >= 1000 else f"${value:,.2f}"


def _layout(fig: go.Figure, mode: str, height: int = 320, legend: bool = True) -> go.Figure:
    t = tokens(mode)
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], size=13),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.15, font=dict(color=t["muted"])),
        hoverlabel=dict(bgcolor=t["surface"], font=dict(color=t["text"])),
    )
    fig.update_xaxes(showgrid=False, linecolor=t["grid"], tickfont=dict(color=t["muted"]))
    fig.update_yaxes(gridcolor=t["grid"], zeroline=False, tickfont=dict(color=t["muted"]))
    return fig


def allocation_donut(allocation: dict[str, float], mode: str = "light", labels: dict[str, str] | None = None) -> go.Figure:
    labels = labels or CLASS_LABELS
    colors = series_colors(mode)
    names = [labels.get(k, k) for k in allocation]
    fig = go.Figure(
        go.Pie(
            labels=names,
            values=list(allocation.values()),
            hole=0.55,
            sort=False,  # keep fixed order so each class keeps its colour
            marker=dict(colors=colors[: len(names)], line=dict(color=tokens(mode)["surface"], width=2)),
            textinfo="percent",
            textposition="outside",  # ink-coloured labels outside: low-contrast slices stay readable
            textfont=dict(color=tokens(mode)["text"], size=13),
            hovertemplate="%{label}<br>%{percent}<extra></extra>",
        )
    )
    return _layout(fig, mode, height=300)


def sector_bars(allocation: dict[str, float], mode: str = "light") -> go.Figure:
    items = sorted(allocation.items(), key=lambda kv: kv[1])
    fig = go.Figure(
        go.Bar(
            x=[v * 100 for _, v in items],
            y=[k for k, _ in items],
            orientation="h",
            marker=dict(color=series_colors(mode)[BLUE]),
            text=[f"{v:.0%}" for _, v in items],
            textposition="outside",
            textfont=dict(color=tokens(mode)["text"]),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_xaxes(title=None, ticksuffix="%", range=[0, max([v for _, v in items] + [0.1]) * 125])
    fig.update_yaxes(showgrid=False)
    return _layout(fig, mode, height=max(220, 44 * len(items) + 40), legend=False)


def positions_frame(a: PortfolioAnalysis) -> pd.DataFrame:
    rows = []
    for p in a.positions:
        rows.append({
            "Symbol": p.symbol,
            "Name": p.name,
            "Type": CLASS_LABELS.get(p.asset_class, p.asset_class),
            "Sector": p.sector,
            "Shares": p.shares,
            "Price": p.price,
            "Value": p.value,
            "Weight %": round(p.weight * 100, 1),
            "Expense ratio %": round(p.expense_ratio * 100, 3),
            "Gain/Loss": p.gain_loss,
            "Gain/Loss %": None if p.gain_loss_pct is None else round(p.gain_loss_pct, 1),
        })
    return pd.DataFrame(rows)


def trend_chart(trend: Trend, mode: str = "light") -> go.Figure:
    pts = trend.points
    x, y = [p.day for p in pts], [p.close for p in pts]
    color = series_colors(mode)[BLUE]
    fig = go.Figure(
        [
            go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=2),
                       name=trend.symbol, hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>"),
            go.Scatter(x=x[-1:], y=y[-1:], mode="markers", showlegend=False, hoverinfo="skip",
                       marker=dict(color=color, size=9, line=dict(color=tokens(mode)["surface"], width=2))),
        ]
    )
    fig.update_yaxes(tickprefix="$")
    return _layout(fig, mode, height=300, legend=False)


def compare_chart(trends: list[Trend], mode: str = "light") -> go.Figure:
    """Each series rebased to 100 at its first point, so differently priced funds share one axis."""
    colors = series_colors(mode)
    t = tokens(mode)
    fig = go.Figure()
    for i, tr in enumerate(trends):
        if not tr.points or not tr.points[0].close:
            continue
        base = tr.points[0].close
        x = [p.day for p in tr.points]
        y = [p.close / base * 100 for p in tr.points]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=tr.symbol, line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f"{tr.symbol} %{{x}}<br>%{{y:.1f}}<extra></extra>"))
        fig.add_annotation(x=x[-1], y=y[-1], text=f"{tr.symbol} {y[-1]:.1f}", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=t["text"], size=12))
    fig.add_hline(y=100, line=dict(color=t["grid"], width=1))
    fig.update_yaxes(title="Indexed (start = 100)")
    fig.update_layout(margin=dict(l=8, r=60, t=16, b=8))
    return _layout(fig, mode, height=320, legend=True)


def projection_chart(plan: GoalPlan, mode: str = "light") -> go.Figure:
    colors = series_colors(mode)
    t = tokens(mode)
    fig = go.Figure()
    for s in plan.scenarios:
        years = list(range(len(s.yearly_path)))
        if plan.years != int(plan.years):
            years[-1] = plan.years
        fig.add_trace(go.Scatter(
            x=years, y=s.yearly_path, mode="lines", name=f"{s.name.title()} ({s.annual_return:.0%}/yr)",
            line=dict(color=colors[SCENARIO_SLOT[s.name]], width=3 if s.name == "expected" else 2),
            hovertemplate=f"{s.name.title()}<br>Year %{{x:g}}: $%{{y:,.0f}}<extra></extra>"))
    fig.add_hline(y=plan.target, line=dict(color=t["muted"], width=1.5, dash="dash"),
                  annotation_text=f"Target {money(plan.target)}", annotation_position="top left",
                  annotation_font=dict(color=t["muted"]))
    fig.update_xaxes(title="Years", dtick=1 if plan.years <= 15 else None)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return _layout(fig, mode, height=360)
