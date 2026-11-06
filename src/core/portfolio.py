"""Deterministic portfolio maths. The LLM explains these numbers; it never computes them."""
from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field

from src.data.asset_info import BOND_CLASSES, EQUITY_CLASSES, lookup
from src.data.market_data import Quote

CLASS_LABELS = {
    "stock": "Individual stocks", "etf_equity": "Stock funds", "etf_bond": "Bond funds",
    "bond": "Bonds", "real_estate": "Real estate", "commodity": "Commodities",
    "crypto": "Crypto", "cash": "Cash",
}


class PortfolioError(ValueError):
    """Invalid portfolio input."""


@dataclass(frozen=True)
class Holding:
    symbol: str
    shares: float
    cost_basis: float | None = None  # per share


@dataclass
class Position:
    symbol: str
    name: str
    asset_class: str
    sector: str
    shares: float
    price: float
    value: float
    weight: float  # fraction of total
    expense_ratio: float
    gain_loss: float | None = None  # absolute, if cost basis known
    gain_loss_pct: float | None = None


@dataclass
class PortfolioAnalysis:
    total_value: float
    positions: list[Position]
    allocation_by_class: dict[str, float]
    allocation_by_sector: dict[str, float]
    diversification_score: float  # 0-100, higher is more diversified
    weighted_expense_ratio: float
    annual_fee_estimate: float
    risk_rating: str  # Low | Moderate | High | Very High
    total_gain_loss: float | None
    warnings: list[str] = field(default_factory=list)


def parse_holdings_csv(text: str) -> list[Holding]:
    """Accepts columns: symbol|ticker, shares|quantity, optional cost_basis|cost|avg_cost."""
    reader = csv.DictReader(io.StringIO(text.strip()))
    if not reader.fieldnames:
        raise PortfolioError("CSV is empty")
    cols = {c.strip().lower(): c for c in reader.fieldnames}

    def col(*names: str) -> str | None:
        return next((cols[n] for n in names if n in cols), None)

    sym_c, sh_c = col("symbol", "ticker"), col("shares", "quantity", "qty")
    cost_c = col("cost_basis", "cost", "avg_cost", "average_cost")
    if not sym_c or not sh_c:
        raise PortfolioError("CSV needs 'symbol' (or 'ticker') and 'shares' columns")

    merged: dict[str, Holding] = {}
    for n, row in enumerate(reader, start=2):
        sym = (row[sym_c] or "").strip().upper()
        if not sym:
            continue
        try:
            shares = float(row[sh_c])
            cost = float(row[cost_c]) if cost_c and (row[cost_c] or "").strip() else None
        except ValueError as exc:
            raise PortfolioError(f"row {n}: invalid number for {sym}") from exc
        merged_holding = merge_holdings(merged.get(sym), Holding(sym, shares, cost))
        merged[sym] = merged_holding
    if not merged:
        raise PortfolioError("no holdings found")
    return validate_holdings(list(merged.values()))


def merge_holdings(existing: Holding | None, new: Holding) -> Holding:
    if existing is None:
        return new
    total = existing.shares + new.shares
    if existing.cost_basis is None or new.cost_basis is None or total == 0:
        cost = None
    else:
        cost = (existing.cost_basis * existing.shares + new.cost_basis * new.shares) / total
    return Holding(new.symbol, total, cost)


def validate_holdings(holdings: list[Holding]) -> list[Holding]:
    for h in holdings:
        if h.shares <= 0:
            raise PortfolioError(f"{h.symbol}: shares must be positive")
        if h.cost_basis is not None and h.cost_basis < 0:
            raise PortfolioError(f"{h.symbol}: cost basis cannot be negative")
    return holdings


def hhi_score(weights: list[float]) -> float:
    """Diversification 0-100 from the effective number of holdings (1 / HHI).

    One holding scores 0; ten or more equally sized holdings score 100.
    """
    hhi = sum(w * w for w in weights)
    if not weights or hhi <= 0:
        return 0.0
    effective_n = 1 / hhi
    return round(max(0.0, min(100.0, (effective_n - 1) / 9 * 100)), 1)


def _risk_rating(by_class: dict[str, float]) -> str:
    equity = sum(v for k, v in by_class.items() if k in EQUITY_CLASSES)
    risky = by_class.get("crypto", 0) + by_class.get("commodity", 0)
    stocks_only = by_class.get("stock", 0)
    score = equity * 1.0 + risky * 1.5 + stocks_only * 0.3
    if score < 0.35:
        return "Low"
    if score < 0.7:
        return "Moderate"
    if score < 1.0:
        return "High"
    return "Very High"


def analyze_portfolio(holdings: list[Holding], quotes: dict[str, Quote]) -> PortfolioAnalysis:
    validate_holdings(holdings)
    if not holdings:
        raise PortfolioError("portfolio is empty")
    rows = []
    for h in holdings:
        q = quotes.get(h.symbol.upper())
        if q is None:
            raise PortfolioError(f"no price for {h.symbol}")
        rows.append((h, q.price, h.shares * q.price))
    total = sum(v for _, _, v in rows)
    if total <= 0:
        raise PortfolioError("portfolio value is zero")

    positions: list[Position] = []
    by_class: dict[str, float] = defaultdict(float)
    by_sector: dict[str, float] = defaultdict(float)
    gain_total, any_cost = 0.0, False
    for h, price, value in sorted(rows, key=lambda r: -r[2]):
        info = lookup(h.symbol)
        weight = value / total
        gl = glp = None
        if h.cost_basis is not None:
            any_cost = True
            gl = (price - h.cost_basis) * h.shares
            glp = ((price / h.cost_basis) - 1) * 100 if h.cost_basis else None
            gain_total += gl
        positions.append(Position(h.symbol.upper(), info.name, info.asset_class, info.sector,
                                  h.shares, price, value, weight, info.expense_ratio, gl, glp))
        by_class[info.asset_class] += weight
        by_sector[info.sector] += weight

    weights = [p.weight for p in positions]
    wer = sum(p.weight * p.expense_ratio for p in positions)
    warnings = _warnings(positions, by_class)
    return PortfolioAnalysis(
        total_value=round(total, 2),
        positions=positions,
        allocation_by_class=dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        allocation_by_sector=dict(sorted(by_sector.items(), key=lambda kv: -kv[1])),
        diversification_score=hhi_score(weights),
        weighted_expense_ratio=wer,
        annual_fee_estimate=round(total * wer, 2),
        risk_rating=_risk_rating(by_class),
        total_gain_loss=round(gain_total, 2) if any_cost else None,
        warnings=warnings,
    )


def _warnings(positions: list[Position], by_class: dict[str, float]) -> list[str]:
    out = []
    top = positions[0]
    if top.weight > 0.30 and len(positions) > 1:
        out.append(f"{top.symbol} is {top.weight:.0%} of the portfolio (a concentrated position).")
    if len(positions) == 1:
        out.append("The portfolio holds a single asset, so it is not diversified.")
    elif len(positions) < 5:
        out.append("Fewer than 5 holdings limits diversification.")
    if not any(k in BOND_CLASSES for k in by_class):
        out.append("No bond exposure; bonds can reduce overall volatility.")
    if by_class.get("crypto", 0) > 0.10:
        out.append("Crypto is over 10% of the portfolio, which adds significant volatility.")
    unknown = [p.symbol for p in positions if p.name == p.symbol]
    if unknown:
        out.append(f"No reference data for {', '.join(unknown)}; classified as 'stock / Other'.")
    return out
