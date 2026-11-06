"""Bundled reference data for common tickers: name, asset class, sector, expense ratio.

Deliberately static so portfolio analysis stays deterministic and works offline.
Expense ratios are approximate and for education only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetInfo:
    symbol: str
    name: str
    asset_class: str  # stock | bond | etf_equity | etf_bond | cash | real_estate | commodity | crypto
    sector: str
    expense_ratio: float = 0.0  # fraction, e.g. 0.0003 == 0.03%


def _a(symbol, name, asset_class, sector, er=0.0):
    return symbol, AssetInfo(symbol, name, asset_class, sector, er)


ASSETS: dict[str, AssetInfo] = dict(
    [
        # Broad equity ETFs / index funds
        _a("VOO", "Vanguard S&P 500 ETF", "etf_equity", "US Broad Market", 0.0003),
        _a("SPY", "SPDR S&P 500 ETF", "etf_equity", "US Broad Market", 0.0009),
        _a("IVV", "iShares Core S&P 500 ETF", "etf_equity", "US Broad Market", 0.0003),
        _a("VTI", "Vanguard Total Stock Market ETF", "etf_equity", "US Broad Market", 0.0003),
        _a("QQQ", "Invesco QQQ Trust", "etf_equity", "Technology", 0.0020),
        _a("VXUS", "Vanguard Total International Stock ETF", "etf_equity", "International", 0.0005),
        _a("VEA", "Vanguard FTSE Developed Markets ETF", "etf_equity", "International", 0.0005),
        _a("VWO", "Vanguard FTSE Emerging Markets ETF", "etf_equity", "International", 0.0008),
        _a("IWM", "iShares Russell 2000 ETF", "etf_equity", "US Small Cap", 0.0019),
        _a("VNQ", "Vanguard Real Estate ETF", "real_estate", "Real Estate", 0.0013),
        _a("XLK", "Technology Select Sector SPDR", "etf_equity", "Technology", 0.0009),
        _a("XLF", "Financial Select Sector SPDR", "etf_equity", "Financials", 0.0009),
        _a("XLE", "Energy Select Sector SPDR", "etf_equity", "Energy", 0.0009),
        _a("XLV", "Health Care Select Sector SPDR", "etf_equity", "Healthcare", 0.0009),
        # Bond ETFs
        _a("BND", "Vanguard Total Bond Market ETF", "etf_bond", "Bonds", 0.0003),
        _a("AGG", "iShares Core US Aggregate Bond ETF", "etf_bond", "Bonds", 0.0003),
        _a("TLT", "iShares 20+ Year Treasury Bond ETF", "etf_bond", "Bonds", 0.0015),
        _a("SHY", "iShares 1-3 Year Treasury Bond ETF", "etf_bond", "Bonds", 0.0015),
        _a("LQD", "iShares Investment Grade Corporate Bond ETF", "etf_bond", "Bonds", 0.0014),
        _a("BNDX", "Vanguard Total International Bond ETF", "etf_bond", "Bonds", 0.0007),
        # Commodities / crypto
        _a("GLD", "SPDR Gold Shares", "commodity", "Commodities", 0.0040),
        _a("BTC-USD", "Bitcoin", "crypto", "Crypto", 0.0),
        # Large-cap stocks
        _a("AAPL", "Apple", "stock", "Technology"),
        _a("MSFT", "Microsoft", "stock", "Technology"),
        _a("GOOGL", "Alphabet", "stock", "Communication Services"),
        _a("AMZN", "Amazon", "stock", "Consumer Discretionary"),
        _a("NVDA", "NVIDIA", "stock", "Technology"),
        _a("META", "Meta Platforms", "stock", "Communication Services"),
        _a("TSLA", "Tesla", "stock", "Consumer Discretionary"),
        _a("JPM", "JPMorgan Chase", "stock", "Financials"),
        _a("V", "Visa", "stock", "Financials"),
        _a("JNJ", "Johnson & Johnson", "stock", "Healthcare"),
        _a("PG", "Procter & Gamble", "stock", "Consumer Staples"),
        _a("KO", "Coca-Cola", "stock", "Consumer Staples"),
        _a("XOM", "Exxon Mobil", "stock", "Energy"),
        _a("WMT", "Walmart", "stock", "Consumer Staples"),
        _a("DIS", "Walt Disney", "stock", "Communication Services"),
        _a("NFLX", "Netflix", "stock", "Communication Services"),
    ]
)

UNKNOWN = AssetInfo("?", "Unknown", "stock", "Other")

# Company-name aliases so users can type "Apple" instead of "AAPL".
NAME_ALIASES: dict[str, str] = {
    info.name.lower(): sym for sym, info in ASSETS.items() if info.asset_class == "stock"
} | {"google": "GOOGL", "facebook": "META", "bitcoin": "BTC-USD", "s&p 500": "SPY"}

EQUITY_CLASSES = {"stock", "etf_equity"}
BOND_CLASSES = {"bond", "etf_bond"}


def lookup(symbol: str) -> AssetInfo:
    sym = symbol.upper()
    return ASSETS.get(sym, AssetInfo(sym, sym, UNKNOWN.asset_class, UNKNOWN.sector))
