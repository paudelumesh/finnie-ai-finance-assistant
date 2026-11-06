"""Market data layer: provider abstraction, TTL cache, retry with backoff, stale/mock fallback."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Protocol

from src.utils.logging import get_logger

log = get_logger("market_data")


class MarketDataError(RuntimeError):
    """A provider could not return data for a symbol."""


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    previous_close: float
    as_of: datetime
    source: str = "live"  # live | cache | stale | mock
    currency: str = "USD"

    @property
    def change(self) -> float:
        return self.price - self.previous_close

    @property
    def change_pct(self) -> float:
        return (self.change / self.previous_close * 100) if self.previous_close else 0.0

    def with_source(self, source: str) -> "Quote":
        return Quote(self.symbol, self.price, self.previous_close, self.as_of, source, self.currency)


@dataclass(frozen=True)
class PricePoint:
    day: str  # ISO date
    close: float


@dataclass
class Trend:
    symbol: str
    points: list[PricePoint] = field(default_factory=list)
    source: str = "live"

    @property
    def change_pct(self) -> float:
        if len(self.points) < 2 or not self.points[0].close:
            return 0.0
        return (self.points[-1].close / self.points[0].close - 1) * 100

    @property
    def direction(self) -> str:
        pct = self.change_pct
        return "up" if pct > 1 else "down" if pct < -1 else "flat"


class QuoteProvider(Protocol):
    async def get_quote(self, symbol: str) -> Quote: ...
    async def get_history(self, symbol: str, days: int) -> list[PricePoint]: ...


class YFinanceProvider:
    """yfinance is synchronous, so each call runs in a worker thread."""

    async def get_quote(self, symbol: str) -> Quote:
        return await asyncio.to_thread(self._quote, symbol)

    async def get_history(self, symbol: str, days: int) -> list[PricePoint]:
        return await asyncio.to_thread(self._history, symbol, days)

    @staticmethod
    def _quote(symbol: str) -> Quote:
        import yfinance as yf

        try:
            hist = yf.Ticker(symbol).history(period="5d")
        except Exception as exc:
            raise MarketDataError(f"yfinance failed for {symbol}: {exc}") from exc
        closes = hist["Close"].dropna() if "Close" in hist else []
        if len(closes) == 0:
            raise MarketDataError(f"no price data for {symbol}")
        price = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) > 1 else price
        return Quote(symbol.upper(), price, prev, datetime.now(timezone.utc), "live")

    @staticmethod
    def _history(symbol: str, days: int) -> list[PricePoint]:
        import yfinance as yf

        try:
            hist = yf.Ticker(symbol).history(period=f"{max(days, 5)}d")
        except Exception as exc:
            raise MarketDataError(f"yfinance failed for {symbol}: {exc}") from exc
        closes = hist["Close"].dropna() if "Close" in hist else []
        if len(closes) == 0:
            raise MarketDataError(f"no history for {symbol}")
        return [PricePoint(idx.strftime("%Y-%m-%d"), float(v)) for idx, v in closes.items()]


_MOCK_PRICES = {
    "AAPL": 190.0, "MSFT": 410.0, "GOOGL": 165.0, "AMZN": 180.0, "NVDA": 120.0,
    "META": 480.0, "TSLA": 240.0, "JPM": 200.0, "V": 270.0, "JNJ": 155.0,
    "PG": 165.0, "KO": 62.0, "XOM": 110.0, "WMT": 68.0, "DIS": 100.0, "NFLX": 620.0,
    "VOO": 480.0, "SPY": 520.0, "IVV": 525.0, "VTI": 255.0, "QQQ": 440.0,
    "VXUS": 60.0, "VEA": 50.0, "VWO": 43.0, "IWM": 205.0, "VNQ": 88.0,
    "BND": 72.0, "AGG": 98.0, "TLT": 92.0, "SHY": 82.0, "LQD": 108.0,
    "BNDX": 49.0, "GLD": 215.0, "BTC-USD": 65000.0,
    "XLK": 210.0, "XLF": 42.0, "XLE": 92.0, "XLV": 145.0,
}


class MockProvider:
    """Deterministic offline data so the demo never dies on a rate limit."""

    @staticmethod
    def _base(symbol: str) -> float:
        """Only symbols with bundled sample prices; we never fabricate prices for unknown tickers."""
        sym = symbol.upper()
        if sym not in _MOCK_PRICES:
            raise MarketDataError(f"no sample data for {sym}")
        return _MOCK_PRICES[sym]

    async def get_quote(self, symbol: str) -> Quote:
        base = self._base(symbol)
        return Quote(symbol.upper(), base, round(base * 0.995, 2), datetime.now(timezone.utc), "mock")

    async def get_history(self, symbol: str, days: int) -> list[PricePoint]:
        base, n = self._base(symbol), max(days, 5)
        today = datetime.now(timezone.utc).date()
        return [
            PricePoint(
                (today.fromordinal(today.toordinal() - (n - 1 - i))).isoformat(),
                round(base * (0.92 + 0.08 * i / (n - 1)), 2),
            )
            for i in range(n)
        ]


class MarketDataService:
    """Cache (TTL) -> provider with retries -> stale cache -> mock, tagging every result's source."""

    def __init__(
        self,
        provider: QuoteProvider,
        ttl_seconds: int = 1800,
        max_retries: int = 2,
        use_mock_on_failure: bool = True,
        fallback: QuoteProvider | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.provider = provider
        self.ttl = ttl_seconds
        self.max_retries = max_retries
        self.fallback = (fallback or MockProvider()) if use_mock_on_failure else None
        self._clock, self._sleep = clock, sleep
        self._quotes: dict[str, tuple[float, Quote]] = {}
        self._history: dict[tuple[str, int], tuple[float, list[PricePoint]]] = {}

    async def _with_retries(self, call: Callable[[], Awaitable]):
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await call()
            except Exception as exc:  # provider errors are heterogeneous
                last = exc
                log.warning("provider attempt %d failed: %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    await self._sleep(2**attempt)  # exponential backoff: 1s, 2s, ...
        raise MarketDataError(str(last))

    def clear_cache(self) -> None:
        """Drop cached quotes/history (used by the UI's refresh button)."""
        self._quotes.clear()
        self._history.clear()

    async def get_quote(self, symbol: str) -> Quote:
        sym = symbol.upper()
        now = self._clock()
        cached = self._quotes.get(sym)
        if cached and now - cached[0] < self.ttl:
            return cached[1].with_source("cache")
        try:
            quote = await self._with_retries(lambda: self.provider.get_quote(sym))
            self._quotes[sym] = (now, quote)
            return quote
        except MarketDataError:
            if cached:
                return cached[1].with_source("stale")
            if self.fallback:
                return await self.fallback.get_quote(sym)
            raise

    async def get_trend(self, symbol: str, days: int = 30) -> Trend:
        sym = symbol.upper()
        key, now = (sym, days), self._clock()
        cached = self._history.get(key)
        if cached and now - cached[0] < self.ttl:
            return Trend(sym, cached[1], "cache")
        try:
            pts = await self._with_retries(lambda: self.provider.get_history(sym, days))
            self._history[key] = (now, pts)
            return Trend(sym, pts, "live")
        except MarketDataError:
            if cached:
                return Trend(sym, cached[1], "stale")
            if self.fallback:
                return Trend(sym, await self.fallback.get_history(sym, days), "mock")
            raise

    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        results = await asyncio.gather(*(self.get_quote(s) for s in symbols))
        return {q.symbol: q for q in results}
