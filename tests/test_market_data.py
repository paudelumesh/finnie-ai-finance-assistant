from datetime import datetime, timezone

import pytest

from src.data.market_data import (
    MarketDataError, MarketDataService, MockProvider, PricePoint, Quote, Trend,
)


def q(symbol="AAPL", price=100.0, prev=98.0):
    return Quote(symbol, price, prev, datetime.now(timezone.utc))


class FlakyProvider:
    def __init__(self, fail_times=0, price=100.0):
        self.fail_times, self.calls, self.price = fail_times, 0, price

    async def get_quote(self, symbol):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise MarketDataError("rate limited")
        return q(symbol, self.price)

    async def get_history(self, symbol, days):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise MarketDataError("rate limited")
        return [PricePoint("2026-01-01", 100.0), PricePoint("2026-01-02", 110.0)]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


async def no_sleep(_):
    pass


def service(provider, clock=None, **kw):
    return MarketDataService(provider, clock=clock or Clock(), sleep=no_sleep, **kw)


def test_quote_change_maths():
    quote = q(price=110, prev=100)
    assert quote.change == 10 and quote.change_pct == 10
    assert q(price=1, prev=0).change_pct == 0


async def test_cache_hit_within_ttl_then_expiry():
    clock, prov = Clock(), FlakyProvider()
    svc = service(prov, clock, ttl_seconds=100)
    assert (await svc.get_quote("aapl")).source == "live"
    clock.t = 50
    assert (await svc.get_quote("AAPL")).source == "cache" and prov.calls == 1
    clock.t = 101
    assert (await svc.get_quote("AAPL")).source == "live" and prov.calls == 2


async def test_retries_with_backoff_then_succeeds():
    sleeps = []

    async def rec(s):
        sleeps.append(s)

    prov = FlakyProvider(fail_times=2)
    svc = MarketDataService(prov, max_retries=2, sleep=rec)
    assert (await svc.get_quote("MSFT")).source == "live"
    assert sleeps == [1, 2]


async def test_falls_back_to_stale_cache_when_provider_dies():
    clock, prov = Clock(), FlakyProvider()
    svc = service(prov, clock, ttl_seconds=10, max_retries=0)
    await svc.get_quote("AAPL")
    clock.t = 100
    prov.fail_times = 10**6
    assert (await svc.get_quote("AAPL")).source == "stale"


async def test_falls_back_to_mock_without_cache():
    svc = service(FlakyProvider(fail_times=10**6), max_retries=0)
    quote = await svc.get_quote("AAPL")
    assert quote.source == "mock" and quote.price > 0


async def test_raises_when_mock_disabled():
    svc = service(FlakyProvider(fail_times=10**6), max_retries=0, use_mock_on_failure=False)
    with pytest.raises(MarketDataError):
        await svc.get_quote("AAPL")


async def test_trend_live_cache_stale_mock():
    clock, prov = Clock(), FlakyProvider()
    svc = service(prov, clock, ttl_seconds=10, max_retries=0)
    t = await svc.get_trend("AAPL", 30)
    assert t.source == "live" and t.direction == "up" and round(t.change_pct) == 10
    assert (await svc.get_trend("AAPL", 30)).source == "cache"
    clock.t, prov.fail_times = 100, 10**6
    assert (await svc.get_trend("AAPL", 30)).source == "stale"
    mock = await service(FlakyProvider(fail_times=10**6), max_retries=0).get_trend("VOO", 10)
    assert mock.source == "mock" and len(mock.points) == 10


async def test_trend_raises_when_mock_disabled():
    svc = service(FlakyProvider(fail_times=10**6), max_retries=0, use_mock_on_failure=False)
    with pytest.raises(MarketDataError):
        await svc.get_trend("AAPL")


async def test_get_quotes_batch():
    out = await service(FlakyProvider()).get_quotes(["aapl", "msft"])
    assert set(out) == {"AAPL", "MSFT"}


async def test_mock_provider_is_deterministic():
    m = MockProvider()
    assert (await m.get_quote("QQQ")).price == (await m.get_quote("QQQ")).price
    with pytest.raises(MarketDataError):
        await m.get_quote("UNKNOWNX")


async def test_unknown_symbol_is_not_fabricated_on_fallback():
    svc = service(FlakyProvider(fail_times=10**6), max_retries=0)
    with pytest.raises(MarketDataError):
        await svc.get_quote("UNKNOWNX")


def test_trend_edge_cases():
    assert Trend("X").change_pct == 0 and Trend("X").direction == "flat"
    down = Trend("X", [PricePoint("a", 100), PricePoint("b", 90)])
    assert down.direction == "down"
