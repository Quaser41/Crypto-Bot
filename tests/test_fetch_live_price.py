import pytest

from data_fetcher import fetch_live_price


def test_fetch_live_price_cryptocompare(monkeypatch):
    """Returns price from CryptoCompare and stops after first source."""
    calls = []

    def fake_safe_request(url, params=None, **kwargs):
        calls.append(url)
        if "cryptocompare" in url:
            return {"USD": 123.0}
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr("data_fetcher.safe_request", fake_safe_request)
    monkeypatch.setattr("data_fetcher.cached_fetch", lambda *a, **k: None)
    monkeypatch.setattr("data_fetcher.update_cache", lambda *a, **k: None)

    price = fetch_live_price("BTC")
    assert price == pytest.approx(123.0)
    assert any("cryptocompare" in url for url in calls)
    assert len(calls) == 1


def test_fetch_live_price_uses_cache(monkeypatch):
    """Uses cached price without performing network request."""
    cache = {"live_price:ETH": 456.0}
    monkeypatch.setattr("data_fetcher.cached_fetch", lambda key, ttl=60: cache.get(key))
    monkeypatch.setattr("data_fetcher.update_cache", lambda *a, **k: None)

    def fail_safe_request(*args, **kwargs):
        raise AssertionError("safe_request should not be called when cache hit")

    monkeypatch.setattr("data_fetcher.safe_request", fail_safe_request)
    price = fetch_live_price("ETH")
    assert price == pytest.approx(456.0)


def test_fetch_live_price_fallback_coinbase(monkeypatch):
    """Falls back to Coinbase when CryptoCompare has no data."""
    calls = []

    def fake_safe_request(url, params=None, **kwargs):
        calls.append(url)
        if "cryptocompare" in url:
            return None
        if "coinbase.com" in url:
            return {"price": "99.0"}
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr("data_fetcher.safe_request", fake_safe_request)
    monkeypatch.setattr("data_fetcher.cached_fetch", lambda *a, **k: None)
    monkeypatch.setattr("data_fetcher.update_cache", lambda *a, **k: None)
    monkeypatch.setattr("data_fetcher.resolve_symbol_coinbase", lambda symbol: f"{symbol}-USD")

    price = fetch_live_price("BTC")
    assert price == pytest.approx(99.0)
    assert any("cryptocompare" in url for url in calls)
    assert any("coinbase" in url for url in calls)
