import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import data_fetcher


def test_concurrent_calls_preserve_source_order(monkeypatch):
    """Concurrent ``fetch_ohlcv_smart`` calls should not interfere with one
    another's source ordering."""

    # Avoid any cached data and network requests
    monkeypatch.setattr(data_fetcher, "load_ohlcv_cache", lambda *a, **k: (None, None))
    monkeypatch.setattr(data_fetcher, "save_ohlcv_cache", lambda *a, **k: None)

    # Always resolve symbols successfully
    monkeypatch.setattr(data_fetcher, "resolve_symbol_coinbase", lambda s: True)
    monkeypatch.setattr(data_fetcher, "resolve_symbol_binance_us", lambda s: True)
    monkeypatch.setattr(data_fetcher, "resolve_symbol_binance_global", lambda s: True)

    # Simplify the data source list to keep the test focused
    monkeypatch.setattr(
        data_fetcher,
        "DATA_SOURCES",
        ["coinbase", "binance_us", "binance"],
    )
    monkeypatch.setattr(data_fetcher, "ENABLE_BINANCE_GLOBAL", True)

    barrier = threading.Barrier(2)
    thread_local = threading.local()

    def coinbase_fetch(symbol, **params):
        barrier.wait()  # ensure both threads hit Coinbase before proceeding
        thread_local.order.append("coinbase")
        return pd.DataFrame({"Timestamp": pd.date_range("2024", periods=10), "Close": range(10)})

    def binance_us_fetch(symbol, **params):
        thread_local.order.append("binance_us")
        return pd.DataFrame()  # empty so the next source is tried

    def binance_fetch(symbol, **params):
        thread_local.order.append("binance")
        return pd.DataFrame({"Timestamp": pd.date_range("2024", periods=100), "Close": range(100)})

    # Patch the fetchers
    monkeypatch.setattr(data_fetcher, "fetch_coinbase_ohlcv", coinbase_fetch)
    monkeypatch.setattr(data_fetcher, "fetch_binance_us_ohlcv", binance_us_fetch)
    monkeypatch.setattr(data_fetcher, "fetch_binance_ohlcv", binance_fetch)

    def run():
        thread_local.order = []
        df = data_fetcher.fetch_ohlcv_smart("BTC", interval="1d", limit=100)
        assert not df.empty
        return list(thread_local.order)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: run(), range(2)))

    assert results[0] == ["coinbase", "binance_us", "binance"]
    assert results[1] == ["coinbase", "binance_us", "binance"]
