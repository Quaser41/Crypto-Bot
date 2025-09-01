import pandas as pd
import data_fetcher


def _fake_df():
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC"),
            "Open": [1, 1],
            "High": [1, 1],
            "Low": [1, 1],
            "Close": [1, 1],
            "Volume": [100, 100],
        }
    )


import pytest


@pytest.mark.parametrize("interval,expected", [("15m", "15m"), ("1h", "1h")])
def test_fetch_from_yfinance_interval(monkeypatch, interval, expected):
    called = {}

    def fake_download(ticker, period, interval, progress, auto_adjust):
        called["ticker"] = ticker
        called["period"] = period
        called["interval"] = interval
        called["auto_adjust"] = auto_adjust
        df = _fake_df().copy()
        df.set_index("Timestamp", inplace=True)
        return df

    monkeypatch.setattr(data_fetcher.yf, "download", fake_download)
    df = data_fetcher.fetch_from_yfinance("btc", interval=interval, days=1)
    assert called["interval"] == expected
    assert called["auto_adjust"] is False
    assert not df.empty
    assert "Timestamp" in df.columns


def test_fetch_ohlcv_smart_uses_yfinance(monkeypatch):
    df = _fake_df()
    monkeypatch.setattr(data_fetcher, "load_ohlcv_cache", lambda *a, **k: (None, None))
    monkeypatch.setattr(data_fetcher, "save_ohlcv_cache", lambda *a, **k: None)
    monkeypatch.setattr(data_fetcher, "fetch_coinbase_ohlcv", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(data_fetcher, "fetch_dexscreener_ohlcv", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(data_fetcher, "fetch_coingecko_ohlcv", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(data_fetcher, "fetch_from_yfinance", lambda *a, **k: df)
    monkeypatch.setattr(data_fetcher, "DATA_SOURCES", ["yfinance"])

    result = data_fetcher.fetch_ohlcv_smart("btc", interval="15m", days=1)
    assert result.equals(df)
