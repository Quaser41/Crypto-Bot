import pandas as pd
import data_fetcher


def test_fetch_coinbase_ohlcv_handles_timezone_cache(monkeypatch):
    ts_cached = pd.Timestamp.utcnow() - pd.Timedelta(hours=2)
    df_cached = pd.DataFrame(
        [{
            "Timestamp": ts_cached,
            "Open": 1,
            "High": 1,
            "Low": 1,
            "Close": 1,
            "Volume": 1,
        }]
    )

    def mock_load(symbol, interval):
        return df_cached, 999999

    sample_ts = int((ts_cached + pd.Timedelta(hours=1)).timestamp())
    sample_data = [[sample_ts, 1, 1, 1, 1, 1]]

    monkeypatch.setattr(data_fetcher, "load_ohlcv_cache", mock_load)
    monkeypatch.setattr(data_fetcher, "save_ohlcv_cache", lambda *a, **k: None)
    monkeypatch.setattr(data_fetcher, "resolve_symbol_coinbase", lambda s: "BTC-USD")
    monkeypatch.setattr(
        data_fetcher,
        "safe_request",
        lambda url, params=None, backoff_on_429=False: sample_data,
    )

    df = data_fetcher.fetch_coinbase_ohlcv("btc", interval="1h", limit=1, ttl=0)
    assert not df.empty
    assert str(df["Timestamp"].dtype) == "datetime64[ns, UTC]"


def test_fetch_coinbase_ohlcv_handles_timezone_no_cache(monkeypatch):
    sample_ts = int(pd.Timestamp.utcnow().timestamp())
    sample_data = [[sample_ts, 1, 1, 1, 1, 1]]

    monkeypatch.setattr(
        data_fetcher, "load_ohlcv_cache", lambda symbol, interval: (None, None)
    )
    monkeypatch.setattr(data_fetcher, "save_ohlcv_cache", lambda *a, **k: None)
    monkeypatch.setattr(data_fetcher, "resolve_symbol_coinbase", lambda s: "BTC-USD")
    monkeypatch.setattr(
        data_fetcher,
        "safe_request",
        lambda url, params=None, backoff_on_429=False: sample_data,
    )

    df = data_fetcher.fetch_coinbase_ohlcv("btc", interval="1h", limit=1, ttl=0)
    assert not df.empty
    assert str(df["Timestamp"].dtype) == "datetime64[ns, UTC]"
