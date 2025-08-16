import os

import data_fetcher


def test_fetch_onchain_metrics_caches_empty_df(monkeypatch, tmp_path):
    calls = {"count": 0}

    def mock_safe_request(*args, **kwargs):
        calls["count"] += 1
        return None

    monkeypatch.setattr(data_fetcher, "safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)

    df1 = data_fetcher.fetch_onchain_metrics()
    df2 = data_fetcher.fetch_onchain_metrics()

    assert df1.empty and df2.empty
    assert calls["count"] == 2


def test_fetch_onchain_metrics_returns_data(monkeypatch, tmp_path):
    sample_tx = {"data": [["2024-07-01", 1000], ["2024-07-02", 1500]]}
    sample_active = {"data": [["2024-07-01", 200], ["2024-07-02", 250]]}
    responses = [sample_tx, sample_active]

    def mock_safe_request(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(data_fetcher, "safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)

    df = data_fetcher.fetch_onchain_metrics(days=2)

    assert not df.empty
    assert set(["Timestamp", "TxVolume", "ActiveAddresses"]) == set(df.columns)
    assert len(df) == 2

