import os

import data_fetcher


def test_fetch_onchain_metrics_caches_fallback_df(monkeypatch, tmp_path):
    calls = {"count": 0}

    def mock_safe_request(*args, **kwargs):
        calls["count"] += 1
        return None

    monkeypatch.setattr(data_fetcher, "safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)

    df1 = data_fetcher.fetch_onchain_metrics(days=3)
    df2 = data_fetcher.fetch_onchain_metrics(days=3)

    assert not df1.empty and not df2.empty
    assert (df1["TxVolume"] == 0).all()
    assert (df1["ActiveAddresses"] == 0).all()
    # Two endpoints plus a legacy-domain retry for each
    assert calls["count"] == 4


def test_fetch_onchain_metrics_returns_data(monkeypatch, tmp_path):
    sample_tx = {
        "values": [
            {"x": 1722384000, "y": 1000},
            {"x": 1722470400, "y": 1500},
        ]
    }
    sample_active = {
        "values": [
            {"x": 1722384000, "y": 200},
            {"x": 1722470400, "y": 250},
        ]
    }
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

