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

