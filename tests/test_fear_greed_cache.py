import os
import data_fetcher


def test_fetch_fear_greed_index_uses_cache(monkeypatch, tmp_path):
    calls = {"count": 0}
    sample = {"data": [{"timestamp": "1700000000", "value": "60"}]}

    def mock_safe_request(url, params=None, **kwargs):
        calls["count"] += 1
        return sample

    monkeypatch.setattr(data_fetcher, "safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)

    df1 = data_fetcher.fetch_fear_greed_index(limit=1)
    df2 = data_fetcher.fetch_fear_greed_index(limit=1)

    assert not df1.empty and not df2.empty
    assert calls["count"] == 1

