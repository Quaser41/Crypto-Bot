import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import data_fetcher


def test_fetch_onchain_metrics_missing_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("BLOCKCHAIN_API_KEY", raising=False)
    calls = {"count": 0}

    def mock_safe_request(*args, **kwargs):
        calls["count"] += 1
        return {}

    monkeypatch.setattr(data_fetcher, "safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)

    df = data_fetcher.fetch_onchain_metrics(days=2)
    assert not df.empty
    assert (df["TxVolume"] == 0).all() and (df["ActiveAddresses"] == 0).all()
    assert calls["count"] == 0


def test_fetch_onchain_metrics_401_returns_placeholder(monkeypatch, tmp_path):
    monkeypatch.setenv("BLOCKCHAIN_API_KEY", "bad-key")
    calls = []

    def fake_safe_request(url, params=None, retry_statuses=None, backoff_on_429=False, headers=None):
        calls.append(url)
        return None

    class Unauthorized:
        status_code = 401
        headers = {"content-type": "text/html"}
        text = "unauthorized"

        def json(self):
            raise ValueError("no json")

    req_calls = {"count": 0}

    def mock_get(*a, **k):
        req_calls["count"] += 1
        return Unauthorized()

    monkeypatch.setattr(data_fetcher, "safe_request", fake_safe_request)
    monkeypatch.setattr(data_fetcher.requests, "get", mock_get)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)
    monkeypatch.setattr(data_fetcher, "SEEN_404_URLS", set())
    monkeypatch.setattr(data_fetcher, "SEEN_NON_JSON_URLS", set())

    df1 = data_fetcher.fetch_onchain_metrics(days=1)
    df2 = data_fetcher.fetch_onchain_metrics(days=1)

    assert not df1.empty and not df2.empty
    assert (df1["TxVolume"] == 0).all()
    assert len(calls) == 2
    assert req_calls["count"] == 2
