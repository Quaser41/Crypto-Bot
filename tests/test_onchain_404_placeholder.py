import os

import data_fetcher


def _setup(monkeypatch, tmp_path):
    monkeypatch.setenv("BLOCKCHAIN_API_KEY", "test-key")
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)
    monkeypatch.setattr(data_fetcher, "SEEN_404_URLS", set())
    monkeypatch.setattr(data_fetcher, "SEEN_NON_JSON_URLS", set())


def test_fetch_onchain_metrics_404_returns_placeholder(monkeypatch, tmp_path, caplog):
    calls = []

    def fake_safe_request(url, params=None, retry_statuses=None, backoff_on_429=False, headers=None):
        calls.append(url)
        return None

    class NotFound:
        status_code = 404
        headers = {"content-type": "text/html"}
        text = "not found"

        def json(self):
            raise ValueError("no json")

    monkeypatch.setattr(data_fetcher, "safe_request", fake_safe_request)
    monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: NotFound())
    _setup(monkeypatch, tmp_path)

    with caplog.at_level("WARNING"):
        df1 = data_fetcher.fetch_onchain_metrics(days=1)
        df2 = data_fetcher.fetch_onchain_metrics(days=1)

    assert not df1.empty and not df2.empty
    assert (df1["TxVolume"] == 0).all() and (df1["ActiveAddresses"] == 0).all()
    # ``safe_request`` should be invoked for each chart endpoint once; subsequent
    # calls use the cached placeholder
    assert len(calls) == 2
