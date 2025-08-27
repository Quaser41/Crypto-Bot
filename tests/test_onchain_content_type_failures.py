import os
import json

import data_fetcher

def _setup_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)


def test_fetch_onchain_metrics_html_response(monkeypatch, tmp_path, caplog):
    calls = []

    def fake_safe_request(url, params=None, retry_statuses=None, backoff_on_429=False, headers=None):
        calls.append(url)
        if "api.blockchain.info" in url:
            return {"values": [{"x": 1722384000, "y": 1}]}
        return None

    class HTMLResp:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html>oops</html>"
        def json(self):
            raise ValueError("no json")

    monkeypatch.setattr(data_fetcher, "safe_request", fake_safe_request)
    monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: HTMLResp())
    _setup_cache(monkeypatch, tmp_path)

    with caplog.at_level("WARNING"):
        df = data_fetcher.fetch_onchain_metrics(days=1)

    assert not df.empty
    assert "Non-JSON response" in caplog.text
    assert any("api.blockchain.com" in c for c in calls)
    assert any("api.blockchain.info" in c for c in calls)


def test_fetch_onchain_metrics_invalid_json(monkeypatch, tmp_path, caplog):
    calls = []

    def fake_safe_request(url, params=None, retry_statuses=None, backoff_on_429=False, headers=None):
        calls.append(url)
        if "api.blockchain.info" in url:
            return {"values": [{"x": 1722384000, "y": 2}]}
        return None

    class BadJSONResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = "bad"
        def json(self):
            raise json.JSONDecodeError("Expecting value", "", 0)

    monkeypatch.setattr(data_fetcher, "safe_request", fake_safe_request)
    monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: BadJSONResp())
    _setup_cache(monkeypatch, tmp_path)

    with caplog.at_level("WARNING"):
        df = data_fetcher.fetch_onchain_metrics(days=1)

    assert not df.empty
    assert "JSON decode error" in caplog.text
    assert any("api.blockchain.com" in c for c in calls)
    assert any("api.blockchain.info" in c for c in calls)
