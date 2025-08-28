import os
import json

import data_fetcher


def _setup_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)
    monkeypatch.setattr(data_fetcher, "SEEN_NON_JSON_URLS", set())


def test_fetch_onchain_metrics_html_response(monkeypatch, tmp_path, caplog):
    calls = []

    def fake_safe_request(
        url, params=None, retry_statuses=None, backoff_on_429=False, headers=None
    ):
        calls.append((url, params))
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
    assert len(calls) == 2
    assert all("api.blockchain.info/charts" in url for url, _ in calls)
    assert all(
        p.get("cors") == "true" and p.get("format") == "json" and p.get("timespan") == "1days"
        for _, p in calls
    )


def test_fetch_onchain_metrics_invalid_json(monkeypatch, tmp_path, caplog):
    calls = []

    def fake_safe_request(
        url, params=None, retry_statuses=None, backoff_on_429=False, headers=None
    ):
        calls.append((url, params))
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
    assert len(calls) == 2
    assert all("api.blockchain.info/charts" in url for url, _ in calls)
    assert all(
        p.get("cors") == "true" and p.get("format") == "json" and p.get("timespan") == "1days"
        for _, p in calls
    )
