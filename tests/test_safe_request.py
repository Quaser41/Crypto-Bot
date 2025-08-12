import types

import pytest

from data_fetcher import safe_request


class DummyResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json


def test_safe_request_404_returns_none_immediately(monkeypatch):
    calls = {"count": 0}

    def fake_get(url, params=None, timeout=10):
        calls["count"] += 1
        return DummyResponse(status_code=404)

    sleeps = []

    monkeypatch.setattr("data_fetcher.requests.get", fake_get)
    monkeypatch.setattr("data_fetcher.wait_for_slot", lambda *a, **k: None)
    monkeypatch.setattr("data_fetcher.time.sleep", lambda s: sleeps.append(s))

    result = safe_request("https://example.com")
    assert result is None
    assert calls["count"] == 1, "Request should not be retried on 404"
    assert sleeps == [], "Should not wait between retries when 404 occurs"
