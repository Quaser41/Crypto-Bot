import types
import pytest

from data_fetcher import safe_request


def test_safe_request_no_retry_on_404(monkeypatch):
    """A 404 response should not be retried."""
    calls = []

    class Resp:
        status_code = 404
        def json(self):
            return {}

    def fake_get(url, params=None, timeout=10):
        calls.append(1)
        return Resp()

    sleep_calls = []
    monkeypatch.setattr("data_fetcher.session.get", fake_get)
    monkeypatch.setattr("data_fetcher.time.sleep", lambda x: sleep_calls.append(x))
    monkeypatch.setattr("data_fetcher.wait_for_slot", lambda *a, **k: None)

    result = safe_request("http://example.com", max_retries=3)
    assert result is None
    assert len(calls) == 1
    assert sleep_calls == []


def test_safe_request_retries_on_429_with_backoff(monkeypatch):
    """429 responses should trigger retry with exponential backoff."""
    status_codes = [429, 200]

    def fake_get(url, params=None, timeout=10):
        code = status_codes.pop(0)
        resp = types.SimpleNamespace(
            status_code=code,
            json=lambda: {"ok": True},
            headers={"content-type": "application/json"},
            text="{\"ok\": true}",
        )
        return resp

    sleep_calls = []
    monkeypatch.setattr("data_fetcher.session.get", fake_get)
    monkeypatch.setattr("data_fetcher.time.sleep", lambda x: sleep_calls.append(x))
    monkeypatch.setattr("data_fetcher.wait_for_slot", lambda *a, **k: None)

    result = safe_request("http://example.com")
    assert result == {"ok": True}
    # Should retry once after the initial 429
    assert not status_codes  # list empty => both responses used
    assert sleep_calls and sleep_calls[0] == 5  # exponential backoff from retry_delay

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

    monkeypatch.setattr("data_fetcher.session.get", fake_get)
    monkeypatch.setattr("data_fetcher.wait_for_slot", lambda *a, **k: None)
    monkeypatch.setattr("data_fetcher.time.sleep", lambda s: sleeps.append(s))

    result = safe_request("https://example.com")
    assert result is None
    assert calls["count"] == 1, "Request should not be retried on 404"
    assert sleeps == [], "Should not wait between retries when 404 occurs"


def test_safe_request_401_returns_none_and_logs(monkeypatch, caplog):
    calls = {"count": 0}

    class Resp:
        status_code = 401
        headers = {"content-type": "application/json"}
        text = "{}"
        def json(self):
            return {}

    def fake_get(url, params=None, timeout=10):
        calls["count"] += 1
        return Resp()

    sleeps = []
    monkeypatch.setattr("data_fetcher.session.get", fake_get)
    monkeypatch.setattr("data_fetcher.wait_for_slot", lambda *a, **k: None)
    monkeypatch.setattr("data_fetcher.time.sleep", lambda s: sleeps.append(s))

    with caplog.at_level("ERROR"):
        result = safe_request("https://example.com")

    assert result is None
    assert calls["count"] == 1
    assert sleeps == []
    assert any("401 Unauthorized" in r.message for r in caplog.records)


def test_safe_request_uses_session_and_passes_headers(monkeypatch):
    """Verify that the module-level session is used and headers propagate."""
    calls = []

    def fake_get(url, params=None, timeout=10, headers=None):
        calls.append(headers)
        return types.SimpleNamespace(
            status_code=200,
            json=lambda: {"ok": True},
            headers={"content-type": "application/json"},
            text="{\"ok\": true}",
        )

    # Ensure session.get is used instead of requests.get
    monkeypatch.setattr("data_fetcher.session.get", fake_get)
    monkeypatch.setattr(
        "data_fetcher.requests.get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("requests.get called")),
    )
    monkeypatch.setattr("data_fetcher.wait_for_slot", lambda *a, **k: None)

    result = safe_request("http://example.com", headers={"X-Test": "1"})
    assert result == {"ok": True}
    assert calls and calls[0]["X-Test"] == "1"

