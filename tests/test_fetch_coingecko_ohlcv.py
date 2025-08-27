import os
import types
import data_fetcher


def test_fetch_coingecko_ohlcv_retries_on_401(monkeypatch, tmp_path):
    responses = [
        types.SimpleNamespace(
            status_code=401,
            json=lambda: {},
            headers={"content-type": "application/json"},
            text="{}",
        ),
        types.SimpleNamespace(
            status_code=200,
            json=lambda: {"prices": [[0, 1]]},
            headers={"content-type": "application/json"},
            text='{"prices": [[0,1]]}',
        ),
    ]
    called_headers = []

    def fake_get(url, params=None, timeout=10, headers=None):
        called_headers.append(headers)
        return responses.pop(0)

    monkeypatch.setattr(data_fetcher.requests, "get", fake_get)
    monkeypatch.setattr(data_fetcher, "wait_for_slot", lambda *a, **k: None)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)
    monkeypatch.setenv("COINGECKO_API_KEY", "test-key")

    df = data_fetcher.fetch_coingecko_ohlcv("bitcoin")
    assert not df.empty
    assert called_headers[0] is None
    assert called_headers[1] == {"x-cg-demo-api-key": "test-key"}
    assert len(called_headers) == 2

