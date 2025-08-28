import os
import types

import data_fetcher


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)
    monkeypatch.setattr(data_fetcher, "SEEN_404_URLS", set())
    monkeypatch.setattr(data_fetcher, "SEEN_NON_JSON_URLS", set())


def test_fetch_onchain_metrics_html_scrape_success(monkeypatch, tmp_path):
    calls = []

    def mock_safe_request(*a, **k):
        return None

    tx_html = "<script>var data = [[1722384000000,1000],[1722470400000,1500]];</script>"
    active_html = "<script>var data = [[1722384000000,200],[1722470400000,250]];</script>"

    def mock_get(url, params=None, **kwargs):
        calls.append(url)

        class Resp:
            headers = {"content-type": "text/html"}

            def json(self):
                raise ValueError("no json")

        if params and params.get("format") == "json":
            resp = Resp()
            resp.status_code = 404
            resp.text = "not found"
            return resp
        resp = Resp()
        resp.status_code = 200
        if "n-transactions" in url:
            resp.text = tx_html
        else:
            resp.text = active_html
        return resp

    monkeypatch.setattr(data_fetcher, "safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher.requests, "get", mock_get)
    _setup(monkeypatch, tmp_path)

    df = data_fetcher.fetch_onchain_metrics(days=2)

    assert len(calls) == 4  # two API inspections + two scrapes
    assert list(df.columns) == ["Timestamp", "TxVolume", "ActiveAddresses"]
    assert df["TxVolume"].tolist() == [1000, 1500]
    assert df["ActiveAddresses"].tolist() == [200, 250]


def test_fetch_onchain_metrics_html_scrape_failure(monkeypatch, tmp_path):
    calls = []

    def mock_safe_request(*a, **k):
        return None

    class NotFound:
        status_code = 404
        headers = {"content-type": "text/html"}
        text = "missing"

        def json(self):
            raise ValueError("no json")

    def mock_get(url, params=None, **kwargs):
        calls.append(url)
        return NotFound()

    monkeypatch.setattr(data_fetcher, "safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher.requests, "get", mock_get)
    _setup(monkeypatch, tmp_path)

    df = data_fetcher.fetch_onchain_metrics(days=1)

    assert len(calls) == 4
    assert (df["TxVolume"] == 0).all()
    assert (df["ActiveAddresses"] == 0).all()
