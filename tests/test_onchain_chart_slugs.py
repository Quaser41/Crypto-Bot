import os
import data_fetcher

def test_fetch_onchain_metrics_uses_new_chart_slugs(monkeypatch, tmp_path):
    sample_tx = {"values": [{"x": 1722384000, "y": 123}]}
    sample_active = {"values": [{"x": 1722384000, "y": 456}]}
    captured = []

    def mock_safe_request(url, params=None, **kwargs):
        captured.append((url, params))
        if "n-transactions" in url:
            return sample_tx
        if "activeaddresses" in url:
            return sample_active
        return None

    monkeypatch.setattr(data_fetcher, "safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)

    df = data_fetcher.fetch_onchain_metrics(days=1)

    assert any("n-transactions" in url for url, _ in captured)
    assert any("activeaddresses" in url for url, _ in captured)
    assert all(
        params.get("format") == "json"
        and params.get("cors") == "true"
        and params.get("timespan") == "1days"
        for _, params in captured
    )
    assert any("api.blockchain.info/charts" in url for url, _ in captured)
    assert list(df.columns) == ["Timestamp", "TxVolume", "ActiveAddresses"]
    assert len(df) == 1
    assert df["TxVolume"].iloc[0] == 123
    assert df["ActiveAddresses"].iloc[0] == 456
