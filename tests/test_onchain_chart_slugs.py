import os
import data_fetcher

def test_fetch_onchain_metrics_uses_new_chart_slugs(monkeypatch, tmp_path):
    sample_tx = {"values": [{"x": 1722384000, "y": 123}]}
    sample_active = {"values": [{"x": 1722384000, "y": 456}]}
    captured = []

    def mock_safe_request(url, params=None, **kwargs):
        captured.append(url)
        if "n-transactions" in url:
            return sample_tx
        if "active-addresses" in url:
            return sample_active
        return None

    monkeypatch.setattr(data_fetcher, "safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)

    df = data_fetcher.fetch_onchain_metrics(days=1)

    assert any("n-transactions" in url for url in captured)
    assert any("active-addresses" in url for url in captured)
    assert list(df.columns) == ["Timestamp", "TxVolume", "ActiveAddresses"]
    assert len(df) == 1
    assert df["TxVolume"].iloc[0] == 123
    assert df["ActiveAddresses"].iloc[0] == 456
