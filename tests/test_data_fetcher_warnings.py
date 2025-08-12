import warnings

from data_fetcher import fetch_fear_greed_index, fetch_onchain_metrics


def test_no_future_warning_on_timestamp_parsing(monkeypatch):
    fg_sample = {
        "data": [
            {"timestamp": "1700000000", "value": "60"},
            {"timestamp": "1700003600", "value": "55"},
        ]
    }
    tx_sample = {
        "values": [
            {"x": "1700000000", "y": 12345},
            {"x": "1700003600", "y": 23456},
        ]
    }
    active_sample = {
        "values": [
            {"x": "1700000000", "y": 1000},
            {"x": "1700003600", "y": 1200},
        ]
    }

    def mock_safe_request(url, params=None, **kwargs):
        if "fng" in url:
            return fg_sample
        if "transaction-volume" in url:
            return tx_sample
        if "activeaddresses" in url:
            return active_sample
        return None

    monkeypatch.setattr('data_fetcher.safe_request', mock_safe_request)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        fg_df = fetch_fear_greed_index()
        onchain_df = fetch_onchain_metrics()

    assert not fg_df.empty
    assert not onchain_df.empty
