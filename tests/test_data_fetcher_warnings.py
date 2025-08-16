import os
import warnings

import data_fetcher
from data_fetcher import fetch_fear_greed_index, fetch_onchain_metrics


def test_no_future_warning_on_timestamp_parsing(monkeypatch, tmp_path):
    fg_sample = {
        "data": [
            {"timestamp": "1700000000", "value": "60"},
            {"timestamp": "1700003600", "value": "55"},
        ]
    }
    tx_sample = {
        "data": [["1700000000", 12345], ["1700003600", 23456]]
    }
    active_sample = {
        "data": [["1700000000", 1000], ["1700003600", 1200]]
    }

    def mock_safe_request(url, params=None, **kwargs):
        if "fng" in url:
            return fg_sample
        if "transactions-per-day" in url:
            return tx_sample
        if "active-addresses" in url:
            return active_sample
        return None

    monkeypatch.setattr("data_fetcher.safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        fg_df = fetch_fear_greed_index()
        onchain_df = fetch_onchain_metrics()

    assert not fg_df.empty
    assert not onchain_df.empty
