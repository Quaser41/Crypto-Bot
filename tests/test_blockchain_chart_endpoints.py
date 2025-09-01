import os
import requests

BASE = os.getenv("BLOCKCHAIN_CHARTS_BASE", "https://api.blockchain.info/charts")
PARAMS = {"timespan": "5days", "format": "json", "cors": "true"}


def _get_chart(slug: str):
    url = f"{BASE}/{slug}"
    resp = requests.get(url, params=PARAMS, timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "values" in data and isinstance(data["values"], list)
    if data["values"]:
        first = data["values"][0]
        assert isinstance(first, dict)
        assert "x" in first and "y" in first
    return data


def test_transactions_chart_endpoint():
    _get_chart("n-transactions")


def test_active_addresses_chart_endpoint():
    _get_chart("activeaddresses")
