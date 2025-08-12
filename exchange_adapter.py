import os
import time
import hmac
import hashlib
from urllib.parse import urlencode
import requests


class ExchangeAdapter:
    """Abstract base class for exchange adapters."""

    def place_market_order(self, symbol, side, *, quantity=None, quote_quantity=None):
        """Place a market order.

        Parameters
        ----------
        symbol: str
            Trading pair symbol, e.g. ``"BTCUSDT"``.
        side: str
            "BUY" or "SELL".
        quantity: float, optional
            Base asset quantity to trade.
        quote_quantity: float, optional
            Quote asset amount to spend. Either ``quantity`` or
            ``quote_quantity`` must be provided.
        """
        raise NotImplementedError

    def get_order(self, symbol, order_id):
        """Fetch order status."""
        raise NotImplementedError


class BinancePaperTradeAdapter(ExchangeAdapter):
    """Adapter for Binance testnet (paper trading)."""

    def __init__(self, api_key=None, api_secret=None, base_url=None):
        self.base_url = base_url or os.getenv("BINANCE_TESTNET_URL", "https://testnet.binance.vision")
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        if not self.api_key or not self.api_secret:
            # Defer failure until first request so tests can run without keys
            self._keys_missing = True
        else:
            self._keys_missing = False

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------
    def _sign(self, params):
        query = urlencode(params)
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method, path, params):
        if self._keys_missing:
            raise RuntimeError("Binance API keys not provided for paper trading")
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params = self._sign(params)
        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------
    def place_market_order(self, symbol, side, *, quantity=None, quote_quantity=None):
        params = {"symbol": symbol, "side": side, "type": "MARKET"}
        if quote_quantity is not None:
            params["quoteOrderQty"] = quote_quantity
        elif quantity is not None:
            params["quantity"] = quantity
        else:
            raise ValueError("quantity or quote_quantity required")

        data = self._request("POST", "/api/v3/order", params)
        fills = data.get("fills", [])
        if fills:
            total_qty = sum(float(f["qty"]) for f in fills)
            total_quote = sum(float(f["price"]) * float(f["qty"]) for f in fills)
            avg_price = total_quote / total_qty if total_qty else 0.0
        else:
            total_qty = float(data.get("executedQty", 0))
            total_quote = float(data.get("cummulativeQuoteQty", 0))
            avg_price = total_quote / total_qty if total_qty else 0.0
        return {"order_id": data.get("orderId"), "executed_qty": total_qty, "price": avg_price}

    def get_order(self, symbol, order_id):
        params = {"symbol": symbol, "orderId": order_id}
        return self._request("GET", "/api/v3/order", params)
