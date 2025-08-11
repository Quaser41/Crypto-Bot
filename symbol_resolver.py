import requests

BINANCE_GLOBAL_SYMBOLS = {}
BINANCE_US_SYMBOLS = {}
COINBASE_SYMBOLS = {}

def load_binance_global_symbols():
    global BINANCE_GLOBAL_SYMBOLS
    if BINANCE_GLOBAL_SYMBOLS:
        return

    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 451:
            print("⚠️ Binance Global blocked (451) – skipping global symbols.")
            return
        r.raise_for_status()
        data = r.json()
        for sym in data["symbols"]:
            if sym["status"] == "TRADING" and sym["quoteAsset"] == "USDT":
                BINANCE_GLOBAL_SYMBOLS[sym["baseAsset"]] = sym["symbol"]
    except Exception as e:
        print(f"❌ Failed to load Binance Global symbols: {type(e).__name__} - {e}")

def load_binance_us_symbols():
    global BINANCE_US_SYMBOLS
    if BINANCE_US_SYMBOLS:
        return

    url = "https://api.binance.us/api/v3/exchangeInfo"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        for sym in data["symbols"]:
            if sym["status"] == "TRADING" and sym["quoteAsset"] == "USDT":
                BINANCE_US_SYMBOLS[sym["baseAsset"]] = sym["symbol"]
    except Exception as e:
        print(f"❌ Failed to load Binance.US symbols: {type(e).__name__} - {e}")


def resolve_symbol_binance_us(base_asset):
    load_binance_us_symbols()
    return BINANCE_US_SYMBOLS.get(base_asset.upper())

def resolve_symbol_binance_global(base_asset):
    load_binance_global_symbols()
    return BINANCE_GLOBAL_SYMBOLS.get(base_asset.upper())


def load_coinbase_symbols():
    """Load Coinbase products and cache USD/USDT pairs."""
    global COINBASE_SYMBOLS
    if COINBASE_SYMBOLS:
        return

    url = "https://api.exchange.coinbase.com/products"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        for prod in data:
            if prod.get("trading_disabled"):
                continue
            base = prod.get("base_currency")
            quote = prod.get("quote_currency")
            if quote in ("USD", "USDT"):
                # Prefer USD pairs if multiple exist
                if base not in COINBASE_SYMBOLS or quote == "USD":
                    COINBASE_SYMBOLS[base] = prod.get("id")
    except Exception as e:
        print(f"❌ Failed to load Coinbase symbols: {type(e).__name__} - {e}")


def resolve_symbol_coinbase(base_asset):
    load_coinbase_symbols()
    return COINBASE_SYMBOLS.get(base_asset.upper())
