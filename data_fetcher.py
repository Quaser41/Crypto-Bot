# data_fetcher.py
import os
import time
import pickle
import requests
import pandas as pd
import websocket
import threading
import json
import yfinance as yf
from datetime import datetime, timedelta
from rate_limiter import wait_for_slot
from symbol_resolver import resolve_symbol_binance_us, resolve_symbol_binance_global
from extract_gainers import extract_gainers


# =========================================================
# ✅ GLOBAL CACHING SYSTEM
# =========================================================
COIN_ID_CACHE = {}

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(key):
    return os.path.join(CACHE_DIR, f"{key.replace(':', '_')}.pkl")

def cached_fetch(key, ttl=180):
    """Return cached data from disk if fresh; else None."""
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            timestamp, data = pickle.load(f)
            if time.time() - timestamp < ttl:
                return data
            else:
                print(f"🕒 Cache expired for {key}")
    except Exception as e:
        print(f"⚠️ Cache read error for {key}: {e}")
    return None

def update_cache(key, data):
    """Persist cache to disk."""
    path = _cache_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump((time.time(), data), f)
    except Exception as e:
        print(f"❌ Cache write error for {key}: {e}")

def clear_old_cache(cache_dir="cache", max_age=600):
    """
    Deletes files in the cache directory older than max_age seconds (default 10 mins).
    """
    now = time.time()
    if not os.path.exists(cache_dir):
        return

    removed = 0
    for filename in os.listdir(cache_dir):
        path = os.path.join(cache_dir, filename)
        if os.path.isfile(path):
            age = now - os.path.getmtime(path)
            if age > max_age:
                try:
                    os.remove(path)
                    removed += 1
                except PermissionError:
                    print(f"⚠️ Skipping in-use file: {filename}")
    if removed:
        print(f"🧹 Cleared {removed} old cache file(s)")        

# =========================================================
# ✅ GENERIC SAFE REQUEST WRAPPER (w/ retry + rate limit)
# =========================================================
def safe_request(url, params=None, timeout=10, max_retries=3, retry_delay=5, backoff_on_429=True):
    """Safe API request with retries, rate-limit handling, and backoff."""
    for attempt in range(1, max_retries + 1):
        wait_for_slot()  # throttle

        try:
            r = requests.get(url, params=params, timeout=timeout)

            if r.status_code == 429 and backoff_on_429:
                print(f"⚠️ Rate limited {url} (attempt {attempt}) → backoff...")
                wait_for_slot(backoff=True)
                time.sleep(2 ** attempt)
                continue

            if r.status_code != 200:
                print(f"❌ Failed ({r.status_code}) {url} attempt {attempt}")
                time.sleep(retry_delay)
                continue

            return r.json()

        except Exception as e:
            print(f"❌ Exception on attempt {attempt} for {url}: {e}")
            time.sleep(retry_delay)

    print(f"❌ All attempts failed for {url}")
    return None

# =========================================================
# ✅ Coinbase Configuration
# =========================================================
LIVE_PRICES = {}
_ws_app = None
_tracked_symbols = set()
_lock = threading.Lock()

def _on_message(ws, message):
    try:
        msg = json.loads(message)
        if msg.get("channel") == "ticker" and "events" in msg:
            for event in msg["events"]:
                # Unified handler for both snapshot and update
                tickers = event.get("tickers", [])
                for ticker in tickers:
                    product_id = ticker.get("product_id", "")
                    price = ticker.get("price")

                    if product_id and price:
                        symbol = product_id.replace("-USD", "").upper()
                        LIVE_PRICES[symbol.upper()] = float(price)
                        #print(f"📡 WebSocket price update: {symbol} → {price}")

    except Exception as e:
        print(f"⚠️ Error parsing WebSocket message: {e}")


def _on_open(ws):
    with _lock:
        if _tracked_symbols:
            sub = {
                "type": "subscribe",
                "channel": "ticker",
                "product_ids": [f"{sym}-USD" for sym in _tracked_symbols]
            }

            print(f"📡 Subscribing to: {sub['product_ids']}")
            ws.send(json.dumps(sub))


def _run_ws():
    global _ws_app
    while True:
        try:
            _ws_app = websocket.WebSocketApp(
                "wss://advanced-trade-ws.coinbase.com",
                on_open=_on_open,
                on_message=_on_message,
                on_close=lambda ws, code, msg: print(f"❌ WebSocket closed: {code} - {msg}"),
                on_error=lambda ws, err: print(f"❌ WebSocket error: {err}")
            )
            _ws_app.run_forever()
        except Exception as e:
            print(f"❌ WebSocket crashed: {e}")
        
        print("🔁 Reconnecting to Coinbase WebSocket in 5 seconds...")
        time.sleep(5)



def start_coinbase_ws():
    t = threading.Thread(target=_run_ws, daemon=True)
    t.start()

def track_symbol(symbol):
    global _ws_app
    symbol = symbol.upper()
    product_id = f"{symbol}-USD"
    
    if product_id not in get_valid_coinbase_products():
        print(f"❌ {product_id} is not supported on Coinbase, skipping.")
        return

    with _lock:
        if symbol not in _tracked_symbols:
            _tracked_symbols.add(symbol)
            
            if _ws_app and _ws_app.sock and _ws_app.sock.connected:
                msg = {
                    "type": "subscribe",
                    "channel": "ticker",
                    "product_ids": [product_id]
                }
                _ws_app.send(json.dumps(msg))
                print(f"🛰️ Subscribed to Coinbase WebSocket for {symbol}")
            else:
                print(f"⚠️ WebSocket not connected when trying to subscribe to {symbol}")


def untrack_symbol(symbol):
    """
    Dynamically unsubscribe from a symbol if it's no longer needed.
    """
    global _ws_app
    symbol = symbol.upper()
    product_id = f"{symbol}-USD"

    with _lock:
        if symbol in _tracked_symbols:
            _tracked_symbols.remove(symbol)
            if _ws_app and _ws_app.sock and _ws_app.sock.connected:
                msg = {
                    "type": "unsubscribe",
                    "channels": [{
                        "name": "ticker",
                        "product_ids": [product_id]
                    }]
                }
                _ws_app.send(json.dumps(msg))
                print(f"❌ Unsubscribed from Coinbase WebSocket for {symbol}")

_VALID_COINBASE_PRODUCTS = set()

def get_valid_coinbase_products():
    global _VALID_COINBASE_PRODUCTS
    if _VALID_COINBASE_PRODUCTS:
        return _VALID_COINBASE_PRODUCTS

    url = "https://api.exchange.coinbase.com/products"
    data = safe_request(url)
    if not data:
        return set()

    _VALID_COINBASE_PRODUCTS = set(p["id"] for p in data)  # 'id' like "BTC-USD"
    return _VALID_COINBASE_PRODUCTS


# =========================================================
# ✅ TOP GAINERS + TRENDING MERGE
# =========================================================
def resolve_coin_id(symbol, name):
    cache_key = f"resolve_id:{symbol}"
    
    # Check in-session memory first
    if symbol in COIN_ID_CACHE:
        return COIN_ID_CACHE[symbol]

    # Then check disk cache
    cached = cached_fetch(cache_key, ttl=86400)  # 1-day TTL
    if cached:
        COIN_ID_CACHE[symbol] = cached
        return cached

    # Fallback: call Coingecko (only if absolutely needed)
    url = "https://api.coingecko.com/api/v3/search"
    params = {"query": name}

    try:
        result = safe_request(url, params=params)
        if result and "coins" in result:
            for coin in result["coins"]:
                if coin["symbol"].upper() == symbol.upper():
                    coin_id = coin["id"]
                    update_cache(cache_key, coin_id)
                    COIN_ID_CACHE[symbol] = coin_id
                    return coin_id
    except Exception as e:
        print(f"❌ Coin ID resolution failed for {symbol}: {e}")

    return None

def get_top_gainers(limit=10):
    raw = extract_gainers()
    result = []

    for g in raw:
        name = g["name"]
        symbol = g["symbol"].upper()
        try:
            change_pct = float(g["change"].strip('%'))
        except ValueError:
            continue

        # ✅ Check Binance.US directly first
        if not is_symbol_on_binance_us(symbol):
            continue

        # 🔍 Resolve coin ID only if symbol is valid
        coin_id = resolve_coin_id(symbol, name)
        if not coin_id:
            continue

        result.append((coin_id, symbol, name, change_pct))

        if len(result) >= limit:
            break

    return result


def is_symbol_on_binance_us(symbol):
    try:
        url = f"https://api.binance.us/api/v3/exchangeInfo"
        data = cached_fetch("binance_us_exchange_info", ttl=3600)
        if not data:
            data = safe_request(url)
            update_cache("binance_us_exchange_info", data)

        symbols = {s["symbol"] for s in data["symbols"]}
        # Convert coin symbol like "HBAR" to valid pairs like "HBARUSDT"
        return any(symbol + quote in symbols for quote in ["USDT", "USD", "BUSD"])
    except:
        return False

# =========================================================
# ✅ BATCH PRICES
# =========================================================

# =========================================================
# ✅ MULTI-SOURCE OHLCV FETCHING
# =========================================================
DATA_SOURCES = ["binance_us", "binance", "coingecko", "dexscreener"]

def fetch_ohlcv_smart(symbol, **kwargs):
    """Smart OHLCV fetcher that tries all sources in order per token."""
    for source in DATA_SOURCES:
        try:
            if source == "binance":
                print(f"⚡ Trying Binance for {symbol}")
                df = fetch_binance_ohlcv(symbol, **kwargs)
                if not df.empty:
                    return df

            elif source == "binance_us":
                print(f"⚡ Trying Binance.US for {symbol}")
                df = fetch_binance_us_ohlcv(symbol, **kwargs)
                if not df.empty:
                    return df

            elif source == "yfinance":
                print(f"⚡ Trying YFinance for {symbol}")
                df = fetch_from_yfinance(symbol, days=kwargs.get("days", 1))
                if not df.empty:
                    return df

            elif source == "dexscreener":
                print(f"⚡ Trying DexScreener for {symbol}")
                df = fetch_dexscreener_ohlcv(symbol)
                if not df.empty:
                    return df

            elif source == "coingecko":
                print(f"⚡ Trying Coingecko for {symbol} (days={kwargs.get('days', 1)})")
                df = fetch_coingecko_ohlcv(kwargs.get("coin_id", symbol), days=kwargs.get("days", 1))
                if not df.empty:
                    return df

        except Exception as e:
            print(f"⚠️ {source} failed for {symbol}: {type(e).__name__} - {e}")
            continue

    print(f"❌ All sources failed for {symbol}")
    return pd.DataFrame()



def fetch_binance_ohlcv(symbol, interval="15m", limit=96, **kwargs):
    binance_symbol = resolve_symbol_binance_global(symbol)
    if not binance_symbol:
        print(f"⚠️ Symbol {symbol.upper()} not found on Binance Global.")
        return pd.DataFrame()

    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}

    for attempt in range(3):
        wait_for_slot()
        r = requests.get(url, params=params, timeout=10)

        if r.status_code == 451:
            print(f"⚠️ Binance 451 for {symbol} — skipping.")
            return pd.DataFrame()

        try:
            r.raise_for_status()
            data = r.json()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "x", "y", "z", "a", "b", "c"
            ])
            df["Close"] = df["close"].astype(float)
            df["Timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df[["Timestamp", "Close"]]
        except Exception as e:
            print(f"❌ Binance fetch fail ({symbol}): {e}")
            time.sleep(2)

    print(f"❌ All Binance fetch attempts failed for {symbol}")
    return pd.DataFrame()


def fetch_binance_us_ohlcv(symbol, interval="15m", limit=96, **kwargs):
    binance_symbol = resolve_symbol_binance_us(symbol)
    if not binance_symbol:
        print(f"⚠️ Symbol {symbol.upper()} not found on Binance.US.")
        return pd.DataFrame()

    url = "https://api.binance.us/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}

    wait_for_slot()
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "x", "y", "z", "a", "b", "c"
        ])
        df["Close"] = df["close"].astype(float)
        df["Timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df[["Timestamp", "Close"]]
    except Exception as e:
        print(f"❌ Binance.US fetch fail ({symbol}): {e}")
        return pd.DataFrame()


def fetch_dexscreener_ohlcv(symbol):
    url = f"https://api.dexscreener.com/latest/dex/search?q={symbol.lower()}"
    data = safe_request(url)
    if not data or "pairs" not in data:
        return pd.DataFrame()
    price = float(data["pairs"][0]["priceUsd"])
    ts = pd.Timestamp.utcnow()
    return pd.DataFrame({"Timestamp": [ts], "Close": [price]})


#def fetch_from_yfinance(symbol, days=10):
    yf_candidates = [
        f"{symbol.upper()}-USD",
        f"{symbol.upper()}-CRYPTO",
        f"{symbol.upper()}"
    ]

    for yf_symbol in yf_candidates:
        try:
            print(f"🔎 Trying yfinance ticker: {yf_symbol}")
            df = yf.download(yf_symbol, period=f"{days}d", interval="1h", progress=False)

            if not df.empty and "Close" in df.columns and df["Close"].dropna().shape[0] > 0:
                df = df.rename(columns={
                    "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"
                })[["Open", "High", "Low", "Close", "Volume"]]
                df.reset_index(drop=True, inplace=True)
                print(f"✅ YFinance data loaded for {yf_symbol}")
                return df

        except Exception as e:
            print(f"⚠️ Failed for {yf_symbol}: {e}")

    print(f"❌ YFinance could not fetch data for {symbol}")
    return pd.DataFrame()


def fetch_coingecko_ohlcv(coin_id, days=1):
    cache_key = f"cg_ohlcv:{coin_id}:{days}"
    cached = cached_fetch(cache_key, ttl=180)  # 3 min TTL
    if cached is not None:
        print(f"📦 Using cached Coingecko OHLCV for {coin_id}")
        return cached

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}

    data = safe_request(url, params=params, max_retries=3, retry_delay=5)
    if data and "prices" in data:
        df = pd.DataFrame(data["prices"], columns=["Timestamp", "Close"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
        update_cache(cache_key, df)
        print(f"✅ Coingecko OHLCV loaded for {coin_id}")
        return df

    print(f"❌ Coingecko OHLCV fetch failed for {coin_id}")
    return pd.DataFrame()


# =========================================================
# ✅ LIVE PRICE (multi-fallback)
# =========================================================
def fetch_live_price(symbol, coin_id=None):
    cache_key = f"live_price:{symbol}"
    
    # ✅ 1st: Check WebSocket
    ws_price = LIVE_PRICES.get(symbol.upper())
    if ws_price:
        print(f"✅ Using live WebSocket price for {symbol}: {ws_price}")
        update_cache(cache_key, ws_price)
        return ws_price

    # ✅ 2nd: Check disk cache only if no WebSocket price
    cached = cached_fetch(cache_key, ttl=60)
    if cached is not None:
        print(f"📦 Using cached disk price for {symbol}: {cached}")
        return cached

    symbol_upper = symbol.upper()
    resolved_global = resolve_symbol_binance_global(symbol)
    resolved_us = resolve_symbol_binance_us(symbol)

    sources = ["coinbase", "binance", "binance_us", "dexscreener", "coingecko"]

    for source in sources:
        try:
            if source == "binance":
                if not resolved_global:
                    print(f"⚠️ Skipping Binance Global: {symbol} not found")
                    continue
                print(f"⚡ Trying Binance Global for {symbol} → {resolved_global}")
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={resolved_global}"
                data = safe_request(url, backoff_on_429=False)
                if data and "price" in data:
                    price = float(data["price"])
                    if price > 0:
                        update_cache(cache_key, price)
                        return price

            elif source == "binance_us":
                if not resolved_us:
                    print(f"⚠️ Skipping Binance.US: {symbol} not found")
                    continue
                print(f"⚡ Trying Binance.US for {symbol} → {resolved_us}")
                url = f"https://api.binance.us/api/v3/ticker/price?symbol={resolved_us}"
                data = safe_request(url, backoff_on_429=False)
                if data and "price" in data:
                    price = float(data["price"])
                    if price > 0:
                        update_cache(cache_key, price)
                        return price

            elif source == "dexscreener":
                print(f"⚡ Trying DexScreener for {symbol}")
                search_query = coin_id or symbol
                dex_url = f"https://api.dexscreener.com/latest/dex/search?q={search_query}"
                data = safe_request(dex_url, backoff_on_429=False)
                if data and "pairs" in data:
                    valid_pairs = [
                        (float(p["priceUsd"]), float(p.get("liquidity", {}).get("usd", 0)))
                        for p in data["pairs"] if "priceUsd" in p
                    ]
                    valid_pairs = [p for p in valid_pairs if p[1] > 10000]
                    if valid_pairs:
                        best_price = sorted(valid_pairs, key=lambda x: x[1], reverse=True)[0][0]
                        update_cache(cache_key, best_price)
                        return best_price

            elif source == "coinbase":
                print(f"⚡ Trying Coinbase WebSocket price for {symbol_upper}")  
                track_symbol(symbol_upper)  # <-- Dynamically subscribe
                time.sleep(1.5)  # Let WebSocket deliver at least 1 tick
                
                ws_price = LIVE_PRICES.get(symbol_upper)
                if ws_price:
                    print(f"✅ WebSocket price found for {symbol_upper}: {ws_price}")
                    update_cache(cache_key, ws_price)
                    return ws_price
                                     
            elif source == "coingecko":
                if not coin_id:
                    print(f"⚠️ Skipping CoinGecko: no coin_id for {symbol}")
                    continue
                print(f"⚡ Trying CoinGecko for {coin_id}")

                url = f"https://api.coingecko.com/api/v3/simple/price"
                params = {"ids": coin_id, "vs_currencies": "usd"}
                data = safe_request(url, params=params)

                if data and coin_id in data and "usd" in data[coin_id]:
                    price = float(data[coin_id]["usd"])
                    if price > 0:
                        update_cache(cache_key, price)
                        return price

        except Exception as e:
            print(f"⚠️ {source} failed for {symbol}: {type(e).__name__} - {e}")

    print(f"❌ No live price found for {symbol}")
    return None

