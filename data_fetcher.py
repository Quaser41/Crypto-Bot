# data_fetcher.py
import os
import time
import pickle
import requests
import pandas as pd
import yfinance as yf
import asyncio
from datetime import datetime, timedelta
from rate_limiter import wait_for_slot
from symbol_resolver import (
    resolve_symbol_binance_us,
    resolve_symbol_binance_global,
    resolve_symbol_coinbase,
)
from extract_gainers import extract_gainers
from utils.logging import get_logger

logger = get_logger(__name__)

# Status codes for which we retry by default (server errors)
SERVER_ERROR_CODES = set(range(500, 600))
SEEN_404_URLS = set()


# =========================================================
# ✅ SENTIMENT & ON-CHAIN METRICS
# =========================================================
def fetch_fear_greed_index(limit=30):
    """Fetch the crypto Fear & Greed index."""
    url = "https://api.alternative.me/fng/"
    params = {"limit": limit, "format": "json"}
    data = safe_request(url, params=params, backoff_on_429=False)
    if not data or "data" not in data:
        return pd.DataFrame()

    try:
        df = pd.DataFrame(data["data"])
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.rename(columns={"timestamp": "Timestamp", "value": "FearGreed"}, inplace=True)
        df["FearGreed"] = pd.to_numeric(df["FearGreed"], errors="coerce")
        return df[["Timestamp", "FearGreed"]].sort_values("Timestamp")
    except Exception as e:
        logger.error(f"❌ Failed to parse Fear & Greed index: {e}")
        return pd.DataFrame()


def fetch_onchain_metrics(days=14):
    """Fetch basic on-chain metrics like transaction volume and active addresses."""
    # ``blockchain.info`` exposes anonymous chart endpoints for a handful of
    # on-chain statistics.  Using this domain keeps the API working reliably.
    # The default ``days`` parameter has been reduced so that requests stay
    # within the range the service supports and avoid ``404`` responses.
    # Some environments have reported ``404`` errors when using the
    # ``api.blockchain.info`` subdomain for chart data.  The same chart
    # endpoints are available on the main ``blockchain.info`` domain, which
    # resolves the issue and continues to serve anonymous JSON data.
    base = "https://blockchain.info/charts"
    tx_url = f"{base}/transaction-volume?timespan={days}days&format=json"
    active_url = f"{base}/activeaddresses?timespan={days}days&format=json"

    # Only retry on server errors for these endpoints
    tx_data = safe_request(tx_url, retry_statuses=SERVER_ERROR_CODES, backoff_on_429=False)
    active_data = safe_request(active_url, retry_statuses=SERVER_ERROR_CODES, backoff_on_429=False)

    frames = []
    if tx_data and "values" in tx_data:
        df_tx = pd.DataFrame(tx_data["values"])
        df_tx["x"] = pd.to_numeric(df_tx["x"], errors="coerce")
        df_tx["x"] = pd.to_datetime(df_tx["x"], unit="s")
        df_tx.rename(columns={"x": "Timestamp", "y": "TxVolume"}, inplace=True)
        frames.append(df_tx)

    if active_data and "values" in active_data:
        df_active = pd.DataFrame(active_data["values"])
        df_active["x"] = pd.to_numeric(df_active["x"], errors="coerce")
        df_active["x"] = pd.to_datetime(df_active["x"], unit="s")
        df_active.rename(columns={"x": "Timestamp", "y": "ActiveAddresses"}, inplace=True)
        frames.append(df_active)

    if not frames:
        return pd.DataFrame()

    df = frames[0]
    for f in frames[1:]:
        df = pd.merge(df, f, on="Timestamp", how="outer")

    return df.sort_values("Timestamp")


# =========================================================
# ✅ GLOBAL CACHING SYSTEM
# =========================================================
COIN_ID_CACHE = {}
COINGECKO_LIST = {}

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
                logger.info(f"🕒 Cache expired for {key}")
    except Exception as e:
        logger.warning(f"⚠️ Cache read error for {key}: {e}")
    return None

def update_cache(key, data):
    """Persist cache to disk."""
    path = _cache_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump((time.time(), data), f)
    except Exception as e:
        logger.error(f"❌ Cache write error for {key}: {e}")

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
                    logger.warning(f"⚠️ Skipping in-use file: {filename}")
    if removed:
        logger.info(f"🧹 Cleared {removed} old cache file(s)")        

# =========================================================
# ✅ GENERIC SAFE REQUEST WRAPPER (w/ retry + rate limit)
# =========================================================
def safe_request(
    url,
    params=None,
    timeout=10,
    max_retries=3,
    retry_delay=5,
    retry_statuses=None,
    backoff_on_429=True,
):
    """Safe API request with configurable retry logic.

    Retries are attempted only for status codes listed in ``retry_statuses``.
    By default, this includes HTTP 429 and all 5xx server errors. Any 4xx
    response other than 429 will immediately return ``None``.
    """

    if retry_statuses is None:
        retry_statuses = SERVER_ERROR_CODES | {429}
    else:
        retry_statuses = set(retry_statuses)

    for attempt in range(1, max_retries + 1):
        wait_for_slot(url)  # throttle

        try:
            r = requests.get(url, params=params, timeout=timeout)


            if r.status_code == 404:
                if url not in SEEN_404_URLS:
                    logger.error(f"❌ 404 Not Found {url}")
                    SEEN_404_URLS.add(url)
                return None
      
            # Special handling for rate limiting
            if r.status_code == 429:
                if 429 in retry_statuses:
                    logger.warning(
                        f"⚠️ Rate limited {url} (attempt {attempt})"
                        + (" → backoff..." if backoff_on_429 else "")
                    )
                    if backoff_on_429:
                        wait_for_slot(url, backoff=True)
                        time.sleep(2 ** attempt)
                    else:
                        time.sleep(retry_delay)
                    continue
                logger.error(f"❌ Failed (429) {url} attempt {attempt}")
                return None


            if r.status_code in retry_statuses:
                logger.error(f"❌ Failed ({r.status_code}) {url} attempt {attempt}")
                time.sleep(retry_delay)
                continue

            if 400 <= r.status_code < 600:
                logger.error(f"❌ Failed ({r.status_code}) {url} (no retry)")
                return None

            return r.json()

        except Exception as e:
            logger.error(f"❌ Exception on attempt {attempt} for {url}: {e}")
            time.sleep(retry_delay)

    logger.error(f"❌ All attempts failed for {url}")
    return None


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

    # Try preloaded Coingecko list before making search calls
    if not COINGECKO_LIST:
        cache = cached_fetch("coingecko_full_list", ttl=86400)
        if cache:
            COINGECKO_LIST.update(cache)
        else:
            try:
                data = safe_request("https://api.coingecko.com/api/v3/coins/list", backoff_on_429=False)
                if data:
                    COINGECKO_LIST.update({item["symbol"].upper(): item["id"] for item in data})
                    update_cache("coingecko_full_list", COINGECKO_LIST)
            except Exception as e:
                logger.error(f"❌ Failed to preload Coingecko list: {e}")

    if symbol.upper() in COINGECKO_LIST:
        coin_id = COINGECKO_LIST[symbol.upper()]
        update_cache(cache_key, coin_id)
        COIN_ID_CACHE[symbol] = coin_id
        return coin_id

    # Fallback: call Coingecko search only if list lookup fails
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
        logger.error(f"❌ Coin ID resolution failed for {symbol}: {e}")

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

        # ✅ Check availability on Coinbase
        if not is_symbol_on_coinbase(symbol):
            continue

        # 🔍 Resolve coin ID only if symbol is valid
        coin_id = resolve_coin_id(symbol, name)
        if not coin_id:
            continue

        result.append((coin_id, symbol, name, change_pct))

        if len(result) >= limit:
            break

    return result


def is_symbol_on_coinbase(symbol):
    try:
        return resolve_symbol_coinbase(symbol) is not None
    except Exception:
        return False

# =========================================================
# ✅ BATCH PRICES
# =========================================================

# =========================================================
# ✅ MULTI-SOURCE OHLCV FETCHING
# =========================================================
DATA_SOURCES = ["coinbase", "coingecko", "dexscreener"]

def fetch_ohlcv_smart(symbol, **kwargs):
    """Smart OHLCV fetcher that tries all sources in order per token."""
    for source in DATA_SOURCES:
        try:
            if source == "coinbase":
                logger.info(f"⚡ Trying Coinbase for {symbol}")
                df = fetch_coinbase_ohlcv(symbol, **kwargs)
                if not df.empty:
                    return df

            elif source == "yfinance":
                logger.info(f"⚡ Trying YFinance for {symbol}")
                df = fetch_from_yfinance(symbol, days=kwargs.get("days", 1))
                if not df.empty:
                    return df

            elif source == "dexscreener":
                logger.info(f"⚡ Trying DexScreener for {symbol}")
                df = fetch_dexscreener_ohlcv(symbol)
                if not df.empty:
                    return df

            elif source == "coingecko":
                logger.info(f"⚡ Trying Coingecko for {symbol} (days={kwargs.get('days', 1)})")
                df = fetch_coingecko_ohlcv(kwargs.get("coin_id", symbol), days=kwargs.get("days", 1))
                if not df.empty:
                    return df

        except Exception as e:
            logger.warning(f"⚠️ {source} failed for {symbol}: {type(e).__name__} - {e}")
            continue

    logger.error(f"❌ All sources failed for {symbol}")
    return pd.DataFrame()


async def fetch_ohlcv_smart_async(symbol, **kwargs):
    """Asynchronous wrapper around :func:`fetch_ohlcv_smart`.

    This implementation avoids spawning background threads, which simplifies
    testing and eliminates issues when ``threading.Thread`` is monkeypatched.
    """
    return fetch_ohlcv_smart(symbol, **kwargs)



def fetch_binance_ohlcv(symbol, interval="15m", limit=96, **kwargs):
    binance_symbol = resolve_symbol_binance_global(symbol)
    if not binance_symbol:
        logger.warning(f"⚠️ Symbol {symbol.upper()} not found on Binance Global.")
        return pd.DataFrame()

    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}

    for attempt in range(3):
        wait_for_slot(url)
        r = requests.get(url, params=params, timeout=10)

        if r.status_code == 451:
            logger.warning(f"⚠️ Binance 451 for {symbol} — skipping.")
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
            logger.error(f"❌ Binance fetch fail ({symbol}): {e}")
            time.sleep(2)

    logger.error(f"❌ All Binance fetch attempts failed for {symbol}")
    return pd.DataFrame()


def fetch_binance_us_ohlcv(symbol, interval="15m", limit=96, **kwargs):
    binance_symbol = resolve_symbol_binance_us(symbol)
    if not binance_symbol:
        logger.warning(f"⚠️ Symbol {symbol.upper()} not found on Binance.US.")
        return pd.DataFrame()

    url = "https://api.binance.us/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}

    wait_for_slot(url)
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
        logger.error(f"❌ Binance.US fetch fail ({symbol}): {e}")
        return pd.DataFrame()


def fetch_coinbase_ohlcv(symbol, interval="15m", days=1, limit=300, **kwargs):
    """Fetch OHLCV data from Coinbase.

    Coinbase's candles endpoint returns at most 300 data points per request.
    This helper splits the desired date range (``days`` back from now) into
    chunks so that the entire span is covered.
    """

    product_id = resolve_symbol_coinbase(symbol)
    if not product_id:
        logger.warning(f"⚠️ Symbol {symbol.upper()} not found on Coinbase.")
        return pd.DataFrame()

    gran_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
    granularity = gran_map.get(interval, 900)

    end = datetime.utcnow()
    start = end - timedelta(days=days)
    max_span = timedelta(seconds=granularity * limit)
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"

    frames = []
    window_start = start
    chunk = 1
    start_time = time.time()
    while window_start < end:
        window_end = min(window_start + max_span, end)
        logger.info(
            "Fetching Coinbase chunk %s: %s → %s",
            chunk,
            window_start.isoformat(),
            window_end.isoformat(),
        )
        params = {
            "granularity": granularity,
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        }

        data = safe_request(url, params=params, backoff_on_429=False)
        if not data:
            df = pd.DataFrame()
            df.attrs["fetch_seconds"] = time.time() - start_time
            return df

        frames.append(pd.DataFrame(data, columns=["Timestamp", "Low", "High", "Open", "Close", "Volume"]))
        window_start = window_end
        chunk += 1

    elapsed = time.time() - start_time

    if not frames:
        df = pd.DataFrame()
        df.attrs["fetch_seconds"] = elapsed
        return df

    try:
        df = pd.concat(frames, ignore_index=True)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
        df = df.sort_values("Timestamp")
        df = df[["Timestamp", "Close"]]
        df.attrs["fetch_seconds"] = elapsed
        return df
    except Exception as e:
        logger.error(f"❌ Coinbase fetch fail ({symbol}): {e}")
        df = pd.DataFrame()
        df.attrs["fetch_seconds"] = elapsed
        return df


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
            logger.info(f"🔎 Trying yfinance ticker: {yf_symbol}")
            df = yf.download(yf_symbol, period=f"{days}d", interval="1h", progress=False)

            if not df.empty and "Close" in df.columns and df["Close"].dropna().shape[0] > 0:
                df = df.rename(columns={
                    "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"
                })[["Open", "High", "Low", "Close", "Volume"]]
                df.reset_index(drop=True, inplace=True)
                logger.info(f"✅ YFinance data loaded for {yf_symbol}")
                return df

        except Exception as e:
            logger.warning(f"⚠️ Failed for {yf_symbol}: {e}")

    logger.error(f"❌ YFinance could not fetch data for {symbol}")
    return pd.DataFrame()


def fetch_coingecko_ohlcv(coin_id, days=1):
    cache_key = f"cg_ohlcv:{coin_id}:{days}"
    cached = cached_fetch(cache_key, ttl=180)  # 3 min TTL
    if cached is not None:
        logger.info(f"📦 Using cached Coingecko OHLCV for {coin_id}")
        return cached

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}

    data = safe_request(url, params=params, max_retries=3, retry_delay=5)
    if data and "prices" in data:
        df = pd.DataFrame(data["prices"], columns=["Timestamp", "Close"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
        update_cache(cache_key, df)
        logger.info(f"✅ Coingecko OHLCV loaded for {coin_id}")
        return df

    logger.error(f"❌ Coingecko OHLCV fetch failed for {coin_id}")
    return pd.DataFrame()


# =========================================================
# ✅ LIVE PRICE (multi-fallback)
# =========================================================
def fetch_live_price(symbol, coin_id=None):
    cache_key = f"live_price:{symbol}"

    # ✅ Check disk cache first
    cached = cached_fetch(cache_key, ttl=60)
    if cached is not None:
        logger.info(f"📦 Using cached disk price for {symbol}: {cached}")
        return cached

    symbol_upper = symbol.upper()
    coinbase_symbol = resolve_symbol_coinbase(symbol)

    sources = ["cryptocompare", "coinbase", "dexscreener", "coingecko"]

    for source in sources:
        try:
            if source == "coinbase":
                if not coinbase_symbol:
                    logger.warning(f"⚠️ Skipping Coinbase: {symbol} not found")
                    continue
                logger.info(f"⚡ Trying Coinbase for {symbol} → {coinbase_symbol}")
                url = f"https://api.exchange.coinbase.com/products/{coinbase_symbol}/ticker"
                data = safe_request(url, backoff_on_429=False)
                if data and "price" in data:
                    price = float(data["price"])
                    if price > 0:
                        update_cache(cache_key, price)
                        return price

            elif source == "dexscreener":
                logger.info(f"⚡ Trying DexScreener for {symbol}")
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

            elif source == "cryptocompare":
                logger.info(f"⚡ Trying CryptoCompare for {symbol_upper}")
                url = "https://min-api.cryptocompare.com/data/price"
                params = {"fsym": symbol_upper, "tsyms": "USD"}
                data = safe_request(url, params=params, backoff_on_429=False)
                if data and "USD" in data:
                    price = float(data["USD"])
                    if price > 0:
                        update_cache(cache_key, price)
                        return price
                                     
            elif source == "coingecko":
                if not coin_id:
                    logger.warning(f"⚠️ Skipping CoinGecko: no coin_id for {symbol}")
                    continue
                logger.info(f"⚡ Trying CoinGecko for {coin_id}")

                url = f"https://api.coingecko.com/api/v3/simple/price"
                params = {"ids": coin_id, "vs_currencies": "usd"}
                data = safe_request(url, params=params)

                if data and coin_id in data and "usd" in data[coin_id]:
                    price = float(data[coin_id]["usd"])
                    if price > 0:
                        update_cache(cache_key, price)
                        return price

        except Exception as e:
            logger.warning(f"⚠️ {source} failed for {symbol}: {type(e).__name__} - {e}")

    logger.error(f"❌ No live price found for {symbol}")
    return None


