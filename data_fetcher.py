# data_fetcher.py
import os
import time
import pickle
import requests
import pandas as pd
import yfinance as yf
import asyncio
import json
import re
from datetime import datetime, timedelta
from rate_limiter import wait_for_slot
from symbol_resolver import (
    resolve_symbol_binance_us,
    resolve_symbol_binance_global,
    resolve_symbol_coinbase,
)
from extract_gainers import extract_gainers
from utils.logging import get_logger
from config import ENABLE_BINANCE_GLOBAL
from filelock import FileLock

logger = get_logger(__name__)

# Status codes for which we retry by default (server errors)
SERVER_ERROR_CODES = set(range(500, 600))
SEEN_404_URLS = set()
SEEN_NON_JSON_URLS = set()


# =========================================================
# ✅ SENTIMENT & ON-CHAIN METRICS
# =========================================================
def fetch_fear_greed_index(limit=30):
    """Fetch the crypto Fear & Greed index."""
    cache_key = f"fg_index:{limit}"
    cached = cached_fetch(cache_key, ttl=3600)
    if cached is not None:
        logger.info(f"📄 Using cached Fear & Greed index for limit {limit}")
        return cached

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
        df = df[["Timestamp", "FearGreed"]].sort_values("Timestamp")
        update_cache(cache_key, df)
        return df
    except Exception as e:
        logger.error(f"❌ Failed to parse Fear & Greed index: {e}")
        return pd.DataFrame()


def fetch_onchain_metrics(days=14):
    """Fetch basic on-chain metrics like transaction volume and active addresses."""

    cache_key = f"onchain:{days}"
    placeholder_key = f"{cache_key}:placeholder"
    placeholder = cached_fetch(placeholder_key, ttl=60)
    if placeholder is not None:
        logger.info("📄 Using cached fallback on-chain metrics")
        return placeholder

    cached = cached_fetch(cache_key, ttl=3600)
    if cached is not None:
        logger.info(f"📄 Using cached on-chain metrics for {days} days")
        return cached

    logger.info(f"🌐 Fetching on-chain metrics for {days} days")

    def _default_df(col: str) -> pd.DataFrame:
        end = pd.Timestamp.utcnow().normalize().tz_localize(None)
        dates = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
        return pd.DataFrame({"Timestamp": dates, col: [0] * len(dates)})

    api_key = os.getenv("BLOCKCHAIN_API_KEY")
    if not api_key:
        logger.info(
            "🔐 BLOCKCHAIN_API_KEY missing – returning placeholder on-chain metrics"
        )
        df = pd.merge(
            _default_df("TxVolume"), _default_df("ActiveAddresses"), on="Timestamp"
        )
        update_cache(placeholder_key, df)
        return df

    if not os.getenv("GLASSNODE_API_KEY"):
        logger.info("🔓 GLASSNODE_API_KEY missing – using public data sources")

    # Blockchain.com exposes public chart endpoints for several Bitcoin
    # on-chain statistics. Each response has a ``values`` array of
    # ``{"x": timestamp, "y": value}`` entries. ``days`` controls how much
    # history to request. As of today the charts API is served from the legacy
    # ``api.blockchain.info`` domain. If a new path is introduced, callers may
    # override the base URL via ``BLOCKCHAIN_CHARTS_BASE``. Requests always pass
    # ``format=json`` to ensure a JSON payload is returned.
    base_url = os.getenv(
        "BLOCKCHAIN_CHARTS_BASE", "https://api.blockchain.info/charts"
    )
    tx_url = f"{base_url}/n-transactions"
    active_url = f"{base_url}/activeaddresses"
    params = {
        "timespan": f"{days}days",
        "format": "json",
        "cors": "true",
        "api_code": api_key,
    }

    # Only retry on server errors for these endpoints
    headers = {"Accept": "application/json", "User-Agent": "CryptoBot/1.0"}

    def _fetch_chart(url: str):
        """Fetch a Blockchain.com chart and return ``(data, status)``."""

        data = safe_request(
            url,
            params=params,
            retry_statuses=SERVER_ERROR_CODES,
            backoff_on_429=False,
            headers=headers,
        )

        if data is None:
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=10)
                status = resp.status_code
                ctype = resp.headers.get("content-type", "")
                snippet = resp.text[:200].replace("\n", " ")

                if status == 401:
                    return None, 401
                if status == 404:
                    return None, 404

                if url not in SEEN_NON_JSON_URLS:
                    SEEN_NON_JSON_URLS.add(url)
                    if "application/json" not in ctype.lower():
                        logger.warning(
                            f"⚠️ Non-JSON response ({ctype}) {url}: {snippet}",
                        )
                    else:
                        try:
                            resp.json()
                            logger.warning(
                                f"⚠️ Invalid JSON content from {url}: {snippet}",
                            )
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ JSON decode error for {url}: {e}")

                return None, status
            except Exception as e:  # pragma: no cover - network exceptions
                if url not in SEEN_NON_JSON_URLS:
                    logger.warning(f"⚠️ Exception fetching {url} for inspection: {e}")
                return None, None

        return data, None

    tx_data, tx_status = _fetch_chart(tx_url)
    active_data, active_status = _fetch_chart(active_url)

    if (
        tx_status not in (401, 404)
        and active_status not in (401, 404)
        and not tx_data
        and not active_data
    ):
        logger.warning(
            "⚠️ On-chain metrics API unavailable; returning placeholder data",
        )

    def _parse_blockchain_chart(data: dict, col: str) -> pd.DataFrame:
        """Convert Blockchain.com chart data to a DataFrame."""
        if not isinstance(data, dict):
            return _default_df(col)

        values = None
        if "values" in data:
            values = data["values"]
        elif "data" in data:
            inner = data["data"]
            # Some APIs nest values under another dict
            if isinstance(inner, dict) and "values" in inner:
                values = inner["values"]
            else:
                values = inner

        if not values:
            return _default_df(col)

        if isinstance(values, list) and values:
            first = values[0]
            if isinstance(first, dict):
                cols = {"x": "Timestamp", "y": col, "t": "Timestamp", "v": col, "timestamp": "Timestamp", "value": col}
                df = pd.DataFrame(values)
                rename = {k: v for k, v in cols.items() if k in df.columns}
                if {"Timestamp", col} <= set(rename.values()):
                    df.rename(columns=rename, inplace=True)
                else:
                    return _default_df(col)
            elif isinstance(first, (list, tuple)) and len(first) >= 2:
                df = pd.DataFrame(values, columns=["Timestamp", col])
            else:
                return _default_df(col)
        else:
            return _default_df(col)

        df["Timestamp"] = pd.to_datetime(
            pd.to_numeric(df["Timestamp"], errors="coerce"),
            unit="s",
            errors="coerce",
        )
        return df[["Timestamp", col]]

    def _scrape_chart(slug: str, col: str) -> pd.DataFrame | None:
        """Scrape Blockchain.com explorer HTML as a fallback."""
        url = f"https://www.blockchain.com/charts/{slug}"
        html_params = {"timespan": f"{days}days"}
        try:
            resp = requests.get(url, params=html_params, timeout=10, headers=headers)
            if resp.status_code != 200:
                return None
            match = re.search(r"var\s+data\s*=\s*(\[\[.*?\]\]);", resp.text, re.DOTALL)
            if not match:
                return None
            arr = json.loads(match.group(1))
            df = pd.DataFrame(arr, columns=["Timestamp", col])
            df["Timestamp"] = pd.to_datetime(
                pd.to_numeric(df["Timestamp"], errors="coerce"),
                unit="ms",
                errors="coerce",
            )
            return df[["Timestamp", col]]
        except Exception as e:  # pragma: no cover - network exceptions
            logger.warning(f"⚠️ HTML scrape failed for {url}: {e}")
            return None

    def _handle(data, status, slug: str, col: str) -> pd.DataFrame:
        nonlocal missing
        if status in (401, 404):
            missing = True
            return _default_df(col)
        if data and (("values" in data and data["values"]) or ("data" in data and data["data"])):
            return _parse_blockchain_chart(data, col)
        df = _scrape_chart(slug, col)
        if df is None or df.empty:
            df = _default_df(col)
            missing = True
        return df

    missing = False
    df_tx = _handle(tx_data, tx_status, "n-transactions", "TxVolume")
    df_active = _handle(active_data, active_status, "activeaddresses", "ActiveAddresses")

    df = pd.merge(df_tx, df_active, on="Timestamp", how="outer").sort_values(
        "Timestamp"
    )

    if missing:
        update_cache(placeholder_key, df)
    else:
        update_cache(cache_key, df)
    return df


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
    lock = FileLock(f"{path}.lock")
    with lock:
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
    lock = FileLock(f"{path}.lock")
    with lock:
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
# ✅ OHLCV CACHING HELPERS
# =========================================================
OHLCV_MEMORY_CACHE = {}
OHLCV_TTL = 60  # seconds
# Increased default cache limit so longer histories aren't truncated
OHLCV_CACHE_LIMIT = 20000


def _ohlcv_key(symbol: str, interval: str) -> str:
    return f"ohlcv:{symbol}:{interval}"


def load_ohlcv_cache(symbol: str, interval: str):
    """Return cached OHLCV DataFrame and its age in seconds."""
    key = _ohlcv_key(symbol, interval)

    # Memory first
    entry = OHLCV_MEMORY_CACHE.get(key)
    if entry:
        ts, df = entry
        return df.copy(), time.time() - ts

    # Disk fallback
    path = _cache_path(key)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                ts, df = pickle.load(f)
            OHLCV_MEMORY_CACHE[key] = (ts, df)
            return df.copy(), time.time() - ts
        except Exception as e:
            logger.warning(f"⚠️ Cache read error for {key}: {e}")
    return None, None


def save_ohlcv_cache(symbol: str, interval: str, df: pd.DataFrame, max_size: int = OHLCV_CACHE_LIMIT):
    """Persist OHLCV data to memory and disk with optional size trimming."""
    if df.empty:
        return
    key = _ohlcv_key(symbol, interval)
    df = df.sort_values("Timestamp").tail(max_size)
    OHLCV_MEMORY_CACHE[key] = (time.time(), df)
    update_cache(key, df)

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
    headers: dict | None = None,
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
            req_headers = headers.copy() if headers else None
            if req_headers:
                r = requests.get(
                    url,
                    params=params,
                    timeout=timeout,
                    headers=req_headers,
                )
            else:
                r = requests.get(url, params=params, timeout=timeout)

            if r.status_code == 401:
                logger.error(f"❌ 401 Unauthorized {url}")
                return None

            if r.status_code == 404:
                if url not in SEEN_404_URLS:
                    logger.warning(f"⚠️ 404 Not Found {url}")
                    SEEN_404_URLS.add(url)
                return None
      
            # Special handling for rate limiting
            if r.status_code == 429:
                if 429 in retry_statuses:
                    logger.warning(
                        f"⚠️ Rate limited {url} (attempt {attempt})"
                        + (" → backoff..." if backoff_on_429 else "")
                    )
                    delay = retry_delay * (2 ** (attempt - 1))
                    if backoff_on_429:
                        wait_for_slot(url, backoff=True)
                    time.sleep(delay)
                    continue
                logger.error(f"❌ Failed (429) {url} attempt {attempt}")
                return None

            if r.status_code in retry_statuses:
                logger.error(f"❌ Failed ({r.status_code}) {url} attempt {attempt}")
                delay = retry_delay * (2 ** (attempt - 1))
                time.sleep(delay)
                continue

            if 400 <= r.status_code < 600:
                logger.error(f"❌ Failed ({r.status_code}) {url} (no retry)")
                return None

            content_type = getattr(r, "headers", {}).get("content-type", "")
            if "application/json" not in content_type.lower():
                snippet = r.text[:200].replace("\n", " ")
                logger.error(
                    f"❌ Non-JSON response ({r.status_code}) {url}: {snippet}"
                )
                return None

            try:
                return r.json()
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON decode error for {url}: {e}")
                return None

        except Exception as e:
            logger.error(f"❌ Exception on attempt {attempt} for {url}: {e}")
            delay = retry_delay * (2 ** (attempt - 1))
            time.sleep(delay)

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

def _parse_volume(vol_str: str) -> float:
    """Parse a volume string like "$1.2M" into a float."""
    if not vol_str:
        return 0.0
    vol_str = vol_str.replace("$", "").replace(",", "").strip()
    multiplier = 1
    if vol_str.endswith("B"):
        multiplier = 1_000_000_000
        vol_str = vol_str[:-1]
    elif vol_str.endswith("M"):
        multiplier = 1_000_000
        vol_str = vol_str[:-1]
    elif vol_str.endswith("K"):
        multiplier = 1_000
        vol_str = vol_str[:-1]
    try:
        return float(vol_str) * multiplier
    except ValueError:
        return 0.0


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

        volume = _parse_volume(g.get("volume", ""))

        # ✅ Check availability on Coinbase
        if not is_symbol_on_coinbase(symbol):
            continue

        # 🔍 Resolve coin ID only if symbol is valid
        coin_id = resolve_coin_id(symbol, name)
        if not coin_id:
            continue

        result.append((coin_id, symbol, name, change_pct, volume))

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
# Order matters: sources earlier in this list are preferred and later ones act
# as fallbacks.  Coinbase is attempted first, followed by Binance.US and
# Binance.  If these fail, the fetcher falls back to YFinance, CoinGecko and
# finally DexScreener. Binance Global can be disabled via configuration.
DATA_SOURCES = [
    "coinbase",
    "binance_us",
    "binance" if ENABLE_BINANCE_GLOBAL else None,
    "yfinance",
    "coingecko",
    "dexscreener",
]
DATA_SOURCES = [s for s in DATA_SOURCES if s]

# ``fetch_ohlcv_smart`` may reorder :data:`DATA_SOURCES` for subsequent calls.
# To avoid race conditions when multiple threads attempt to persist such
# reordering, updates to the module-level list are protected by a lock.
from threading import Lock

_DATA_SOURCES_LOCK = Lock()


def fetch_ohlcv_smart(
    symbol,
    interval="15m",
    limit=None,
    cache_limit=None,
    *,
    reorder_global: bool = False,
    **kwargs,
):
    """Smart OHLCV fetcher that tries all sources in order per token.

    Parameters
    ----------
    symbol : str
        Trading pair symbol.
    interval : str, default ``"15m"``
        Candle interval to request.
    limit, cache_limit : int, optional
        Maximum number of rows to return/persist.  ``cache_limit`` defaults to
        ``limit`` and ultimately :data:`OHLCV_CACHE_LIMIT`.
    reorder_global : bool, optional
        When ``True``, a detected source reordering (e.g. deprioritising
        Coinbase) also updates the module-level :data:`DATA_SOURCES`.  This
        mutates global state and affects subsequent calls; updates are
        synchronized via a lock to avoid race conditions.
    """

    ttl = kwargs.get("ttl", OHLCV_TTL)
    if cache_limit is None:
        cache_limit = limit if limit is not None else OHLCV_CACHE_LIMIT

    cached_df, age = load_ohlcv_cache(symbol, interval)
    if cached_df is not None and age is not None and age < ttl:
        logger.info(f"📦 Using cached OHLCV for {symbol} ({interval})")
        return cached_df

    params = kwargs.copy()
    params["interval"] = interval
    params.setdefault("ttl", ttl)
    params.setdefault("cache_limit", cache_limit)
    if limit is not None:
        params.setdefault("limit", limit)
    # Some data providers (e.g. Coinbase, Coingecko, yfinance) expect a
    # start time derived from the number of requested rows.  If the caller
    # supplies ``limit`` but not ``days``, those providers would otherwise
    # default to their own origins (often the Unix epoch) which returns far
    # more history than needed.  Estimate the number of days from ``limit``
    # and ``interval`` so the start date is anchored relative to
    # ``datetime.utcnow()``.
    if "days" not in params and limit is not None:
        try:
            # ``pd.to_timedelta`` understands strings like "15m", "1h", "1d".
            interval_delta = pd.to_timedelta(interval)
        except Exception:
            # Fallback to a conservative 1 day if the interval is unknown.
            interval_delta = pd.Timedelta("1d")
        span = interval_delta * limit
        # Always request at least one day to avoid zero-length ranges.
        params["days"] = max(int(span / pd.Timedelta("1d")), 1)

    # Snapshot of preferred order for this invocation.  ``sources`` is mutated
    # locally so that concurrent calls do not interfere with one another.
    sources = list(DATA_SOURCES)

    # Iterate with a while-loop so ``sources`` can be reordered on the fly for
    # this specific call.
    while sources:
        source = sources.pop(0)
        try:
            if source == "coinbase":
                if not resolve_symbol_coinbase(symbol):
                    logger.info(f"⏭️ Skipping Coinbase for {symbol} (unresolved)")
                    continue
                logger.info(f"⚡ Trying Coinbase for {symbol}")
                df = fetch_coinbase_ohlcv(symbol, **params)
                if len(df) >= 60:
                    return df
                logger.warning(
                    f"⚠️ Coinbase returned {len(df)} rows for {symbol}; prioritizing Binance"
                )
                if ENABLE_BINANCE_GLOBAL:
                    remaining = [
                        "binance_us",
                        "binance",
                        "yfinance",
                        "coingecko",
                        "dexscreener",
                    ]
                    new_global = [
                        "binance_us",
                        "binance",
                        "coinbase",
                        "yfinance",
                        "coingecko",
                        "dexscreener",
                    ]
                else:
                    remaining = [
                        "binance_us",
                        "yfinance",
                        "coingecko",
                        "dexscreener",
                    ]
                    new_global = [
                        "binance_us",
                        "coinbase",
                        "yfinance",
                        "coingecko",
                        "dexscreener",
                    ]

                # Reorder for this call so Binance is attempted before Coinbase
                # on the remaining iterations.
                sources = remaining + [s for s in sources if s not in remaining]

                if reorder_global:
                    with _DATA_SOURCES_LOCK:
                        DATA_SOURCES[:] = new_global
                continue

            elif source == "binance_us":
                if not resolve_symbol_binance_us(symbol):
                    logger.info(f"⏭️ Skipping Binance.US for {symbol} (unresolved)")
                    continue
                logger.info(f"⚡ Trying Binance.US for {symbol}")
                df = fetch_binance_us_ohlcv(symbol, **params)
                if not df.empty:
                    return df

            elif source == "binance":
                if not resolve_symbol_binance_global(symbol):
                    logger.info(f"⏭️ Skipping Binance for {symbol} (unresolved or blocked)")
                    continue
                logger.info(f"⚡ Trying Binance for {symbol}")
                df = fetch_binance_ohlcv(symbol, **params)
                if not df.empty:
                    return df

            elif source == "yfinance":
                logger.info(f"⚡ Trying YFinance for {symbol}")
                df = fetch_from_yfinance(
                    symbol,
                    interval=interval,
                    days=params.get("days", 1),
                    limit=params.get("limit"),
                )
                if not df.empty:
                    save_ohlcv_cache(symbol, interval, df, cache_limit)
                    return df

            elif source == "dexscreener":
                logger.info(f"⚡ Trying DexScreener for {symbol}")
                df = fetch_dexscreener_ohlcv(symbol, limit=params.get("limit"))
                if not df.empty:
                    save_ohlcv_cache(symbol, interval, df, cache_limit)
                    return df

            elif source == "coingecko":
                logger.info(
                    f"⚡ Trying Coingecko for {symbol} (days={params.get('days', 1)})"
                )
                df = fetch_coingecko_ohlcv(
                    params.get("coin_id", symbol),
                    days=params.get("days", 1),
                    limit=params.get("limit"),
                    headers=params.get("headers"),
                )
                if not df.empty:
                    save_ohlcv_cache(symbol, interval, df, cache_limit)
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
    ttl = kwargs.get("ttl", OHLCV_TTL)
    cache_limit = kwargs.get(
        "cache_limit", kwargs.get("limit") or OHLCV_CACHE_LIMIT
    )
    cached, age = load_ohlcv_cache(symbol, interval)
    if cached is not None and age is not None and age < ttl:
        logger.info(f"📦 Using cached Binance OHLCV for {symbol}")
        return cached
    if not ENABLE_BINANCE_GLOBAL:
        return cached if cached is not None else pd.DataFrame()

    binance_symbol = resolve_symbol_binance_global(symbol)
    if not binance_symbol:
        logger.warning(f"⚠️ Symbol {symbol.upper()} not found on Binance Global.")
        return cached if cached is not None else pd.DataFrame()

    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": interval}
    if cached is None:
        params["limit"] = min(limit, 1000)
    else:
        last_ts = cached["Timestamp"].max()
        params["startTime"] = int((last_ts.to_pydatetime().timestamp() + 1) * 1000)

    for attempt in range(3):
        wait_for_slot(url)
        r = requests.get(url, params=params, timeout=10)

        if r.status_code == 451:
            logger.warning(f"⚠️ Binance 451 for {symbol} — skipping.")
            return cached if cached is not None else pd.DataFrame()

        try:
            r.raise_for_status()
            data = r.json()
            if not data:
                return cached if cached is not None else pd.DataFrame()
            df_new = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "x", "y", "z", "a", "b", "c"
            ])
            df_new["Close"] = df_new["close"].astype(float)
            df_new["Timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")
            df = df_new[["Timestamp", "Close"]]
            if cached is not None:
                df = pd.concat([cached, df])
            df = df.drop_duplicates("Timestamp").sort_values("Timestamp").tail(cache_limit)
            save_ohlcv_cache(symbol, interval, df, cache_limit)
            return df
        except Exception as e:
            logger.error(f"❌ Binance fetch fail ({symbol}): {e}")
            time.sleep(2)

    logger.error(f"❌ All Binance fetch attempts failed for {symbol}")
    return cached if cached is not None else pd.DataFrame()


def fetch_binance_us_ohlcv(symbol, interval="15m", limit=96, **kwargs):
    ttl = kwargs.get("ttl", OHLCV_TTL)
    cache_limit = kwargs.get(
        "cache_limit", kwargs.get("limit") or OHLCV_CACHE_LIMIT
    )
    cached, age = load_ohlcv_cache(symbol, interval)
    if cached is not None and not cached.empty:
        # Ensure cached timestamps are timezone-aware (UTC) so comparisons
        # against newly fetched data do not raise ``TypeError`` about mixing
        # offset-naive and offset-aware datetimes.
        cached["Timestamp"] = pd.to_datetime(cached["Timestamp"], utc=True)

    if cached is not None and age is not None and age < ttl:
        logger.info(f"📦 Using cached Binance.US OHLCV for {symbol}")
        return cached

    binance_symbol = resolve_symbol_binance_us(symbol)
    if not binance_symbol:
        logger.warning(f"⚠️ Symbol {symbol.upper()} not found on Binance.US.")
        return cached if cached is not None else pd.DataFrame()

    url = "https://api.binance.us/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": interval}
    if cached is None:
        params["limit"] = min(limit, 1000)
    else:
        # ``cached`` has timezone-aware timestamps, so convert to POSIX
        # milliseconds using UTC to request only missing candles.
        last_ts = cached["Timestamp"].max()
        params["startTime"] = int((last_ts.to_pydatetime().timestamp() + 1) * 1000)

    wait_for_slot(url)
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return cached if cached is not None else pd.DataFrame()

        df_new = pd.DataFrame(
            data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "x",
                "y",
                "z",
                "a",
                "b",
                "c",
            ],
        )
        df_new["Close"] = df_new["close"].astype(float)
        # Parse Binance timestamps as UTC to match cached data
        df_new["Timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
        df = df_new[["Timestamp", "Close"]]
        if cached is not None:
            # Concatenate with cached data (already UTC) and sort
            df = pd.concat([cached, df])
        df = df.drop_duplicates("Timestamp").sort_values("Timestamp").tail(cache_limit)
        save_ohlcv_cache(symbol, interval, df, cache_limit)
        return df
    except Exception as e:
        logger.error(f"❌ Binance.US fetch fail ({symbol}): {e}")
        return cached if cached is not None else pd.DataFrame()


def fetch_coinbase_ohlcv(symbol, interval="15m", days=1, limit=300, **kwargs):
    """Fetch OHLCV data from Coinbase with caching and incremental updates."""

    ttl = kwargs.get("ttl", OHLCV_TTL)
    cache_limit = kwargs.get(
        "cache_limit", kwargs.get("limit") or OHLCV_CACHE_LIMIT
    )
    chunk_limit = min(limit or 300, 300)
    cached, age = load_ohlcv_cache(symbol, interval)
    if cached is not None and not cached.empty:
        # Normalize cached timestamps to UTC so subsequent comparisons work
        cached["Timestamp"] = pd.to_datetime(cached["Timestamp"], utc=True)
    if cached is not None and age is not None and age < ttl:
        logger.info(f"📦 Using cached Coinbase OHLCV for {symbol}")
        return cached

    product_id = resolve_symbol_coinbase(symbol)
    if not product_id:
        logger.warning(f"⚠️ Symbol {symbol.upper()} not found on Coinbase.")
        return cached if cached is not None else pd.DataFrame()

    gran_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
    granularity = gran_map.get(interval, 900)

    # Use timezone-aware timestamps (UTC) for start/end so comparisons with
    # cached data do not raise ``TypeError``.
    end = pd.Timestamp.utcnow().tz_convert("UTC")
    if cached is not None and not cached.empty:
        last = cached["Timestamp"].max().tz_convert("UTC")
        start = last + pd.Timedelta(seconds=granularity)
    else:
        start = end - pd.Timedelta(days=days)
    max_span = pd.Timedelta(seconds=granularity * chunk_limit)
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
            df = cached if cached is not None else pd.DataFrame()
            df.attrs["fetch_seconds"] = time.time() - start_time
            return df

        frames.append(pd.DataFrame(data, columns=["Timestamp", "Low", "High", "Open", "Close", "Volume"]))
        window_start = window_end
        chunk += 1

    elapsed = time.time() - start_time

    if not frames:
        df = cached if cached is not None else pd.DataFrame()
        df.attrs["fetch_seconds"] = elapsed
        return df

    try:
        df_new = pd.concat(frames, ignore_index=True)
        df_new["Timestamp"] = pd.to_datetime(df_new["Timestamp"], unit="s", utc=True)
        df_new = df_new.sort_values("Timestamp")
        df_new = df_new[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
        if cached is not None:
            df = pd.concat([cached, df_new])
        else:
            df = df_new
        # Ensure the concatenated frame retains UTC timezone information
        tz = getattr(df["Timestamp"].dt, "tz", None)
        if tz is None:
            df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
        else:
            df["Timestamp"] = df["Timestamp"].dt.tz_convert("UTC")
        df = df.drop_duplicates("Timestamp").sort_values("Timestamp").tail(cache_limit)
        df.attrs["fetch_seconds"] = elapsed
        save_ohlcv_cache(symbol, interval, df, cache_limit)
        return df
    except Exception as e:
        logger.error(f"❌ Coinbase fetch fail ({symbol}): {e}")
        df = cached if cached is not None else pd.DataFrame()
        df.attrs["fetch_seconds"] = elapsed
        return df


def fetch_dexscreener_ohlcv(symbol, limit=None):
    """DexScreener only provides the latest price; ``limit`` is ignored."""
    url = f"https://api.dexscreener.com/latest/dex/search?q={symbol.lower()}"
    data = safe_request(url)
    if not data or "pairs" not in data:
        return pd.DataFrame()
    price = float(data["pairs"][0]["priceUsd"])
    ts = pd.Timestamp.utcnow()
    return pd.DataFrame({"Timestamp": [ts], "Close": [price]})


def fetch_from_yfinance(symbol, interval="1h", days=10, limit=None):
    yf_candidates = [
        f"{symbol.upper()}-USD",
        f"{symbol.upper()}-CRYPTO",
        f"{symbol.upper()}",
    ]
    interval_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "1d": "1d",
    }
    yf_interval = interval_map.get(interval, "1h")

    for yf_symbol in yf_candidates:
        try:
            logger.info(f"🔎 Trying yfinance ticker: {yf_symbol}")
            df = yf.download(
                yf_symbol,
                period=f"{days}d",
                interval=yf_interval,
                progress=False,
                auto_adjust=False,
            )
            if not df.empty and "Close" in df.columns and df["Close"].dropna().shape[0] > 0:
                df = df.rename(
                    columns={
                        "Open": "Open",
                        "High": "High",
                        "Low": "Low",
                        "Close": "Close",
                        "Volume": "Volume",
                    }
                )[["Open", "High", "Low", "Close", "Volume"]]
                df = df.rename_axis("Timestamp").reset_index()
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
                if limit is not None:
                    df = df.sort_values("Timestamp").tail(limit)
                logger.info(f"✅ YFinance data loaded for {yf_symbol}")
                return df
        except Exception as e:
            logger.warning(f"⚠️ Failed for {yf_symbol}: {e}")

    logger.error(f"❌ YFinance could not fetch data for {symbol}")
    return pd.DataFrame()


def fetch_coingecko_ohlcv(coin_id, days=1, limit=None, headers=None):
    cache_key = f"cg_ohlcv:{coin_id}:{days}"
    cached = cached_fetch(cache_key, ttl=180)  # 3 min TTL
    if cached is not None:
        logger.info(f"📦 Using cached Coingecko OHLCV for {coin_id}")
        return cached

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    api_key = os.getenv("COINGECKO_API_KEY", "CG-DEMO-API-KEY")

    combined_headers = headers.copy() if headers else {}
    # First attempt without the API key header
    for attempt in range(2):
        req_headers = combined_headers.copy()
        if attempt == 1:
            req_headers["x-cg-demo-api-key"] = api_key

        try:
            wait_for_slot(url)
            r = requests.get(url, params=params, timeout=10, headers=req_headers or None)
        except Exception as e:  # pragma: no cover - network exceptions
            logger.error(f"❌ Exception fetching Coingecko OHLCV: {e}")
            return pd.DataFrame()

        if r.status_code == 401 and attempt == 0:
            logger.warning("⚠️ Coingecko 401 Unauthorized, retrying with API key")
            continue

        if r.status_code != 200:
            logger.error(
                f"❌ Coingecko OHLCV fetch failed ({r.status_code}) for {coin_id}"
            )
            return pd.DataFrame()

        try:
            data = r.json()
        except Exception as e:
            logger.error(f"❌ JSON decode error for Coingecko OHLCV: {e}")
            return pd.DataFrame()

        if data and "prices" in data:
            df = pd.DataFrame(data["prices"], columns=["Timestamp", "Close"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
            if limit is not None:
                df = df.sort_values("Timestamp").tail(limit)
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


