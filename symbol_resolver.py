import requests

from utils.logging import get_logger
from config import (
    MIN_SYMBOL_WIN_RATE,
    MIN_SYMBOL_AVG_PNL,
    MIN_HOLD_BUCKET,
    MIN_24H_VOLUME,
)
from analytics import performance as perf_utils

logger = get_logger(__name__)

BINANCE_GLOBAL_SYMBOLS = {}
BINANCE_US_SYMBOLS = {}
COINBASE_SYMBOLS = {}

# Duration bucket ordering used when enforcing minimum hold requirements.
DURATION_BUCKETS = ["<1m", "1-5m", "5-30m", "30m-2h", ">2h"]


def _bucket_index(bucket: str) -> int:
    """Return numeric index for duration bucket comparison."""
    try:
        return DURATION_BUCKETS.index(bucket)
    except ValueError:
        return len(DURATION_BUCKETS)

def load_binance_global_symbols():
    global BINANCE_GLOBAL_SYMBOLS
    if BINANCE_GLOBAL_SYMBOLS:
        return

    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 451:
            logger.warning("⚠️ Binance Global blocked (451) – skipping global symbols.")
            return
        r.raise_for_status()
        data = r.json()
        for sym in data["symbols"]:
            if sym["status"] == "TRADING" and sym["quoteAsset"] == "USDT":
                BINANCE_GLOBAL_SYMBOLS[sym["baseAsset"]] = sym["symbol"]
    except Exception as e:
        logger.error("❌ Failed to load Binance Global symbols: %s - %s", type(e).__name__, e)

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
        logger.error("❌ Failed to load Binance.US symbols: %s - %s", type(e).__name__, e)


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
        logger.error("❌ Failed to load Coinbase symbols: %s - %s", type(e).__name__, e)


def resolve_symbol_coinbase(base_asset):
    load_coinbase_symbols()
    return COINBASE_SYMBOLS.get(base_asset.upper())


def filter_candidates(movers, open_symbols, performance):
    """Filter mover list using historical performance thresholds.

    Parameters
    ----------
    movers : iterable
        Sequence of tuples ``(coin_id, symbol, name, change, volume)``.
    open_symbols : iterable
        Symbols that already have open positions and should be skipped.
    performance : dict
        Mapping of symbol -> {"avg_pnl": float, "win_rate": float}.

    Returns
    -------
    list
        Filtered list of tuples ``(coin_id, symbol, name)`` for further
        processing.
    """

    candidates = []
    for coin_id, symbol, name, _, volume in movers:
        if symbol in open_symbols:
            logger.info(f"⏭️ Skipping {symbol} (already open trade)")
            continue

        if volume < MIN_24H_VOLUME:
            logger.info(
                f"⏭️ Skipping {symbol}: volume {volume:.0f} below {MIN_24H_VOLUME}"
            )
            continue

        perf = performance.get(symbol)
        if perf:
            if perf.get("avg_pnl", 0) <= MIN_SYMBOL_AVG_PNL:
                logger.info(
                    f"⏭️ Skipping {symbol}: avg PnL {perf['avg_pnl']:.2f} below {MIN_SYMBOL_AVG_PNL}"
                )
                continue
            if perf.get("win_rate", 0) < MIN_SYMBOL_WIN_RATE:
                logger.info(
                    f"⏭️ Skipping {symbol}: win rate {perf['win_rate']:.2f}% below {MIN_SYMBOL_WIN_RATE}%"
                )
                continue

            holding_times = perf.get("holding_times")
            if holding_times:
                avg_hold = sum(holding_times) / len(holding_times)
                duration_bucket = perf_utils.get_duration_bucket(avg_hold)
                if _bucket_index(duration_bucket) < _bucket_index(MIN_HOLD_BUCKET):
                    logger.info(
                        f"⏭️ Skipping {symbol}: average hold {duration_bucket} below {MIN_HOLD_BUCKET}"
                    )
                    continue
                if perf_utils.is_blacklisted(symbol, duration_bucket):
                    logger.info(
                        f"⏭️ Skipping {symbol}: blacklisted for duration {duration_bucket}"
                    )
                    continue

        candidates.append((coin_id, symbol, name))

    return candidates
