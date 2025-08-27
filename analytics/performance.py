import csv
import os
import time
from typing import Dict, Set, Tuple

# Default location for the trade statistics CSV
DEFAULT_STATS_FILE = os.path.join(os.path.dirname(__file__), "trade_stats.csv")

# Maximum acceptable fees relative to PnL before blacklisting
# Allow override via ``FEE_RATIO_THRESHOLD`` environment variable.
FEE_RATIO_THRESHOLD = float(os.getenv("FEE_RATIO_THRESHOLD", "1.0"))

# Minimum number of trades required before considering a pair for blacklisting
# Allow override via ``MIN_TRADE_COUNT`` environment variable.
MIN_TRADE_COUNT = int(os.getenv("MIN_TRADE_COUNT", "3"))

# Cached blacklist, trade counts, and timestamp of last refresh
_blacklist: Set[Tuple[str, str]] = set()
_trade_counts: Dict[Tuple[str, str], int] = {}
_last_loaded: float = 0.0


def _parse_stats(path: str) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], int]]:
    """Parse the trade stats CSV and return blacklist pairs and trade counts.

    A pair ``(symbol, duration_bucket)`` is blacklisted when the win rate is 0,
    the average PnL is negative, or the ``fee_ratio`` exceeds
    :data:`FEE_RATIO_THRESHOLD`, *and* it has at least
    :data:`MIN_TRADE_COUNT` trades.
    """
    pairs: Set[Tuple[str, str]] = set()
    counts: Dict[Tuple[str, str], int] = {}
    if not os.path.exists(path):
        return pairs, counts

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                trade_count = int(row.get("trade_count", 0))
                win_rate = float(row.get("win_rate", 0))
                avg_pnl = float(row.get("avg_pnl", 0))
                fee_ratio = float(row.get("fee_ratio", 0))
            except (TypeError, ValueError):
                continue
            symbol = row.get("symbol", "").upper()
            bucket = row.get("duration_bucket", "")
            counts[(symbol, bucket)] = trade_count
            if (
                trade_count >= MIN_TRADE_COUNT
                and (win_rate == 0 or avg_pnl < 0 or fee_ratio > FEE_RATIO_THRESHOLD)
            ):
                pairs.add((symbol, bucket))
    return pairs, counts


def load_blacklist(path: str = DEFAULT_STATS_FILE, refresh_seconds: int = 3600) -> Set[Tuple[str, str]]:
    """Return cached blacklist, reloading from CSV when stale."""
    global _blacklist, _trade_counts, _last_loaded
    now = time.time()
    if not _blacklist or now - _last_loaded > refresh_seconds:
        _blacklist, _trade_counts = _parse_stats(path)
        _last_loaded = now
    return _blacklist


def is_blacklisted(
    symbol: str,
    duration_bucket: str,
    path: str = DEFAULT_STATS_FILE,
    refresh_seconds: int = 3600,
) -> bool:
    """Return True if the symbol and duration bucket are blacklisted."""
    bl = load_blacklist(path, refresh_seconds)
    return (symbol.upper(), duration_bucket) in bl


def get_trade_count(
    symbol: str,
    duration_bucket: str,
    path: str = DEFAULT_STATS_FILE,
    refresh_seconds: int = 3600,
) -> int:
    """Return the trade count for the given symbol and duration bucket."""
    load_blacklist(path, refresh_seconds)
    return _trade_counts.get((symbol.upper(), duration_bucket), 0)


def get_duration_bucket(seconds: float) -> str:
    """Map a duration in seconds to the analytics bucket label."""
    if seconds < 60:
        return "<1m"
    if seconds < 5 * 60:
        return "1-5m"
    if seconds < 30 * 60:
        return "5-30m"
    if seconds < 2 * 3600:
        return "30m-2h"
    return ">2h"


def reset_cache() -> None:
    """Clear cached blacklist (primarily for tests)."""
    global _blacklist, _trade_counts, _last_loaded
    _blacklist = set()
    _trade_counts = {}
    _last_loaded = 0.0
