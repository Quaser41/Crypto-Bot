import csv
import os
import time
from typing import Set, Tuple

# Default location for the trade statistics CSV
DEFAULT_STATS_FILE = os.path.join(os.path.dirname(__file__), "trade_stats.csv")

# Cached blacklist and timestamp of last refresh
_blacklist: Set[Tuple[str, str]] = set()
_last_loaded: float = 0.0


def _parse_stats(path: str) -> Set[Tuple[str, str]]:
    """Parse the trade stats CSV and return blacklist pairs.

    A pair ``(symbol, duration_bucket)`` is blacklisted when the win rate is 0
    or the average PnL is negative.
    """
    pairs: Set[Tuple[str, str]] = set()
    if not os.path.exists(path):
        return pairs

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                win_rate = float(row.get("win_rate", 0))
                avg_pnl = float(row.get("avg_pnl", 0))
            except (TypeError, ValueError):
                continue
            if win_rate == 0 or avg_pnl < 0:
                symbol = row.get("symbol", "").upper()
                bucket = row.get("duration_bucket", "")
                pairs.add((symbol, bucket))
    return pairs


def load_blacklist(path: str = DEFAULT_STATS_FILE, refresh_seconds: int = 3600) -> Set[Tuple[str, str]]:
    """Return cached blacklist, reloading from CSV when stale."""
    global _blacklist, _last_loaded
    now = time.time()
    if not _blacklist or now - _last_loaded > refresh_seconds:
        _blacklist = _parse_stats(path)
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
    global _blacklist, _last_loaded
    _blacklist = set()
    _last_loaded = 0.0
