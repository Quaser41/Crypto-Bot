import time
from collections import defaultdict
from threading import Lock
from urllib.parse import urlparse

# Default rate limits
MIN_DELAY = 1.5              # CoinGecko free tier -> ~50 calls/minute (but safer ~30/min)
COINBASE_DELAY = 0.35        # ~3 requests/second for Coinbase
BACKOFF_DELAY = 5            # if 429 hit, wait 5s before retry

# Track last-call times and locks per host so different APIs don't block each other
_last_call = defaultdict(float)
_locks = defaultdict(Lock)


def _host_from_url(url: str) -> str:
    if not url:
        return ""
    return urlparse(url).netloc


def wait_for_slot(url: str = "", backoff: bool = False):
    """Rate‑limit requests on a per-host basis.

    When ``url`` points to Coinbase, use a much smaller delay so we don't wait
    1.5s between requests. Other hosts fall back to ``MIN_DELAY``. Calls to
    different hosts do not block each other.
    """

    host = _host_from_url(url)
    min_delay = COINBASE_DELAY if "coinbase.com" in host else MIN_DELAY

    lock = _locks[host]
    with lock:
        now = time.time()
        elapsed = now - _last_call[host]

        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)

        if backoff:
            time.sleep(BACKOFF_DELAY)

        _last_call[host] = time.time()
