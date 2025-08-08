import time
from threading import Lock

# CoinGecko free tier -> ~50 calls/minute (but safer ~30/min)
MIN_DELAY = 1.5   # slightly safer than 1.3s
BACKOFF_DELAY = 5 # if 429 hit, wait 5s before retry

_last_call = 0.0
_lock = Lock()

def wait_for_slot(backoff=False):
    """
    Ensures thread-safe sequential API requests:
    - Only one request at a time (via Lock)
    - Enforces MIN_DELAY spacing
    - If backoff=True, adds an extra delay (after a 429)
    """
    global _last_call
    with _lock:
        now = time.time()
        elapsed = now - _last_call

        # ✅ Enforce minimum spacing between ANY calls
        if elapsed < MIN_DELAY:
            time.sleep(MIN_DELAY - elapsed)

        # ✅ Extra backoff if explicitly requested
        if backoff:
            time.sleep(BACKOFF_DELAY)

        # ✅ Update last call timestamp *after* sleeping
        _last_call = time.time()
