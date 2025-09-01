import os
import threading
import time

import data_fetcher


def test_fetch_fear_greed_index_uses_cache(monkeypatch, tmp_path):
    calls = {"count": 0}
    sample = {"data": [{"timestamp": "1700000000", "value": "60"}]}

    def mock_safe_request(url, params=None, **kwargs):
        calls["count"] += 1
        return sample

    monkeypatch.setattr(data_fetcher, "safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)

    df1 = data_fetcher.fetch_fear_greed_index(limit=1)
    df2 = data_fetcher.fetch_fear_greed_index(limit=1)

    assert not df1.empty and not df2.empty
    assert calls["count"] == 1


def test_fear_greed_cache_concurrent(monkeypatch, tmp_path):
    calls = {"count": 0}
    sample = {"data": [{"timestamp": "1700000000", "value": "60"}]}

    def mock_safe_request(url, params=None, **kwargs):
        calls["count"] += 1
        return sample

    # Slow down cache writing to simulate race conditions
    write_started = threading.Event()
    proceed_write = threading.Event()

    def slow_update_cache(key, data):
        path = data_fetcher._cache_path(key)
        lock = data_fetcher.FileLock(f"{path}.lock")
        with lock:
            write_started.set()
            proceed_write.wait()
            with open(path, "wb") as f:
                import pickle
                pickle.dump((time.time(), data), f)

    monkeypatch.setattr(data_fetcher, "safe_request", mock_safe_request)
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_fetcher, "update_cache", slow_update_cache)
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)

    def first_call():
        df = data_fetcher.fetch_fear_greed_index(limit=1)
        assert not df.empty

    def second_call():
        write_started.wait()
        df = data_fetcher.fetch_fear_greed_index(limit=1)
        assert not df.empty

    t1 = threading.Thread(target=first_call)
    t2 = threading.Thread(target=second_call)
    t1.start()
    t2.start()
    # Ensure the second thread attempts to read while the first is writing
    time.sleep(0.1)
    proceed_write.set()
    t1.join()
    t2.join()

    assert calls["count"] == 1

