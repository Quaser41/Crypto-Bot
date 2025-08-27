import json
import threading


def test_rotation_audit_thread_safety(tmp_path):
    import main

    # Start with a clean audit log
    with main.ROTATION_AUDIT_LOCK:
        main.ROTATION_AUDIT_LOG.clear()

    def worker(prefix):
        for i in range(5):
            current = {
                "symbol": f"CUR_{prefix}_{i}",
                "confidence": 0.0,
                "label": "",
                "entry_price": 0.0,
                "movement": 0.0,
            }
            candidate = {
                "symbol": f"NEW_{prefix}_{i}",
                "confidence": 0.0,
                "label": "",
                "price": 0.0,
            }
            main.record_rotation_audit(current, candidate, pnl_before=0.0)

    # Two threads writing concurrently
    threads = [threading.Thread(target=worker, args=("A",)), threading.Thread(target=worker, args=("B",))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify all entries were recorded (5 per thread, log limit is 10)
    with main.ROTATION_AUDIT_LOCK:
        assert len(main.ROTATION_AUDIT_LOG) == 10
        snapshot = list(main.ROTATION_AUDIT_LOG)

    # Persist the audit log and ensure the same data is saved
    audit_path = tmp_path / "audit.json"
    with main.ROTATION_AUDIT_LOCK:
        main.save_rotation_audit(filepath=str(audit_path), max_entries=100)

    with open(audit_path) as f:
        saved = json.load(f)
    assert saved == snapshot

    # Log should be cleared after saving
    with main.ROTATION_AUDIT_LOCK:
        assert main.ROTATION_AUDIT_LOG == []
