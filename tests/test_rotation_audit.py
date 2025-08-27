import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _add_entry(idx):
    import main
    current = {
        "symbol": f"CUR_{idx}",
        "confidence": 0.0,
        "label": "",
        "entry_price": 0.0,
        "movement": 0.0,
    }
    candidate = {
        "symbol": f"NEW_{idx}",
        "confidence": 0.0,
        "label": "",
        "price": 0.0,
    }
    main.record_rotation_audit(current, candidate, pnl_before=0.0)


def test_save_rotation_audit_respects_max_entries(tmp_path):
    import main
    audit_path = tmp_path / "audit.json"
    with main.ROTATION_AUDIT_LOCK:
        main.ROTATION_AUDIT_LOG.clear()

    for i in range(3):
        _add_entry(i)

    with main.ROTATION_AUDIT_LOCK:
        main.save_rotation_audit(filepath=str(audit_path), max_entries=2)

    with open(audit_path) as f:
        data = json.load(f)
    assert len(data) == 2
    assert [d["current"]["symbol"] for d in data] == ["CUR_1", "CUR_2"]

    for i in range(3, 7):
        _add_entry(i)

    with main.ROTATION_AUDIT_LOCK:
        main.save_rotation_audit(filepath=str(audit_path), max_entries=2)

    with open(audit_path) as f:
        data = json.load(f)
    assert len(data) == 2
    assert [d["current"]["symbol"] for d in data] == ["CUR_5", "CUR_6"]

    audit_path.unlink()


def test_save_rotation_audit_empty_log(tmp_path):
    import main
    audit_path = tmp_path / "audit.json"
    with main.ROTATION_AUDIT_LOCK:
        main.ROTATION_AUDIT_LOG.clear()
        main.save_rotation_audit(filepath=str(audit_path), max_entries=2)

    assert not audit_path.exists()
    if audit_path.exists():
        audit_path.unlink()
