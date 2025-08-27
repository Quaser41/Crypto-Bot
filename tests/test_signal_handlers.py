import json
import signal
import importlib
from pathlib import Path
import sys
import pytest


def test_signal_handlers_write_state_and_audit(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    sys.path.append(str(repo_root))
    main = importlib.import_module("main")
    importlib.reload(main)

    main.ENABLE_ROTATION_AUDIT = True
    main.PERSIST_ROTATION_AUDIT = True
    main.ROTATION_AUDIT_LOG.append({"dummy": True})

    original_term = signal.getsignal(signal.SIGTERM)
    original_int = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(SystemExit):
            signal.raise_signal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGINT, original_int)

    state_file = tmp_path / "trade_manager_state.json"
    audit_file = tmp_path / "rotation_audit.json"
    assert state_file.exists()
    with state_file.open() as f:
        json.load(f)
    assert audit_file.exists()
    with audit_file.open() as f:
        assert json.load(f)
