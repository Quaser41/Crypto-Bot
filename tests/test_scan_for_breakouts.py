import importlib
import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import threading


def _reload_main(monkeypatch):
    """Reload the main module with thread creation suppressed."""
    dummy_thread = types.SimpleNamespace(start=lambda: None)
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: dummy_thread)
    if "main" in sys.modules:
        main = sys.modules["main"]
        return importlib.reload(main)
    return importlib.import_module("main")


def _mock_df():
    data = {
        "Close": [100] * 60,
        "RSI": [60] * 60,
        "MACD": [1] * 60,
        "Signal": [0] * 60,
        "Hist": [1] * 60,
        "SMA_20": [95] * 60,
        "SMA_50": [90] * 60,
        "Return_1d": [0.02] * 60,
        "Volatility_7d": [0.02] * 60,
        "Momentum_Tier": ["Tier 1"] * 60,
        "Momentum_Score": [0.8] * 60,
        "Return_3d": [0.04] * 60,
    }
    return pd.DataFrame(data)


def test_scan_for_breakouts_opens_trade(monkeypatch):
    main = _reload_main(monkeypatch)

    tm = types.SimpleNamespace(
        positions={},
        can_trade=MagicMock(return_value=True),
        open_trade=MagicMock(),
        save_state=lambda: None,
        summary=lambda: None,
    )
    monkeypatch.setattr(main, "tm", tm)
    monkeypatch.setattr(main, "get_top_gainers", lambda limit=15: [("id1", "ABC", "ABC Coin", 10.0)])
    monkeypatch.setattr(main, "fetch_ohlcv_smart", lambda *a, **k: _mock_df())
    monkeypatch.setattr(main, "add_indicators", lambda d: d)
    monkeypatch.setattr(main, "predict_signal", lambda df, threshold: ("BUY", 0.95, 4))
    monkeypatch.setattr(main, "get_dynamic_threshold", lambda vol, base: 0.5)

    main.scan_for_breakouts()

    tm.open_trade.assert_called_once()
    args, kwargs = tm.open_trade.call_args
    assert args[0] == "ABC"
    assert args[1] == 100


def test_scan_for_breakouts_respects_risk_limits(monkeypatch):
    main = _reload_main(monkeypatch)

    tm = types.SimpleNamespace(
        positions={},
        can_trade=MagicMock(return_value=False),
        open_trade=MagicMock(),
    )
    monkeypatch.setattr(main, "tm", tm)

    def fail_get_top_gainers(*a, **k):
        raise AssertionError("get_top_gainers should not be called when risk limits hit")

    monkeypatch.setattr(main, "get_top_gainers", fail_get_top_gainers)

    main.scan_for_breakouts()

    assert not tm.open_trade.called


def test_scan_for_breakouts_skips_low_performance(monkeypatch):
    main = _reload_main(monkeypatch)

    tm = types.SimpleNamespace(
        positions={},
        can_trade=MagicMock(return_value=True),
        open_trade=MagicMock(),
    )
    monkeypatch.setattr(main, "tm", tm)

    # Configure performance data to fail thresholds
    main.SYMBOL_PERFORMANCE = {"BAD": {"avg_pnl": -0.1, "win_rate": 10}}
    main.MIN_SYMBOL_WIN_RATE = 60
    main.MIN_SYMBOL_AVG_PNL = 0.05

    monkeypatch.setattr(main, "get_top_gainers", lambda limit=15: [("id1", "BAD", "Bad", 10.0)])

    fetch_mock = MagicMock(side_effect=lambda *a, **k: _mock_df())
    monkeypatch.setattr(main, "fetch_ohlcv_smart", fetch_mock)

    main.scan_for_breakouts()

    fetch_mock.assert_not_called()
    assert not tm.open_trade.called


def test_low_buy_override_threshold_allows_trade(monkeypatch):
    main = _reload_main(monkeypatch)

    tm = types.SimpleNamespace(
        positions={},
        can_trade=MagicMock(return_value=True),
        open_trade=MagicMock(),
        save_state=lambda: None,
        summary=lambda: None,
    )
    monkeypatch.setattr(main, "tm", tm)
    monkeypatch.setattr(main, "get_top_gainers", lambda limit=15: [("id1", "ABC", "ABC Coin", 10.0)])
    monkeypatch.setattr(main, "fetch_ohlcv_smart", lambda *a, **k: _mock_df())
    monkeypatch.setattr(main, "add_indicators", lambda d: d)

    def fake_predict(df, threshold):
        return ("BUY" if 0.7 >= main.HIGH_CONF_BUY_OVERRIDE else "HOLD", 0.7, 3)

    monkeypatch.setattr(main, "predict_signal", fake_predict)
    monkeypatch.setattr(main, "get_dynamic_threshold", lambda vol, base: 0.5)

    main.CONFIDENCE_THRESHOLD = 0.65
    main.HIGH_CONF_BUY_OVERRIDE = 0.8
    main.scan_for_breakouts()
    assert not tm.open_trade.called

    tm.open_trade.reset_mock()
    main.HIGH_CONF_BUY_OVERRIDE = 0.6
    main.scan_for_breakouts()
    tm.open_trade.assert_called_once()


def test_low_volatility_threshold_allows_trade(monkeypatch):
    main = _reload_main(monkeypatch)

    tm = types.SimpleNamespace(
        positions={},
        can_trade=MagicMock(return_value=True),
        open_trade=MagicMock(),
        save_state=lambda: None,
        summary=lambda: None,
    )
    monkeypatch.setattr(main, "tm", tm)
    monkeypatch.setattr(main, "get_top_gainers", lambda limit=15: [("id1", "ABC", "ABC Coin", 10.0)])

    def low_vol_df():
        df = _mock_df()
        df["Volatility_7d"] = [0.00005] * 60
        return df

    monkeypatch.setattr(main, "fetch_ohlcv_smart", lambda *a, **k: low_vol_df())
    monkeypatch.setattr(main, "add_indicators", lambda d: d)
    monkeypatch.setattr(main, "predict_signal", lambda df, threshold: ("BUY", 0.95, 4))
    monkeypatch.setattr(main, "get_dynamic_threshold", lambda vol, base: 0.5)

    main.MIN_VOLATILITY_7D = 0.0001
    main.scan_for_breakouts()
    assert not tm.open_trade.called

    tm.open_trade.reset_mock()
    main.MIN_VOLATILITY_7D = 0.0
    main.scan_for_breakouts()
    tm.open_trade.assert_called_once()
