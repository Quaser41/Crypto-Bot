import pandas as pd
import pytest

from trade_manager import TradeManager


def create_tm():
    tm = TradeManager(starting_balance=1000)
    tm.risk_per_trade = 0.1
    tm.min_trade_usd = 0
    tm.trade_fee_pct = 0.0
    return tm


def mock_indicator_df():
    return pd.DataFrame({
        'ATR': [1.0],
        'Return_1d': [0.01],
        'Return_3d': [0.03],
        'RSI': [55],
        'Hist': [0.1]
    })


def test_calculate_allocation():
    tm = create_tm()
    alloc = tm.calculate_allocation(confidence=0.5)
    assert alloc == pytest.approx(50.0)


def test_open_trade_uses_atr_for_stops(monkeypatch):
    tm = create_tm()
    tm.risk_per_trade = 1.0
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)
    price = 10.0
    tm.open_trade('ABC', price)
    pos = tm.positions['ABC']
    assert pos['atr'] == pytest.approx(1.0)
    assert pos['stop_loss'] == pytest.approx(price - tm.atr_mult_sl * 1.0)
    assert pos['take_profit'] == pytest.approx(price + tm.atr_mult_tp * 1.0)


def test_close_trade_records_rotation_price(monkeypatch):
    tm = create_tm()
    tm.risk_per_trade = 1.0
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)
    tm.open_trade('ABC', 10.0)
    tm.close_trade('ABC', 12.0, reason='Rotated to better candidate')
    record = tm.trade_history[-1]
    assert record['rotation_exit_price'] == pytest.approx(12.0)


def test_open_trade_respects_risk_limits(monkeypatch):
    tm = create_tm()
    monkeypatch.setattr(tm, "can_trade", lambda: False)

    def fail_fetch(*a, **k):
        raise AssertionError("fetch_ohlcv_smart should not be called")

    monkeypatch.setattr("data_fetcher.fetch_ohlcv_smart", fail_fetch)
    tm.open_trade("ABC", 10.0)
    assert "ABC" not in tm.positions
    assert tm.balance == pytest.approx(1000)


def test_close_trade_handles_missing_position(capfd):
    tm = create_tm()
    tm.close_trade("XYZ", 5.0)
    out = capfd.readouterr().out
    assert "No open position" in out
