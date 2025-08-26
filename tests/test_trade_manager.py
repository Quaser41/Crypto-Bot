import pandas as pd
import pytest
import numpy as np
import logging

from trade_manager import TradeManager
from config import (
    ATR_MULT_SL,
    MIN_PROFIT_FEE_RATIO,
    HOLDING_PERIOD_SECONDS,
    ALLOCATION_MAX_DD,
    ALLOCATION_MIN_FACTOR,
    STAGNATION_THRESHOLD_PCT,
    STAGNATION_DURATION_SEC,
)


def create_tm():
    tm = TradeManager(starting_balance=1000, hold_period_sec=0, min_hold_bucket="<1m")
    tm.risk_per_trade = 0.1
    tm.min_trade_usd = 0
    tm.slippage_pct = 0.0
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


def test_allocation_scales_with_drawdown():
    tm = create_tm()

    # At new equity highs, factor should be 1.0
    alloc_high = tm.calculate_allocation(confidence=1.0)
    assert alloc_high == pytest.approx(100.0)

    # Simulate drawdown below max threshold
    dd_mid = ALLOCATION_MAX_DD / 2
    tm.balance = tm.peak_equity * (1 - dd_mid)
    tm._update_equity_metrics()
    alloc_dd_mid = tm.calculate_allocation(confidence=1.0)
    expected_factor_mid = max(
        ALLOCATION_MIN_FACTOR,
        1 - (dd_mid / ALLOCATION_MAX_DD) * (1 - ALLOCATION_MIN_FACTOR),
    )
    expected_dd_mid = tm.balance * tm.risk_per_trade * expected_factor_mid
    assert alloc_dd_mid == pytest.approx(expected_dd_mid)

    # Simulate drawdown exceeding max threshold
    dd_exceed = ALLOCATION_MAX_DD * 1.5
    dd_exceed = min(dd_exceed, 0.99)
    tm.balance = tm.peak_equity * (1 - dd_exceed)
    tm._update_equity_metrics()
    alloc_dd_exceed = tm.calculate_allocation(confidence=1.0)
    expected_dd_exceed = tm.balance * tm.risk_per_trade * ALLOCATION_MIN_FACTOR
    assert alloc_dd_exceed == pytest.approx(expected_dd_exceed)


@pytest.mark.parametrize(
    "pnl_history, expected_factor",
    [
        ([], 1.0),
        ([-10], 1.0),
        ([-10, -5], 0.8),
        ([-10, -5, -2], 0.6),
        ([-10, -5, -2, -1], 0.6),
    ],
)
def test_allocation_scales_with_loss_streak(pnl_history, expected_factor):
    tm = create_tm()
    tm.closed_pnl_history = pnl_history
    alloc = tm.calculate_allocation(confidence=1.0)
    expected = tm.balance * tm.risk_per_trade * expected_factor
    assert alloc == pytest.approx(expected)


def test_open_trade_uses_atr_for_stops(monkeypatch):
    tm = create_tm()
    tm.risk_per_trade = 1.0
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)
    price = 10.0
    tm.open_trade('ABC', price, confidence=1.0)
    pos = tm.positions['ABC']
    assert tm.atr_mult_sl == ATR_MULT_SL
    assert pos['atr'] == pytest.approx(1.0)
    assert pos['stop_loss'] == pytest.approx(price - ATR_MULT_SL * 1.0)
    assert pos['take_profit'] == pytest.approx(price + tm.atr_mult_tp * 1.0)


def test_close_trade_records_rotation_price(monkeypatch):
    tm = create_tm()
    tm.risk_per_trade = 1.0
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)
    tm.open_trade('ABC', 10.0, confidence=1.0)
    tm.close_trade(
        'ABC',
        12.0,
        reason='Rotated to better candidate',
        candidate={'symbol': 'XYZ', 'price': 5.0, 'confidence': 1.0},
    )
    record = tm.trade_history[-1]
    assert record['rotation_exit_price'] == pytest.approx(12.0)
    assert record['rotation_projected_gain'] > 0
    assert record['rotation_cost'] == pytest.approx(0.0)
    assert record['rotation_net_gain'] == pytest.approx(record['rotation_projected_gain'])


def test_rotation_aborted_when_gain_insufficient(monkeypatch):
    tm = create_tm()
    tm.risk_per_trade = 1.0
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)
    tm.open_trade('ABC', 10.0, confidence=1.0)
    closed = tm.close_trade(
        'ABC',
        9.5,
        reason='Rotated to better candidate',
        candidate={'symbol': 'XYZ', 'price': 5.0, 'confidence': 0.1},
    )
    assert closed is False
    assert 'ABC' in tm.positions
    assert len(tm.trade_history) == 0


def test_rotation_executes_when_gain_covers_cost(monkeypatch):
    tm = create_tm()
    tm.risk_per_trade = 1.0
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)
    tm.open_trade('ABC', 10.0, confidence=1.0)
    closed = tm.close_trade(
        'ABC',
        9.0,
        reason='Rotated to better candidate',
        candidate={'symbol': 'XYZ', 'price': 5.0, 'confidence': 2.0},
    )
    assert closed is True
    assert 'ABC' not in tm.positions
    record = tm.trade_history[-1]
    assert record['rotation_net_gain'] == pytest.approx(80.0)


def test_slippage_applied_to_trade(monkeypatch):
    tm = create_tm()
    tm.risk_per_trade = 1.0
    tm.slippage_pct = 0.01
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)
    price = 10.0
    tm.open_trade('ABC', price, confidence=1.0)
    pos = tm.positions['ABC']
    assert pos['entry_price'] == pytest.approx(price * 1.01)
    tm.close_trade('ABC', 12.0)
    record = tm.trade_history[-1]
    assert record['exit_price'] == pytest.approx(12.0 * (1 - tm.slippage_pct))
    expected_pnl = (record['exit_price'] - pos['entry_price']) * pos['qty']
    assert record['pnl'] == pytest.approx(expected_pnl)


def test_hold_period_delays_exits(monkeypatch):
    tm = TradeManager(starting_balance=1000, hold_period_sec=HOLDING_PERIOD_SECONDS, min_hold_bucket="<1m")
    tm.risk_per_trade = 1.0
    tm.slippage_pct = 0.0
    tm.trade_fee_pct = 0.0
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)

    tm.open_trade('ABC', 10.0, confidence=1.0)
    pos = tm.positions['ABC']

    # Price hits stop-loss immediately but hold period prevents closing
    tm.manage('ABC', pos['stop_loss'] - 0.01)
    assert tm.has_position('ABC')

    # After hold period elapsed, manage should close on the same price
    tm.positions['ABC']['entry_time'] -= HOLDING_PERIOD_SECONDS + 1
    tm.manage('ABC', pos['stop_loss'] - 0.01)
    assert not tm.has_position('ABC')


def test_close_trade_respects_hold_bucket(monkeypatch):
    tm = TradeManager(
        starting_balance=1000,
        hold_period_sec=0,
        min_hold_bucket="30m-2h",
        early_exit_fee_mult=2,
        min_profit_fee_ratio=0,
    )
    tm.risk_per_trade = 1.0
    tm.slippage_pct = 0.0
    tm.trade_fee_pct = 0.01
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)

    tm.open_trade('ABC', 10.0, confidence=1.0)
    pos = tm.positions['ABC']
    pos['entry_time'] -= 600  # 10 minutes < 30m
    closed = tm.close_trade('ABC', 10.1, reason='Take-Profit')
    assert closed is False
    assert tm.has_position('ABC')

    closed = tm.close_trade('ABC', 12.0, reason='Take-Profit')
    assert closed is True
    assert not tm.has_position('ABC')


def test_stagnation_closes_position(monkeypatch):
    tm = TradeManager(
        starting_balance=1000,
        hold_period_sec=0,
        stagnation_threshold_pct=STAGNATION_THRESHOLD_PCT,
        stagnation_duration_sec=STAGNATION_DURATION_SEC,
        min_hold_bucket="<1m",
    )
    tm.risk_per_trade = 1.0
    tm.slippage_pct = 0.0
    tm.trade_fee_pct = 0.0

    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)

    tm.open_trade('ABC', 100.0, confidence=1.0)
    pos = tm.positions['ABC']
    assert pos['entry_price'] == pytest.approx(100.0)
    assert pos['entry_time'] > 0

    pos['entry_time'] -= STAGNATION_DURATION_SEC + 1
    pos['last_movement_time'] -= STAGNATION_DURATION_SEC + 1

    tm.manage('ABC', pos['entry_price'] * (1 + STAGNATION_THRESHOLD_PCT / 2))
    assert 'ABC' not in tm.positions
    assert tm.trade_history[-1]['reason'] == 'Stagnant Price'


def test_skips_trade_when_profit_insufficient(monkeypatch):
    tm = create_tm()
    tm.risk_per_trade = 1.0
    tm.trade_fee_pct = 0.01
    tm.min_profit_fee_ratio = MIN_PROFIT_FEE_RATIO

    df = pd.DataFrame({'ATR': [0.01]})
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)

    tm.open_trade('ABC', 10.0, confidence=1.0)
    assert 'ABC' not in tm.positions


def test_open_trade_skips_on_insufficient_margin(caplog):
    tm = create_tm()
    tm.risk_per_trade = 2.0  # forces allocation greater than balance
    with caplog.at_level(logging.WARNING):
        tm.open_trade('ABC', 10.0, confidence=1.0)
    assert 'ABC' not in tm.positions
    assert any('insufficient margin' in msg.lower() for msg in caplog.messages)


def test_open_trade_respects_confidence_threshold(monkeypatch):
    tm = create_tm()
    tm.risk_per_trade = 1.0
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)

    tm.open_trade('ABC', 10.0, confidence=0.6)
    assert 'ABC' not in tm.positions

    tm.open_trade('ABC', 10.0, confidence=0.8)
    assert 'ABC' in tm.positions


def test_open_trade_enforces_hold_period(monkeypatch):
    tm = TradeManager(starting_balance=1000, hold_period_sec=HOLDING_PERIOD_SECONDS, min_hold_bucket="<1m")
    tm.risk_per_trade = 0.5
    tm.min_trade_usd = 0
    tm.slippage_pct = 0.0
    tm.trade_fee_pct = 0.0
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)

    tm.open_trade('ABC', 10.0, confidence=1.0)
    tm.open_trade('DEF', 5.0, confidence=1.0)
    assert 'DEF' not in tm.positions

    tm.last_trade_time -= HOLDING_PERIOD_SECONDS + 1
    tm.open_trade('DEF', 5.0, confidence=1.0)
    assert 'DEF' in tm.positions


def test_compute_trail_offset_uses_volatility():
    tm = create_tm()
    tm.trail_pct = 0.02
    pos = {"recent_prices": [100, 102, 98, 101]}
    offset = tm._compute_trail_offset(pos, 100)
    # std of recent_prices ≈ 1.479 -> volatility pct ≈ 0.01479
    expected = 100 * 0.02 * (1 + np.std(pos["recent_prices"]) / 100)
    assert offset == pytest.approx(expected)


def test_trailing_stop_respects_profit_threshold(monkeypatch):
    tm = TradeManager(starting_balance=1000, trade_fee_pct=0.01, trail_pct=0.03)
    tm.positions['ABC'] = {
        'coin_id': 'abc',
        'entry_price': 100.0,
        'qty': 1.0,
        'stop_loss': 90.0,
        'take_profit': 110.0,
        'entry_fee': 1.0,
        'highest_price': 100.0,
        'confidence': 1.0,
        'label': None,
        'side': 'BUY',
        'entry_time': 0.0,
        'last_movement_time': 0.0,
        'atr': None,
    }

    prices = [104.0, 105.0]

    def mock_price(symbol, coin_id=None):
        return prices.pop(0)

    monkeypatch.setattr('trade_manager.fetch_live_price', mock_price)

    tm.monitor_open_trades(single_run=True)
    assert 'trail_triggered' not in tm.positions['ABC']

    tm.monitor_open_trades(single_run=True)
    assert tm.positions['ABC'].get('trail_triggered') is True
    assert 'trail_price' in tm.positions['ABC']



def test_losing_position_does_not_trigger_trailing_stop(monkeypatch):
    tm = TradeManager(starting_balance=1000, trade_fee_pct=0.02, trail_pct=0.03, trail_profit_fee_ratio=0.0)
    tm.positions['ABC'] = {
        'coin_id': 'abc',
        'entry_price': 100.0,
        'qty': 1.0,
        'stop_loss': 90.0,
        'take_profit': 110.0,
        'entry_fee': 2.0,
        'highest_price': 100.0,
        'confidence': 1.0,
        'label': None,
        'side': 'BUY',
        'entry_time': 0.0,
        'last_movement_time': 0.0,
        'atr': None,
    }

    prices = [103.0]

    def mock_price(symbol, coin_id=None):
        return prices.pop(0)

    monkeypatch.setattr('trade_manager.fetch_live_price', mock_price)

    tm.monitor_open_trades(single_run=True)
    pos = tm.positions['ABC']
    assert 'trail_triggered' not in pos
    assert 'trail_price' not in pos
    assert pos['stop_loss'] == pytest.approx(90.0)



def test_close_trade_skips_when_profit_ratio_low(monkeypatch):
    tm = TradeManager(starting_balance=1000, trade_fee_pct=0.01,
                      min_profit_fee_ratio=MIN_PROFIT_FEE_RATIO, hold_period_sec=0)
    tm.positions['ABC'] = {
        'coin_id': 'abc',
        'entry_price': 100.0,
        'qty': 1.0,
        'stop_loss': 90.0,
        'take_profit': 110.0,
        'entry_fee': 1.0,
        'highest_price': 100.0,
        'confidence': 1.0,
        'label': None,
        'side': 'BUY',
        'entry_time': 0.0,
        'last_movement_time': 0.0,
        'atr': None,
    }
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: mock_indicator_df())
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)

    tm.close_trade('ABC', 100.5, reason='Take-Profit')
    assert tm.has_position('ABC')

    tm.close_trade('ABC', 120.0, reason='Take-Profit')
    assert not tm.has_position('ABC')


def test_state_persists_trade_fee_pct(tmp_path):
    tm = TradeManager(starting_balance=1000, trade_fee_pct=0.01)
    tm.STATE_FILE = str(tmp_path / "state.json")
    tm.save_state()

    tm_loaded = TradeManager(starting_balance=1000, trade_fee_pct=0.0)
    tm_loaded.STATE_FILE = tm.STATE_FILE
    tm_loaded.load_state()
    assert tm_loaded.trade_fee_pct == pytest.approx(0.01)


def test_state_persists_closed_pnl_history(tmp_path):
    tm = TradeManager(starting_balance=1000)
    tm.closed_pnl_history = [5.0, -3.0]
    tm.STATE_FILE = str(tmp_path / "state.json")
    tm.save_state()

    tm_loaded = TradeManager(starting_balance=1000)
    tm_loaded.STATE_FILE = tm.STATE_FILE
    tm_loaded.load_state()
    assert tm_loaded.closed_pnl_history == [5.0, -3.0]


def test_blacklist_skips_trade(monkeypatch):
    tm = TradeManager(starting_balance=1000, hold_period_sec=HOLDING_PERIOD_SECONDS, min_hold_bucket="5-30m")
    tm.risk_per_trade = 1.0
    tm.min_trade_usd = 0
    tm.slippage_pct = 0.0
    tm.trade_fee_pct = 0.0

    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)

    # INJ with bucket 5-30m is blacklisted in analytics/trade_stats.csv
    tm.open_trade('INJ', 10.0, confidence=1.0)
    assert 'INJ' not in tm.positions

    # BTC is not blacklisted
    tm.open_trade('BTC', 10.0, confidence=1.0)
    assert 'BTC' in tm.positions


def test_fee_ratio_blacklist(monkeypatch):
    tm = TradeManager(starting_balance=1000, hold_period_sec=10, min_hold_bucket="30m-2h")
    tm.risk_per_trade = 1.0
    tm.min_trade_usd = 0
    tm.slippage_pct = 0.0
    tm.trade_fee_pct = 0.0

    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d: d)

    # LINK in the 30m-2h bucket is blacklisted due to historical fee ratio
    tm.open_trade('LINK', 10.0, confidence=1.0)
    assert 'LINK' not in tm.positions
