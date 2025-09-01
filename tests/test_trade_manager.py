import pandas as pd
import pytest
import numpy as np
import logging
import time

from trade_manager import TradeManager
from config import (
    ATR_MULT_SL,
    MIN_PROFIT_FEE_RATIO,
    HOLDING_PERIOD_SECONDS,
    ALLOCATION_MAX_DD,
    ALLOCATION_MIN_FACTOR,
    STAGNATION_THRESHOLD_PCT,
    STAGNATION_DURATION_SEC,
    PERF_MIN_TRADE_COUNT,
)
import analytics.performance as perf


def create_tm(include_unrealized=False):
    tm = TradeManager(starting_balance=1000, hold_period_sec=0, min_hold_bucket="<1m", include_unrealized_pnl=include_unrealized)
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


def test_env_var_overrides_drawdown(monkeypatch):
    import importlib
    import config as config_module
    import trade_manager as tm_module

    monkeypatch.setenv("MAX_DRAWDOWN_PCT", "0.1")
    monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "0.02")

    importlib.reload(config_module)
    importlib.reload(tm_module)

    tm = tm_module.TradeManager(starting_balance=1000, hold_period_sec=0, min_hold_bucket="<1m")
    assert tm.max_drawdown_pct == pytest.approx(0.1)
    assert tm.max_daily_loss_pct == pytest.approx(0.02)

    monkeypatch.delenv("MAX_DRAWDOWN_PCT", raising=False)
    monkeypatch.delenv("MAX_DAILY_LOSS_PCT", raising=False)
    importlib.reload(config_module)
    importlib.reload(tm_module)


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


def test_unrealized_drawdown_reduces_allocation(monkeypatch):
    tm = create_tm(include_unrealized=True)
    tm.risk_per_trade = 0.1
    tm.min_trade_usd = 0
    tm.slippage_pct = 0.0
    tm.trade_fee_pct = 0.0
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

    tm.open_trade('ABC', 10.0, confidence=1.0)

    # Price drops 50%, creating unrealized drawdown
    monkeypatch.setattr('trade_manager.fetch_live_price', lambda *a, **k: 5.0)
    alloc = tm.calculate_allocation(confidence=1.0)

    expected = tm.balance * tm.risk_per_trade * ALLOCATION_MIN_FACTOR
    assert alloc == pytest.approx(expected)


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
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)
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
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)
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
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)
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
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)
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


def test_rotation_aborted_when_net_gain_below_margin(monkeypatch):
    tm = create_tm()
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)
    tm.open_trade('ABC', 10.0, confidence=1.0)

    # Force projected gain to barely exceed cost but not the safety margin
    monkeypatch.setattr(tm, '_estimate_rotation_gain', lambda *a, **k: 0.52)
    closed = tm.close_trade(
        'ABC',
        9.95,
        reason='Rotated to better candidate',
        candidate={'symbol': 'XYZ', 'price': 5.0, 'confidence': 1.0},
    )
    assert closed is False
    assert 'ABC' in tm.positions
    assert len(tm.trade_history) == 0


def test_rotation_outcome_logging(monkeypatch, caplog):
    tm = create_tm()
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

    tm.open_trade('ABC', 10.0, confidence=1.0)
    tm.close_trade(
        'ABC',
        12.0,
        reason='Rotated to better candidate',
        candidate={'symbol': 'XYZ', 'price': 5.0, 'confidence': 1.0},
    )
    tm.open_trade('XYZ', 5.0, confidence=1.0)
    with caplog.at_level(logging.INFO, logger='trade_manager'):
        tm.close_trade('XYZ', 6.0, reason='Take-Profit')
    assert 'Rotation outcome for XYZ' in caplog.text


def test_rotation_gain_scaled_by_fee_ratio(caplog):
    tm = create_tm()
    tm.trade_fee_pct = 0.05
    tm.min_profit_fee_ratio = 1.0
    tm.min_hold_bucket = "30m-2h"
    perf.reset_cache()
    with caplog.at_level(logging.INFO, logger="trade_manager"):
        gain = tm._estimate_rotation_gain("LINK", 10.0, confidence=1.0)
    assert gain == 0.0
    fee_ratio = perf.get_avg_fee_ratio("LINK", "30m-2h", refresh_seconds=0)
    expected_threshold = tm.min_profit_fee_ratio * (1 + fee_ratio)
    assert f"{expected_threshold:.2f}" in caplog.text


def test_slippage_applied_to_trade(monkeypatch):
    tm = create_tm()
    tm.risk_per_trade = 1.0
    tm.slippage_pct = 0.01
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)
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
    tm.sl_buffer_atr_mult = 0.0
    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

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
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

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
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

    tm.open_trade('ABC', 100.0, confidence=1.0)
    pos = tm.positions['ABC']
    assert pos['entry_price'] == pytest.approx(100.0)
    assert pos['entry_time'] > 0

    pos['entry_time'] -= STAGNATION_DURATION_SEC + 1
    pos['last_movement_time'] -= STAGNATION_DURATION_SEC + 1

    tm.manage('ABC', pos['entry_price'] * (1 + STAGNATION_THRESHOLD_PCT / 2))
    assert 'ABC' not in tm.positions
    assert tm.trade_history[-1]['reason'] == 'Stagnant Price'


def test_trailing_stop_volatility_multiplier():
    tm = TradeManager(trail_pct=0.05, trail_atr_mult=None, trail_vol_mult=0)
    pos = {
        'entry_price': 100.0,
        'qty': 1,
        'side': 'BUY',
        'recent_prices': [100, 120, 80, 110, 90]
    }
    base = tm._compute_trail_offset(pos, 105.0)
    tm.trail_vol_mult = 2.0
    widened = tm._compute_trail_offset(pos, 105.0)
    assert widened > base


def test_adaptive_stagnation_params():
    tm = TradeManager(
        stagnation_threshold_pct=0.01,
        stagnation_duration_sec=100,
        adaptive_stagnation=True,
        stagnation_vol_mult=2.0,
    )
    pos = {'recent_prices': [100, 120, 80, 110, 90]}
    thresh, dur = tm._compute_stagnation_params(pos, 105.0)
    assert thresh > tm.stagnation_threshold_pct
    assert dur > tm.stagnation_duration_sec


def test_trailing_stop_scales_with_atr_and_logging(monkeypatch, caplog):
    tm = TradeManager(trail_pct=0.05, trail_atr_mult=1.0, trail_vol_mult=2.0, trade_fee_pct=0.0)
    pos = {
        'coin_id': 'abc',
        'entry_price': 100.0,
        'qty': 1.0,
        'stop_loss': 90.0,
        'take_profit': 110.0,
        'entry_fee': 0.0,
        'highest_price': 100.0,
        'confidence': 1.0,
        'label': None,
        'side': 'BUY',
        'entry_time': 0.0,
        'last_movement_time': 0.0,
        'atr': 1.0,
        'recent_prices': [100, 120, 80, 110, 90],
    }
    tm.positions['ABC'] = pos
    prices = [105.0, 108.0]

    def mock_price(symbol, coin_id=None):
        return prices.pop(0)

    monkeypatch.setattr('trade_manager.fetch_live_price', mock_price)
    with caplog.at_level(logging.INFO, logger='trade_manager'):
        tm.monitor_open_trades(single_run=True)
        tm.monitor_open_trades(single_run=True)

    expected_prices = pos['recent_prices']
    vol_pct = np.std(expected_prices) / 108.0
    expected_offset = pos['atr'] * tm.trail_atr_mult * max(1.0, vol_pct * tm.trail_vol_mult)
    expected_trail = 108.0 - expected_offset
    assert pos['trail_price'] == pytest.approx(expected_trail)
    assert f"{expected_trail:.4f}" in caplog.text


def test_adaptive_stagnation_logs_scaled_threshold(monkeypatch, caplog):
    tm = TradeManager(
        stagnation_threshold_pct=0.01,
        stagnation_duration_sec=10,
        adaptive_stagnation=True,
        stagnation_vol_mult=2.0,
        trade_fee_pct=0.0,
        hold_period_sec=0,
        min_hold_bucket="<1m",
    )
    now = time.time()
    pos = {
        'entry_price': 100.0,
        'qty': 1.0,
        'side': 'BUY',
        'entry_time': now - 100,
        'last_movement_time': now - 100,
        'highest_price': 100.0,
        'stop_loss': 90.0,
        'take_profit': 110.0,
        'entry_fee': 0.0,
        'confidence': 1.0,
        'label': None,
        'recent_prices': [100, 120, 80, 110, 90],
    }
    tm.positions['ABC'] = pos
    current_price = 100.2
    vol_pct = np.std(pos['recent_prices']) / current_price
    mult = 1 + vol_pct * tm.stagnation_vol_mult
    expected_threshold = tm.stagnation_threshold_pct * mult
    expected_duration = tm.stagnation_duration_sec * mult
    pos['last_movement_time'] = now - expected_duration - 1

    dummy_df = pd.DataFrame({
        'Close': [100, 101, 102],
        'Return_1d': [0.01, 0.01, 0.01],
        'Return_3d': [0.02, 0.02, 0.02],
        'RSI': [50, 50, 50],
        'Hist': [0.1, 0.1, 0.1],
    })
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: dummy_df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

    with caplog.at_level(logging.INFO, logger='trade_manager'):
        tm.manage('ABC', current_price)

    assert 'ABC' not in tm.positions
    assert f"below {expected_threshold * 100:.2f}%" in caplog.text


def test_skips_trade_when_profit_insufficient(monkeypatch):
    tm = create_tm()
    tm.risk_per_trade = 1.0
    tm.trade_fee_pct = 0.01
    tm.min_profit_fee_ratio = MIN_PROFIT_FEE_RATIO

    df = pd.DataFrame({'ATR': [0.01]})
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

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
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

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
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

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
    offset = tm._compute_trail_offset(pos, 105)
    vol_pct = np.std(pos["recent_prices"]) / 105
    expected = 105 * max(0.02, vol_pct * tm.trail_vol_mult)
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
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

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
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

    # INJ with bucket 5-30m is blacklisted in analytics/trade_stats.csv
    tm.open_trade('INJ', 10.0, confidence=1.0)
    assert 'INJ' not in tm.positions

    # BTC is not blacklisted
    tm.open_trade('BTC', 10.0, confidence=1.0)
    assert 'BTC' in tm.positions


def test_fee_ratio_blacklist(monkeypatch):
    # Lower the trade count threshold so the LINK sample is eligible for blacklisting
    monkeypatch.setattr("config.PERF_MIN_TRADE_COUNT", 1)
    monkeypatch.setattr(perf, "PERF_MIN_TRADE_COUNT", max(1, PERF_MIN_TRADE_COUNT - 2))
    perf.reset_cache()

    tm = TradeManager(starting_balance=1000, hold_period_sec=10, min_hold_bucket="30m-2h")
    tm.risk_per_trade = 1.0
    tm.min_trade_usd = 0
    tm.slippage_pct = 0.0
    tm.trade_fee_pct = 0.0

    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

    # LINK in the 30m-2h bucket is blacklisted due to historical fee ratio
    tm.open_trade('LINK', 10.0, confidence=1.0)
    assert 'LINK' not in tm.positions


def test_symbol_pnl_threshold_skips_symbol(monkeypatch, caplog):
    tm = TradeManager(
        starting_balance=1000,
        hold_period_sec=0,
        min_hold_bucket="<1m",
        max_drawdown_pct=1.0,
        max_daily_loss_pct=1.0,
        symbol_pnl_threshold=-5.0,
    )
    tm.risk_per_trade = 1.0
    tm.slippage_pct = 0.0
    tm.trade_fee_pct = 0.0

    df = mock_indicator_df()
    monkeypatch.setattr('data_fetcher.fetch_ohlcv_smart', lambda *a, **k: df)
    monkeypatch.setattr('feature_engineer.add_indicators', lambda d, **k: d)

    tm.open_trade('ABC', 10.0, confidence=1.0)
    tm.close_trade('ABC', 5.0, reason='Stop-Loss')
    assert tm.symbol_pnl['ABC'] < 0

    caplog.clear()
    with caplog.at_level(logging.INFO, logger='trade_manager'):
        tm.open_trade('ABC', 10.0, confidence=1.0)
    assert 'cumulative PnL' in caplog.text
    assert 'ABC' not in tm.positions
