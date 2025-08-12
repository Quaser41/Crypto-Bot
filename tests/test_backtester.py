import pandas as pd
import numpy as np
import pytest

import backtester
from utils.prediction_class import PredictionClass


def test_compute_metrics_cagr_and_sharpe():
    pattern = [0.02, -0.01, 0.015, -0.005, 0.01]
    returns = pd.Series(pattern * 20)
    equity_curve = (1 + returns).cumprod()
    timestamps = pd.Series(pd.date_range('2020-01-01', periods=len(returns), freq='D'))
    metrics = backtester.compute_metrics(returns, equity_curve, timestamps)
    assert metrics['Total Return'] == pytest.approx(equity_curve.iloc[-1] - 1, rel=1e-3)
    assert not np.isnan(metrics['CAGR'])
    assert not np.isnan(metrics['Sharpe'])


def test_compute_metrics_skips_short_duration():
    returns = pd.Series([0.1, -0.05])
    equity_curve = pd.Series([1.1, 1.045])
    timestamps = pd.Series(pd.to_datetime(['2020-01-01', '2020-01-02']))
    metrics = backtester.compute_metrics(returns, equity_curve, timestamps)
    assert np.isnan(metrics['CAGR'])
    assert not np.isnan(metrics['Sharpe'])


def test_compute_metrics_skips_low_frequency():
    returns = pd.Series([0.1, -0.05])
    equity_curve = pd.Series([1.1, 1.045])
    timestamps = pd.Series(pd.to_datetime(['2020-01-01', '2020-07-01']))
    metrics = backtester.compute_metrics(returns, equity_curve, timestamps)
    assert np.isnan(metrics['Sharpe'])
    assert not np.isnan(metrics['CAGR'])


def test_generate_signal_triggers_buy_when_confidence_exceeds_threshold(monkeypatch):
    window = pd.DataFrame({
        'Volatility_7d': [0.05],
        'Momentum_Tier': ['Tier 1'],
        'Return_1d': [0.02],
        'Return_3d': [0.06],
        'RSI': [60],
        'Close': [100],
    })

    def fake_predict_signal(df, threshold):
        assert pytest.approx(threshold, rel=1e-3) == 0.55
        return 'BUY', 0.60, PredictionClass.SMALL_GAIN.value

    monkeypatch.setattr(backtester, 'predict_signal', fake_predict_signal)
    signal, confidence, label = backtester.generate_signal(window)
    assert signal == 'BUY'


def test_backtest_symbol_generates_positive_return(monkeypatch):
    close_prices = pd.Series([100, 110, 120, 130, 140])
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2020-01-01', periods=5, freq='D'),
        'Open': close_prices,
        'Close': close_prices,
        'Return_1d': close_prices.pct_change().fillna(0),
        'Return_3d': close_prices.pct_change(3).fillna(0),
        'RSI': [60] * 5,
        'MACD': [1] * 5,
        'Signal': [0.5] * 5,
        'Hist': [0.5] * 5,
        'SMA_20': [95] * 5,
        'SMA_50': [90] * 5,
        'Volatility_7d': [0.05] * 5,
        'Momentum_Tier': ['Tier 1'] * 5,
    })

    monkeypatch.setattr(backtester, 'fetch_ohlcv_smart', lambda symbol, days, limit: df[['Timestamp', 'Open', 'Close']])
    monkeypatch.setattr(backtester, 'add_indicators', lambda raw_df: df)
    monkeypatch.setattr(backtester, 'add_atr', lambda x, period=14: x)

    call_state = {'n': 0}

    def fake_predict_signal(window, threshold):
        call_state['n'] += 1
        if call_state['n'] == 1:
            return 'BUY', 0.60, PredictionClass.SMALL_GAIN.value
        return 'HOLD', 0.60, PredictionClass.SMALL_LOSS.value

    monkeypatch.setattr(backtester, 'predict_signal', fake_predict_signal)

    metrics = backtester.backtest_symbol('TEST', days=5, slippage_pct=0.0, fee_pct=0.0, execution_delay_bars=0)
    assert metrics['Total Return'] > 0


def test_backtest_symbol_applies_fee(monkeypatch):
    close_prices = pd.Series([100, 110, 120, 130, 140])
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2020-01-01', periods=5, freq='D'),
        'Open': close_prices,
        'Close': close_prices,
        'Return_1d': close_prices.pct_change().fillna(0),
        'Return_3d': close_prices.pct_change(3).fillna(0),
        'RSI': [60] * 5,
        'MACD': [1] * 5,
        'Signal': [0.5] * 5,
        'Hist': [0.5] * 5,
        'SMA_20': [95] * 5,
        'SMA_50': [90] * 5,
        'Volatility_7d': [0.05] * 5,
        'Momentum_Tier': ['Tier 1'] * 5,
    })

    monkeypatch.setattr(backtester, 'fetch_ohlcv_smart', lambda symbol, days, limit: df[['Timestamp', 'Open', 'Close']])
    monkeypatch.setattr(backtester, 'add_indicators', lambda raw_df: df)
    monkeypatch.setattr(backtester, 'add_atr', lambda x, period=14: x)

    call_state = {'n': 0}

    def fake_predict_signal(window, threshold):
        call_state['n'] += 1
        if call_state['n'] == 1:
            return 'BUY', 0.60, PredictionClass.SMALL_GAIN.value
        if call_state['n'] == 3:
            return 'SELL', 0.90, PredictionClass.SMALL_LOSS.value
        return 'HOLD', 0.60, PredictionClass.SMALL_LOSS.value

    monkeypatch.setattr(backtester, 'predict_signal', fake_predict_signal)

    metrics = backtester.backtest_symbol('TEST', days=5, slippage_pct=0.0, fee_pct=0.01, execution_delay_bars=0)
    assert metrics['Fees Paid'] == pytest.approx(0.020725, rel=1e-3)


def test_backtest_symbol_respects_execution_delay(monkeypatch):
    open_prices = pd.Series([100, 101, 102, 103, 104])
    close_prices = pd.Series([101, 102, 103, 104, 105])
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2020-01-01', periods=5, freq='D'),
        'Open': open_prices,
        'Close': close_prices,
        'Return_1d': close_prices.pct_change().fillna(0),
        'Return_3d': close_prices.pct_change(3).fillna(0),
        'RSI': [60] * 5,
        'MACD': [1] * 5,
        'Signal': [0.5] * 5,
        'Hist': [0.5] * 5,
        'SMA_20': [95] * 5,
        'SMA_50': [90] * 5,
        'Volatility_7d': [0.05] * 5,
        'Momentum_Tier': ['Tier 1'] * 5,
    })

    monkeypatch.setattr(backtester, 'fetch_ohlcv_smart', lambda symbol, days, limit: df[['Timestamp', 'Open', 'Close']])
    monkeypatch.setattr(backtester, 'add_indicators', lambda raw_df: df)
    monkeypatch.setattr(backtester, 'add_atr', lambda x, period=14: x)

    call_state = {'n': 0}

    def fake_predict_signal(window, threshold):
        call_state['n'] += 1
        if call_state['n'] == 1:
            return 'BUY', 0.60, PredictionClass.SMALL_GAIN.value
        return 'HOLD', 0.60, PredictionClass.SMALL_LOSS.value

    monkeypatch.setattr(backtester, 'predict_signal', fake_predict_signal)

    no_delay = backtester.backtest_symbol('TEST', days=5, slippage_pct=0.0, fee_pct=0.0, execution_delay_bars=0)
    delay = backtester.backtest_symbol('TEST', days=5, slippage_pct=0.0, fee_pct=0.0, execution_delay_bars=1)
    assert delay['Total Return'] < no_delay['Total Return']
