import pandas as pd
import pytest

import backtester


def test_compute_metrics_cagr_and_sharpe():
    returns = pd.Series([0.1, -0.05, 0.2])
    equity_curve = pd.Series([1.1, 1.045, 1.254])
    timestamps = pd.Series(pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01']))
    metrics = backtester.compute_metrics(returns, equity_curve, timestamps)
    assert metrics['Total Return'] == pytest.approx(0.254, rel=1e-3)
    assert metrics['CAGR'] == pytest.approx(0.119648, rel=1e-3)
    assert metrics['Sharpe'] == pytest.approx(0.661813, rel=1e-3)


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
        return 'BUY', 0.60, 3

    monkeypatch.setattr(backtester, 'predict_signal', fake_predict_signal)
    signal, confidence, label = backtester.generate_signal(window)
    assert signal == 'BUY'


def test_backtest_symbol_generates_positive_return(monkeypatch):
    close_prices = pd.Series([100, 110, 120, 130, 140])
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2020-01-01', periods=5, freq='D'),
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

    monkeypatch.setattr(backtester, 'fetch_ohlcv_smart', lambda symbol, days, limit: df[['Timestamp', 'Close']])
    monkeypatch.setattr(backtester, 'add_indicators', lambda raw_df: df)
    monkeypatch.setattr(backtester, 'add_atr', lambda x, period=14: x)

    call_state = {'n': 0}

    def fake_predict_signal(window, threshold):
        call_state['n'] += 1
        if call_state['n'] == 1:
            return 'BUY', 0.60, 3
        return 'HOLD', 0.60, 1

    monkeypatch.setattr(backtester, 'predict_signal', fake_predict_signal)

    metrics = backtester.backtest_symbol('TEST', days=5, slippage_pct=0.0)
    assert metrics['Total Return'] > 0
