import numpy as np
import pandas as pd

from feature_engineer import add_indicators


def test_add_indicators_merges_sentiment_and_onchain(monkeypatch):
    periods = 70
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')
    df = pd.DataFrame({
        'Timestamp': dates,
        'Close': np.linspace(100, 170, periods),
        'High': np.linspace(101, 171, periods),
        'Low': np.linspace(99, 169, periods),
        'Volume': np.linspace(1000, 2000, periods)
    })

    sentiment = pd.DataFrame({
        'Timestamp': dates,
        'FearGreed': np.linspace(20, 80, periods)
    })
    onchain = pd.DataFrame({
        'Timestamp': dates,
        'TxVolume': np.linspace(1000, 2000, periods),
        'ActiveAddresses': np.linspace(100, 200, periods)
    })

    monkeypatch.setattr('feature_engineer.fetch_fear_greed_index', lambda limit=365: sentiment)
    monkeypatch.setattr('feature_engineer.fetch_onchain_metrics', lambda: onchain)
    btc = pd.DataFrame({'Timestamp': dates, 'Close': np.linspace(20000, 20100, periods)})
    monkeypatch.setattr('feature_engineer.fetch_ohlcv_smart', lambda *args, **kwargs: btc)

    result = add_indicators(df, min_rows=20)
    assert 'Momentum_Tier' in result.columns
    assert result['Momentum_Tier'].iloc[-1] in {'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'}
    assert 'FearGreed_norm' in result.columns
    assert result['FearGreed_norm'].notna().all()
    assert 'TxVolume_norm' in result.columns
    assert 'ActiveAddresses_norm' in result.columns
    assert 'MACD_4h' in result.columns
    assert 'SMA_4h' in result.columns
    assert result['MACD_4h'].notna().all()
    assert result['SMA_4h'].notna().all()
    for col in ['BB_Upper', 'BB_Middle', 'BB_Lower', 'EMA_9', 'EMA_26', 'OBV', 'Volume_vs_SMA20', 'RelStrength_BTC']:
        assert col in result.columns
        assert result[col].notna().all()


def test_add_indicators_handles_all_nan_onchain(monkeypatch):
    periods = 120
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')
    df = pd.DataFrame({
        'Timestamp': dates,
        'Close': np.linspace(100, 220, periods),
        'High': np.linspace(101, 221, periods),
        'Low': np.linspace(99, 219, periods)
    })

    sentiment = pd.DataFrame({
        'Timestamp': dates,
        'FearGreed': np.linspace(20, 80, periods)
    })
    onchain = pd.DataFrame({
        'Timestamp': dates,
        'UnknownMetric': [np.nan] * periods
    })

    monkeypatch.setattr('feature_engineer.fetch_fear_greed_index', lambda limit=365: sentiment)
    monkeypatch.setattr('feature_engineer.fetch_onchain_metrics', lambda: onchain)
    monkeypatch.setattr('feature_engineer.fetch_ohlcv_smart', lambda *args, **kwargs: pd.DataFrame())

    result = add_indicators(df)
    assert len(result) >= 60


def test_add_indicators_insufficient_4h_history(monkeypatch):
    periods = 50
    dates = pd.date_range('2023-01-01', periods=periods, freq='H')
    df = pd.DataFrame({
        'Timestamp': dates,
        'Close': np.linspace(100, 150, periods),
        'High': np.linspace(101, 151, periods),
        'Low': np.linspace(99, 149, periods)
    })

    monkeypatch.setattr('feature_engineer.fetch_fear_greed_index', lambda limit=365: pd.DataFrame())
    monkeypatch.setattr('feature_engineer.fetch_onchain_metrics', lambda: pd.DataFrame())
    monkeypatch.setattr('feature_engineer.fetch_ohlcv_smart', lambda *args, **kwargs: pd.DataFrame())

    result = add_indicators(df, min_rows=20)
    cols = {'SMA_4h', 'MACD_4h', 'Signal_4h', 'Hist_4h'}
    assert result.empty
    assert cols.issubset(result.columns)
    assert result[list(cols)].isna().all().all()


def test_add_indicators_skips_when_insufficient_rows(monkeypatch, caplog):
    periods = 55  # less than default min_rows=60 but enough for indicator windows
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')
    df = pd.DataFrame({
        'Timestamp': dates,
        'Close': np.linspace(100, 155, periods),
        'High': np.linspace(101, 156, periods),
        'Low': np.linspace(99, 154, periods),
    })

    def fail_fetch(*args, **kwargs):
        raise AssertionError('fetch should not be called')

    monkeypatch.setattr('feature_engineer.fetch_fear_greed_index', fail_fetch)
    monkeypatch.setattr('feature_engineer.fetch_onchain_metrics', fail_fetch)
    monkeypatch.setattr('feature_engineer.fetch_ohlcv_smart', fail_fetch)

    with caplog.at_level('WARNING'):
        result = add_indicators(df)

    assert result.empty
    assert any('Skipping symbol' in r.message for r in caplog.records)


def test_add_indicators_skips_constant_price(monkeypatch, caplog):
    periods = 70
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')
    df = pd.DataFrame({
        'Timestamp': dates,
        'Close': [100] * periods,
        'High': [101] * periods,
        'Low': [99] * periods,
    })

    def fail_fetch(*args, **kwargs):
        raise AssertionError('fetch should not be called')

    monkeypatch.setattr('feature_engineer.fetch_fear_greed_index', fail_fetch)
    monkeypatch.setattr('feature_engineer.fetch_onchain_metrics', fail_fetch)
    monkeypatch.setattr('feature_engineer.fetch_ohlcv_smart', fail_fetch)

    with caplog.at_level('WARNING'):
        result = add_indicators(df)

    assert result.empty
    assert any('Volatility_7d is zero for all points' in r.message for r in caplog.records)


def test_add_indicators_no_settingwithcopy_warning(monkeypatch):
    periods = 70
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')
    closes = np.concatenate([np.full(10, 100), np.linspace(101, 170, periods - 10)])
    df = pd.DataFrame({
        'Timestamp': dates,
        'Close': closes,
        'High': closes + 1,
        'Low': closes - 1,
    })

    sentiment = pd.DataFrame({
        'Timestamp': dates,
        'FearGreed': np.linspace(20, 80, periods)
    })
    onchain = pd.DataFrame({
        'Timestamp': dates,
        'TxVolume': np.linspace(1000, 2000, periods),
        'ActiveAddresses': np.linspace(100, 200, periods)
    })

    monkeypatch.setattr('feature_engineer.fetch_fear_greed_index', lambda limit=365: sentiment)
    monkeypatch.setattr('feature_engineer.fetch_onchain_metrics', lambda: onchain)
    btc = pd.DataFrame({'Timestamp': dates, 'Close': np.linspace(20000, 20100, periods)})
    monkeypatch.setattr('feature_engineer.fetch_ohlcv_smart', lambda *args, **kwargs: btc)

    from pandas.errors import SettingWithCopyWarning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('error', SettingWithCopyWarning)
        result = add_indicators(df, min_rows=20)

    assert not result.empty
