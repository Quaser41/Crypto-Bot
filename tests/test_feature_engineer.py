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
        'Low': np.linspace(99, 169, periods)
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

    with caplog.at_level('WARNING'):
        result = add_indicators(df)

    assert result.empty
    assert any('Skipping symbol' in r.message for r in caplog.records)
