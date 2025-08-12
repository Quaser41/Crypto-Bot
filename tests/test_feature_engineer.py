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
