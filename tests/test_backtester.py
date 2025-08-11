import pandas as pd
import pytest

from backtester import compute_metrics


def test_compute_metrics_cagr_and_sharpe():
    returns = pd.Series([0.1, -0.05, 0.2])
    equity_curve = pd.Series([1.1, 1.045, 1.254])
    timestamps = pd.Series(pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01']))
    metrics = compute_metrics(returns, equity_curve, timestamps)
    assert metrics['Total Return'] == pytest.approx(0.254, rel=1e-3)
    assert metrics['CAGR'] == pytest.approx(0.119648, rel=1e-3)
    assert metrics['Sharpe'] == pytest.approx(0.661813, rel=1e-3)
