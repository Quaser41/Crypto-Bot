import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import train_real_model


def _make_df(returns, features_first=None):
    prices = [100, 100, 100]
    for i, r in enumerate(returns):
        prices.append(prices[i] * (1 + r))
    if features_first is None:
        features_first = list(range(len(returns)))
    features = features_first + [1000, 1001, 1002]
    df = pd.DataFrame({"Close": prices, "feat": features})
    return df


def test_prepare_training_data_augment(monkeypatch):
    returns = (
        [-0.05] * 10
        + [-0.02] * 10
        + [0.01] * 10
        + [0.02] * 10
        + [0.05] * 30
    )
    df = _make_df(returns)
    monkeypatch.setattr(train_real_model, "fetch_ohlcv_smart", lambda *a, **k: df)
    monkeypatch.setattr(train_real_model, "add_indicators", lambda d: d)
    monkeypatch.setattr(train_real_model, "load_feature_list", lambda: ["feat"])
    monkeypatch.setattr(
        train_real_model.data_fetcher,
        "fetch_binance_us_ohlcv",
        lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(
        train_real_model.data_fetcher,
        "fetch_dexscreener_ohlcv",
        lambda *a, **k: pd.DataFrame(),
    )

    X, y = train_real_model.prepare_training_data("SYM", "coin", min_unique_samples=3)
    assert X is not None and y is not None
    counts = y.value_counts().to_dict()
    assert counts[0] == 20
    assert counts[4] == 50


def test_prepare_training_data_drops_on_few_unique(monkeypatch, caplog):
    returns = (
        [-0.05] * 10
        + [-0.02] * 10
        + [0.01] * 10
        + [0.02] * 10
        + [0.05] * 10
    )
    features_first = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1] + list(range(10, 50))
    df = _make_df(returns, features_first)
    monkeypatch.setattr(train_real_model, "fetch_ohlcv_smart", lambda *a, **k: df)
    monkeypatch.setattr(train_real_model, "add_indicators", lambda d: d)
    monkeypatch.setattr(train_real_model, "load_feature_list", lambda: ["feat"])

    with caplog.at_level("WARNING", logger=train_real_model.logger.name):
        X, y = train_real_model.prepare_training_data("SYM", "coin", min_unique_samples=6)
    assert X is None and y is None
    assert any("unique rows" in r.getMessage() for r in caplog.records)


def test_prepare_training_data_passes_when_threshold_lowered(monkeypatch):
    returns = (
        [-0.05] * 10
        + [-0.02] * 10
        + [0.01] * 10
        + [0.02] * 10
        + [0.05] * 10
    )
    features_first = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1] + list(range(10, 50))
    df = _make_df(returns, features_first)
    monkeypatch.setattr(train_real_model, "fetch_ohlcv_smart", lambda *a, **k: df)
    monkeypatch.setattr(train_real_model, "add_indicators", lambda d: d)
    monkeypatch.setattr(train_real_model, "load_feature_list", lambda: ["feat"])

    X, y = train_real_model.prepare_training_data("SYM", "coin", min_unique_samples=3)
    assert X is not None and y is not None
