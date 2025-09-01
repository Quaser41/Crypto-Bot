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
    monkeypatch.setattr(train_real_model, "add_indicators", lambda d, **k: d)
    monkeypatch.setattr(train_real_model, "load_feature_list", lambda: ["feat"])

    X, y = train_real_model.prepare_training_data("SYM", "coin", min_unique_samples=3)
    assert X is not None and y is not None
    counts = y.value_counts().to_dict()
    assert counts[0] == 28
    assert counts[4] == 32


def test_prepare_training_data_widens_quantiles(monkeypatch):
    """Ensure quantile widening populates all classes when needed."""
    returns = [-0.04, -0.03, -0.02, 0.02, 0.03, 0.04] * 10
    df = _make_df(returns)
    monkeypatch.setattr(train_real_model, "fetch_ohlcv_smart", lambda *a, **k: df)
    monkeypatch.setattr(train_real_model, "add_indicators", lambda d, **k: d)
    monkeypatch.setattr(train_real_model, "load_feature_list", lambda: ["feat"])

    X, y = train_real_model.prepare_training_data("SYM", "coin", min_unique_samples=3)
    assert X is not None and y is not None
    counts = y.value_counts().to_dict()
    assert set(counts.keys()) == {0, 1, 2, 3, 4}
    assert all(v > 0 for v in counts.values())


def test_prepare_training_data_drops_on_few_unique(monkeypatch, caplog):
    returns = (
        [-0.05] * 10
        + [-0.02] * 10
        + [0.01] * 10
        + [0.02] * 10
        + [0.05] * 30
    )
    features_first = list(range(70))
    dup_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17]
    dup_vals = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
    for i, v in zip(dup_idx, dup_vals):
        features_first[i] = v
    df = _make_df(returns, features_first)
    monkeypatch.setattr(train_real_model, "fetch_ohlcv_smart", lambda *a, **k: df)
    monkeypatch.setattr(train_real_model, "add_indicators", lambda d, **k: d)
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
        + [0.05] * 30
    )
    features_first = list(range(70))
    dup_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17]
    dup_vals = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
    for i, v in zip(dup_idx, dup_vals):
        features_first[i] = v
    df = _make_df(returns, features_first)
    monkeypatch.setattr(train_real_model, "fetch_ohlcv_smart", lambda *a, **k: df)
    monkeypatch.setattr(train_real_model, "add_indicators", lambda d, **k: d)
    monkeypatch.setattr(train_real_model, "load_feature_list", lambda: ["feat"])

    X, y = train_real_model.prepare_training_data("SYM", "coin", min_unique_samples=3)
    assert X is not None and y is not None


def test_prepare_training_data_drops_when_insufficient_rows(monkeypatch, caplog):
    returns = (
        [-0.05] * 10
        + [-0.02] * 10
        + [0.01] * 10
        + [0.02] * 10
    )
    df = _make_df(returns)

    def fake_fetch(*a, **k):
        return df

    monkeypatch.setattr(train_real_model, "fetch_ohlcv_smart", fake_fetch)

    called = {"add": False}

    def fake_add(d, **k):
        called["add"] = True
        return pd.DataFrame()

    monkeypatch.setattr(train_real_model, "add_indicators", fake_add)
    monkeypatch.setattr(train_real_model, "load_feature_list", lambda: ["feat"])

    with caplog.at_level("WARNING", logger=train_real_model.logger.name):
        X, y = train_real_model.prepare_training_data("SYM", "coin", min_unique_samples=3)

    assert X is None and y is None
    assert called["add"]
    assert any("dropping symbol" in r.getMessage().lower() for r in caplog.records)


def test_prepare_training_data_short_history_kept(monkeypatch):
    returns = (
        [-0.05] * 10
        + [-0.02] * 10
        + [0.01] * 10
        + [0.02] * 10
        + [0.05] * 40
    )
    raw_df = _make_df(returns)

    monkeypatch.setattr(train_real_model, "fetch_ohlcv_smart", lambda *a, **k: raw_df)

    def fake_add(d, min_rows, **k):
        assert min_rows == int(min(60, len(d) * 0.6))
        return d.head(50)

    monkeypatch.setattr(train_real_model, "add_indicators", fake_add)
    monkeypatch.setattr(train_real_model, "load_feature_list", lambda: ["feat"])

    X, y = train_real_model.prepare_training_data("SYM", "coin", min_unique_samples=3)
    assert X is not None and y is not None
