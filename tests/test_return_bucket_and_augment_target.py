import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import train_real_model


def test_return_bucket_class_separation():
    thresholds = {
        "big_loss": -0.05,
        "loss": -0.01,
        "gain": 0.01,
        "big_gain": 0.05,
    }
    returns = [-0.06, -0.02, 0.0, 0.02, 0.06]
    labels = [train_real_model.return_bucket(r, thresholds) for r in returns]
    assert labels == [0, 1, 2, 3, 4]


@pytest.fixture(autouse=True)
def _mock_min_history(monkeypatch):
    monkeypatch.setattr(
        train_real_model.data_fetcher,
        "has_min_history",
        lambda *a, **k: (True, 1000),
    )
    train_real_model.data_fetcher.MIN_HISTORY_BARS = 0


def _make_df(returns):
    prices = [100, 100, 100]
    for i, r in enumerate(returns):
        prices.append(prices[i] * (1 + r))
    timestamps = pd.date_range(
        "2020-01-01", periods=len(prices), freq="D", tz="UTC"
    )
    df = pd.DataFrame({"Timestamp": timestamps, "Close": prices, "feat": range(len(prices))})
    return df


def test_augment_target_produces_all_classes(monkeypatch):
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

    X, y = train_real_model.prepare_training_data(
        "SYM", "coin", min_unique_samples=3, augment_ratio=1.0
    )
    assert X is not None and y is not None
    counts = y.value_counts().to_dict()
    assert set(counts.keys()) == {0, 1, 2, 3, 4}
    assert all(v == 16 for v in counts.values())
