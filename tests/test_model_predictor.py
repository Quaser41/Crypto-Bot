import logging
import types
import threading
import time

import numpy as np
import pandas as pd
import model_predictor


def test_class_probabilities_logged_at_debug(monkeypatch, caplog):
    class DummyModel:
        def predict(self, dmatrix):
            return np.array([[0.1, 0.2, 0.3, 0.15, 0.25]])

    monkeypatch.setattr(
        model_predictor,
        "load_model",
        lambda: (DummyModel(), ["a", "b", "c", "d", "e"], [0, 1, 2, 3, 4]),
    )
    monkeypatch.setattr(model_predictor, "xgb", types.SimpleNamespace(DMatrix=lambda X: X))

    df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "e": [5]})

    with caplog.at_level(logging.DEBUG, logger=model_predictor.logger.name):
        model_predictor.predict_signal(df, threshold=0.5)

    assert any("Class probabilities" in record.getMessage() for record in caplog.records)


def test_predict_signal_fills_missing_4h_features(monkeypatch):
    class DummyModel:
        def predict(self, dmatrix):
            return np.array([[0.1, 0.2, 0.3, 0.15, 0.25]])

    features = [
        "RSI", "MACD", "Signal", "Hist", "SMA_20", "SMA_50",
        "Return_1d", "Return_2d", "Return_3d", "Return_5d", "Return_7d",
        "Price_vs_SMA20", "Price_vs_SMA50", "Volatility_7d", "MACD_Hist_norm",
        "MACD_4h", "Signal_4h", "Hist_4h", "SMA_4h"
    ]

    monkeypatch.setattr(
        model_predictor,
        "load_model",
        lambda: (DummyModel(), features, [0, 1, 2, 3, 4]),
    )
    monkeypatch.setattr(model_predictor, "xgb", types.SimpleNamespace(DMatrix=lambda X: X))

    df = pd.DataFrame({f: [1.0] for f in features if not f.endswith("_4h")})

    signal, confidence, cls = model_predictor.predict_signal(df, threshold=0.5)
    assert signal is not None
    for col in ["SMA_4h", "MACD_4h", "Signal_4h", "Hist_4h"]:
        assert col in df.columns
        assert df[col].iloc[-1] == 0.0


def test_predict_signal_uses_label_mapping(monkeypatch):
    class DummyModel:
        def predict(self, dmatrix):
            return np.array([[0.1, 0.3, 0.4, 0.2]])

    features = ["a", "b", "c", "d"]
    labels = [0, 2, 3, 4]

    monkeypatch.setattr(
        model_predictor,
        "load_model",
        lambda: (DummyModel(), features, labels),
    )
    monkeypatch.setattr(model_predictor, "xgb", types.SimpleNamespace(DMatrix=lambda X: X))

    df = pd.DataFrame({f: [1.0] for f in features})

    signal, confidence, cls = model_predictor.predict_signal(df, threshold=0.2)
    assert cls == 3  # maps index 2 -> original label 3
    assert signal == "BUY"
    assert abs(confidence - 0.4) < 1e-6


def test_predict_signal_thread_safe_logging(monkeypatch, caplog):
    class DummyModel:
        def predict(self, dmatrix):
            return np.array([[0.9, 0.05, 0.03, 0.01, 0.01]])

    features = ["a", "b", "c", "d", "e"]

    monkeypatch.setattr(
        model_predictor,
        "load_model",
        lambda: (DummyModel(), features, [0, 1, 2, 3, 4]),
    )
    monkeypatch.setattr(model_predictor, "xgb", types.SimpleNamespace(DMatrix=lambda X: X))

    model_predictor._prediction_counter = 0
    model_predictor._last_logged_class = None

    original_info = model_predictor.logger.info

    def slow_info(*args, **kwargs):
        time.sleep(0.01)
        return original_info(*args, **kwargs)

    monkeypatch.setattr(model_predictor.logger, "info", slow_info)

    def call():
        df = pd.DataFrame({f: [1.0] for f in features})
        model_predictor.predict_signal(df, threshold=0.5, log_frequency=1000)

    with caplog.at_level(logging.INFO, logger=model_predictor.logger.name):
        threads = [threading.Thread(target=call) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert model_predictor._prediction_counter == 20
    info_logs = [r for r in caplog.records if "Predicted class" in r.getMessage()]
    assert len(info_logs) == 1
