import logging
import types

import numpy as np
import pandas as pd
import model_predictor


def test_class_probabilities_logged_at_debug(monkeypatch, caplog):
    class DummyModel:
        def predict(self, dmatrix):
            return np.array([[0.1, 0.2, 0.3, 0.15, 0.25]])

    monkeypatch.setattr(
        model_predictor, "load_model", lambda: (DummyModel(), ["a", "b", "c", "d", "e"])
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

    monkeypatch.setattr(model_predictor, "load_model", lambda: (DummyModel(), features))
    monkeypatch.setattr(model_predictor, "xgb", types.SimpleNamespace(DMatrix=lambda X: X))

    df = pd.DataFrame({f: [1.0] for f in features if not f.endswith("_4h")})

    signal, confidence, cls = model_predictor.predict_signal(df, threshold=0.5)
    assert signal is not None
    for col in ["SMA_4h", "MACD_4h", "Signal_4h", "Hist_4h"]:
        assert col in df.columns
        assert df[col].iloc[-1] == 0.0
