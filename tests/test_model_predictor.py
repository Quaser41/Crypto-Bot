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
