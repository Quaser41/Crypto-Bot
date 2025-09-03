import os
import sys
import numpy as np
import pandas as pd

# Ensure repository root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from train_real_model import select_top_features

def test_select_top_features_identifies_informative_columns():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=200))
    x1 = y + rng.normal(scale=0.1, size=len(y))  # informative
    x2 = rng.normal(size=len(y))  # noise
    X = pd.DataFrame({"x1": x1, "x2": x2})

    X_sel = select_top_features(X, y, top_n=1)
    assert list(X_sel.columns) == ["x1"]


def test_select_top_features_handles_large_top_n():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, size=100))
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])

    X_sel = select_top_features(X, y, top_n=10)
    assert set(X_sel.columns) == {"a", "b", "c"}
