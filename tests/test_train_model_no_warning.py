import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

import train_real_model


def test_train_model_no_undefined_metric_warning(tmp_path):
    os.chdir(tmp_path)
    X = pd.DataFrame(np.random.rand(60, 3), columns=["a", "b", "c"])
    y = pd.Series(np.concatenate([np.zeros(50), np.ones(10)]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UndefinedMetricWarning)
        train_real_model.train_model(X, y)
