import os
import sys
import pandas as pd
import numpy as np

# Ensure repository root on path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from train_real_model import train_model


def test_train_model_runs_with_smote():
    # Create deterministic synthetic dataset with 5 classes
    n_classes = 5
    samples_per_class = 10  # ensures sufficient samples for SMOTE
    y = pd.Series(np.tile(np.arange(n_classes), samples_per_class))
    X = pd.DataFrame({
        "feat1": np.arange(len(y)),
        "feat2": np.arange(len(y)) * 2,
    })

    model, labels = train_model(
        X,
        y,
        oversampler="smote",
        param_scale="small",
        cv_splits=2,
        verbose=0,
    )

    # Model should be returned and all classes should be preserved
    assert model is not None
    assert set(labels) == set(range(n_classes))
