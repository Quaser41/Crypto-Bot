import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.base import is_classifier
from sklearn.model_selection import TimeSeriesSplit


def calibrate_and_analyze(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    label_names,
    method: str = "isotonic",
) -> Tuple[CalibratedClassifierCV, Dict[str, float]]:
    """Calibrate ``model`` and derive class thresholds.

    The function wraps ``model`` with
    :class:`~sklearn.calibration.CalibratedClassifierCV` using a
    :class:`~sklearn.model_selection.TimeSeriesSplit`.  The calibrator is
    fitted on ``X_train``/``y_train`` and evaluated on ``X_val``/``y_val``
    to compute ROC and precision/recall curves for each class.  A simple
    profit heuristic then determines a recommended probability cutoff for
    each class.

    Parameters
    ----------
    model : estimator
        Unfit classifier providing ``predict_proba`` after fitting.
    X_train, y_train : array-like
        Training feature matrix and labels used to fit the calibrator.
    X_val, y_val : array-like
        Validation feature matrix and labels used for analysis.
    label_names : list of str
        Names corresponding to each column in ``predict_proba``.
    method : {"sigmoid", "isotonic"}
        Calibration method passed to :class:`CalibratedClassifierCV`.

    Returns
    -------
    calibrated_model : :class:`~sklearn.calibration.CalibratedClassifierCV`
        The calibrated wrapper around ``model``.
    thresholds : Dict[str, float]
        Mapping of class name to recommended probability cutoff.
    """

    if not is_classifier(model) and hasattr(model, "predict_proba"):
        setattr(model, "_estimator_type", "classifier")

    calibrator = CalibratedClassifierCV(
        model, cv=TimeSeriesSplit(n_splits=3), method=method
    )
    calibrator.fit(X_train, y_train)
    probas = calibrator.predict_proba(X_val)

    profit_per_class = {0: -2.0, 1: -1.0, 2: 0.0, 3: 1.0, 4: 2.0}
    thresholds: Dict[str, float] = {}
    pred_classes = np.argmax(probas, axis=1)

    os.makedirs("analytics", exist_ok=True)
    for idx, name in enumerate(label_names):
        cls_probs = probas[:, idx]
        y_bin = (y_val == idx).astype(int)
        fpr, tpr, roc_thr = roc_curve(y_bin, cls_probs)
        pr_prec, pr_rec, pr_thr = precision_recall_curve(y_bin, cls_probs)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": np.r_[roc_thr, np.nan]}).to_csv(
            os.path.join("analytics", f"roc_{name}.csv"), index=False
        )
        pd.DataFrame(
            {
                "precision": pr_prec,
                "recall": pr_rec,
                "threshold": np.r_[pr_thr, np.nan],
            }
        ).to_csv(os.path.join("analytics", f"pr_{name}.csv"), index=False)

        best_thr, best_profit = 0.0, float("-inf")
        for thr in np.linspace(0, 1, 101):
            mask = (pred_classes == idx) & (cls_probs >= thr)
            if not mask.any():
                continue
            profit = float(np.sum([profit_per_class[int(y)] for y in y_val[mask]]))
            if profit > best_profit:
                best_profit = profit
                best_thr = thr
        thresholds[name] = round(float(best_thr), 2)

    with open(os.path.join("analytics", "recommended_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    return calibrator, thresholds
