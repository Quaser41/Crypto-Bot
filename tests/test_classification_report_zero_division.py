import warnings

from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning


def test_classification_report_zero_division_suppresses_warning():
    """classification_report should not emit warnings when zero_division=0."""
    y_true = [0, 1]
    y_pred = [0, 0]  # class 1 has no predictions
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        classification_report(y_true, y_pred, labels=[0, 1], zero_division=0)
    assert not any(
        issubclass(warning.category, UndefinedMetricWarning) for warning in w
    )
