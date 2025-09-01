"""Utility helpers for dynamic threshold calculations.

This module centralises logic for determining various thresholds used
throughout the project.  Historically only :func:`get_dynamic_threshold`
was provided, but more advanced return bucketing now requires dynamic
cutoffs derived from each asset's historical behaviour.
"""

from __future__ import annotations

from typing import Iterable, Dict

import numpy as np


def get_dynamic_threshold(volatility_7d: float, base: float = 0.65) -> float:
    """Adjust the confidence threshold based on 7d volatility."""
    if volatility_7d > 0.10:
        return base - 0.05
    if volatility_7d < 0.03:
        return base + 0.05
    return base


def compute_return_thresholds(
    series: Iterable[float],
    quantiles: Iterable[float] = (0.2, 0.4, 0.6, 0.8),
    min_gap: float | None = None,
) -> Dict[str, float]:
    """Compute percentile-based return thresholds.

    Parameters
    ----------
    series:
        Sequence of historical return values.
    quantiles:
        Percentiles used to determine the bucket boundaries. The default
        (20th, 40th, 60th and 80th) yields five buckets with dynamic cutoffs.
    min_gap:
        Minimum allowed distance between adjacent thresholds. When provided
        and the computed thresholds are closer than ``min_gap`` the outer
        quantiles are widened to ``(0.1, 0.3, 0.7, 0.9)`` to avoid overlapping
        or overly narrow classes.

    Returns
    -------
    dict
        Mapping with keys ``"big_loss"``, ``"loss"``, ``"gain"`` and
        ``"big_gain"`` representing the four boundary points used by
        :func:`train_real_model.return_bucket`.

    Notes
    -----
    If any of the five buckets end up empty the outer quantiles are widened
    and the thresholds recomputed using ``(0.1, 0.3, 0.7, 0.9)``.  This helps
    ensure the training data spans all five classes even when return
    distributions are tightly clustered.
    """

    values = list(series)

    def _calc_thresholds(qts: Iterable[float]) -> Dict[str, float]:
        q = np.quantile(values, qts)
        return {
            "big_loss": float(q[0]),
            "loss": float(q[1]),
            "gain": float(q[2]),
            "big_gain": float(q[3]),
        }

    def _bucket_counts(th: Dict[str, float]) -> list[int]:
        counts = [0, 0, 0, 0, 0]
        for r in values:
            if r <= th["big_loss"]:
                counts[0] += 1
            elif r <= th["loss"]:
                counts[1] += 1
            elif r < th["gain"]:
                counts[2] += 1
            elif r < th["big_gain"]:
                counts[3] += 1
            else:
                counts[4] += 1
        return counts

    def _too_narrow(th: Dict[str, float]) -> bool:
        if min_gap is None:
            return False
        gaps = (
            th["loss"] - th["big_loss"],
            th["gain"] - th["loss"],
            th["big_gain"] - th["gain"],
        )
        return any(g < min_gap for g in gaps)

    thresholds = _calc_thresholds(quantiles)
    counts = _bucket_counts(thresholds)

    if 0 in counts or _too_narrow(thresholds):
        thresholds = _calc_thresholds((0.1, 0.3, 0.7, 0.9))
        # Recompute counts only for potential future adjustments/debugging
        counts = _bucket_counts(thresholds)

    return thresholds

