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
) -> Dict[str, float]:
    """Compute percentile-based return thresholds.

    Parameters
    ----------
    series:
        Sequence of historical return values.
    quantiles:
        Percentiles used to determine the bucket boundaries. The default
        (20th, 40th, 60th and 80th) yields five buckets with dynamic cutoffs.

    Returns
    -------
    dict
        Mapping with keys ``"big_loss"``, ``"loss"``, ``"gain"`` and
        ``"big_gain"`` representing the four boundary points used by
        :func:`train_real_model.return_bucket`.
    """

    q = np.quantile(list(series), quantiles)
    return {
        "big_loss": float(q[0]),
        "loss": float(q[1]),
        "gain": float(q[2]),
        "big_gain": float(q[3]),
    }

