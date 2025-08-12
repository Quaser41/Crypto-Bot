from enum import IntEnum

class PredictionClass(IntEnum):
    """Semantic labels for model prediction classes."""

    BIG_LOSS = 0
    SMALL_LOSS = 1
    NEUTRAL = 2
    SMALL_GAIN = 3
    BIG_GAIN = 4
