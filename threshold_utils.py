def get_dynamic_threshold(volatility_7d, base=0.65):
    """Adjust the confidence threshold based on 7d volatility."""
    if volatility_7d > 0.10:
        return base - 0.05
    if volatility_7d < 0.03:
        return base + 0.05
    return base
