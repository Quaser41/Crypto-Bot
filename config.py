import os

# Lowest allowed momentum tier (1=strongest, 4=weakest).
# Assets with a tier value higher than this will be skipped.
# Override with environment variable MOMENTUM_TIER_THRESHOLD.
MOMENTUM_TIER_THRESHOLD = int(os.getenv("MOMENTUM_TIER_THRESHOLD", "3"))


# Thresholds and weights used when computing the momentum score. Each entry
# specifies the minimum value required to contribute to the score and the
# weight that will be added when the condition is met. All values can be
# overridden via environment variables so momentum criteria can be tuned
# without modifying code.
MOMENTUM_SCORE_CONFIG = {
    "Return_3d": {
        "threshold": float(os.getenv("MOMENTUM_RETURN_3D_THRESHOLD", "0.015")),
        "weight": float(os.getenv("MOMENTUM_RETURN_3D_WEIGHT", "1")),
    },
    "RSI": {
        "threshold": float(os.getenv("MOMENTUM_RSI_THRESHOLD", "45")),
        "weight": float(os.getenv("MOMENTUM_RSI_WEIGHT", "1")),
    },
    # Difference between MACD and signal line
    "MACD_minus_Signal": {
        "threshold": float(os.getenv("MOMENTUM_MACD_DIFF_THRESHOLD", "0")),
        "weight": float(os.getenv("MOMENTUM_MACD_DIFF_WEIGHT", "1")),
    },
    "Price_vs_SMA20": {
        "threshold": float(os.getenv("MOMENTUM_PRICE_SMA20_THRESHOLD", "0")),
        "weight": float(os.getenv("MOMENTUM_PRICE_SMA20_WEIGHT", "1")),
    },
    "MACD_Hist_norm": {
        "threshold": float(os.getenv("MOMENTUM_MACD_HIST_NORM_THRESHOLD", "0")),
        "weight": float(os.getenv("MOMENTUM_MACD_HIST_NORM_WEIGHT", "1")),
    },
}

# Optional delay between processing symbols during scans.
# Set via environment variable SCAN_DELAY (seconds).
SCAN_DELAY = float(os.getenv("SCAN_DELAY", "0"))

# Delay applied when an error occurs during symbol processing.
# Set via environment variable ERROR_DELAY (seconds).
ERROR_DELAY = float(os.getenv("ERROR_DELAY", "0"))

