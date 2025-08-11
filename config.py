import os

# Lowest allowed momentum tier (1=strongest, 4=weakest).
# Assets with a tier value higher than this will be skipped.
# Override with environment variable MOMENTUM_TIER_THRESHOLD.
MOMENTUM_TIER_THRESHOLD = int(os.getenv("MOMENTUM_TIER_THRESHOLD", "3"))

# Optional delay between processing symbols during scans.
# Set via environment variable SCAN_DELAY (seconds).
SCAN_DELAY = float(os.getenv("SCAN_DELAY", "0"))

# Delay applied when an error occurs during symbol processing.
# Set via environment variable ERROR_DELAY (seconds).
ERROR_DELAY = float(os.getenv("ERROR_DELAY", "0"))
