import os

# Lowest allowed momentum tier (1=strongest, 4=weakest).
# Assets with a tier value higher than this will be skipped.
# Override with environment variable MOMENTUM_TIER_THRESHOLD.
MOMENTUM_TIER_THRESHOLD = int(os.getenv("MOMENTUM_TIER_THRESHOLD", "3"))
