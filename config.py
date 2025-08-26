import os

# Trading mode: "live" routes orders to the real exchange, while "paper"
# uses the sandbox/testnet APIs.  Default to paper trading for safety.
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()

# Lowest allowed momentum tier (1=strongest, 4=weakest).
# Assets with a tier value higher than this will be skipped.
# The default of ``4`` temporarily admits all tiers so we can analyze score
# distribution and decide on future cut-offs. Override with environment
# variable MOMENTUM_TIER_THRESHOLD.
MOMENTUM_TIER_THRESHOLD = int(os.getenv("MOMENTUM_TIER_THRESHOLD", "4"))

# When enabled, the feature engineering pipeline will print the distribution of
# computed momentum tiers to help tune thresholds and weights.
LOG_MOMENTUM_DISTRIBUTION = os.getenv("LOG_MOMENTUM_DISTRIBUTION", "0") == "1"


# Thresholds and weights used when computing the momentum score. Each entry
# specifies the minimum value required to contribute to the score and the
# weight that will be added when the condition is met. All values can be
# overridden via environment variables so momentum criteria can be tuned
# without modifying code.
MOMENTUM_SCORE_CONFIG = {
    "Return_3d": {
        # Lowered to allow mildly positive three-day returns to contribute.
        "threshold": float(os.getenv("MOMENTUM_RETURN_3D_THRESHOLD", "0.01")),
        "weight": float(os.getenv("MOMENTUM_RETURN_3D_WEIGHT", "1")),
    },
    "RSI": {
        # Broaden acceptable RSI range to capture assets exiting oversold zones.
        "threshold": float(os.getenv("MOMENTUM_RSI_THRESHOLD", "40")),
        "weight": float(os.getenv("MOMENTUM_RSI_WEIGHT", "1")),
    },
    # Difference between MACD and signal line
    "MACD_minus_Signal": {
        "threshold": float(os.getenv("MOMENTUM_MACD_DIFF_THRESHOLD", "0")),
        "weight": float(os.getenv("MOMENTUM_MACD_DIFF_WEIGHT", "1")),
    },
    "Price_vs_SMA20": {
        "threshold": float(os.getenv("MOMENTUM_PRICE_SMA20_THRESHOLD", "0")),
        # Slightly higher weight to favor assets trending above the 20-day SMA.
        "weight": float(os.getenv("MOMENTUM_PRICE_SMA20_WEIGHT", "1.2")),
    },
    "MACD_Hist_norm": {
        "threshold": float(os.getenv("MOMENTUM_MACD_HIST_NORM_THRESHOLD", "0")),
        "weight": float(os.getenv("MOMENTUM_MACD_HIST_NORM_WEIGHT", "1")),
    },
}

# Delay applied when an error occurs during symbol processing.
# Set via environment variable ERROR_DELAY (seconds).
ERROR_DELAY = float(os.getenv("ERROR_DELAY", "0"))


# Multipliers for ATR-based stop-loss and take-profit calculations.
# These can be overridden via environment variables ATR_MULT_SL and ATR_MULT_TP.
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "2.0"))
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "3.0"))

# === Risk management and trade sizing ===
# Percentage of account equity risked per trade (e.g. 0.01 = 1%)
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))

# Minimum dollar allocation for any trade. Trades below this are skipped.
MIN_TRADE_USD = float(os.getenv("MIN_TRADE_USD", "10"))

# Slippage percentage applied to entry and exit prices. Defaults to 0.1%
# to approximate typical market impact on each trade.
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.001"))

# Trading fee percentage applied to entry and exit prices. Defaults to 0.1%
# to model exchange fees and is used as the default fee in TradeManager.
FEE_PCT = float(os.getenv("FEE_PCT", "0.001"))

# Minimum acceptable ratio of expected profit to total fees for a trade.
# Trades falling below this profit-to-fee threshold will be skipped.
# Bumped to a stricter default of 7.0 to demand greater edge over fees
# before executing any position.
MIN_PROFIT_FEE_RATIO = float(os.getenv("MIN_PROFIT_FEE_RATIO", "7.0"))

# Price stagnation detection parameters. If price movement stays below
# ``STAGNATION_THRESHOLD_PCT`` for ``STAGNATION_DURATION_SEC`` seconds, the
# position will be closed.
STAGNATION_THRESHOLD_PCT = float(os.getenv("STAGNATION_THRESHOLD_PCT", "0.005"))
STAGNATION_DURATION_SEC = int(os.getenv("STAGNATION_DURATION_SEC", "1800"))

# Allocation scaling parameters for drawdown control.
ALLOCATION_MAX_DD = float(os.getenv("ALLOCATION_MAX_DD", "0.10"))
ALLOCATION_MIN_FACTOR = float(os.getenv("ALLOCATION_MIN_FACTOR", "0.5"))

# Number of bars to delay trade execution in backtests to model latency.
EXECUTION_DELAY_BARS = int(os.getenv("EXECUTION_DELAY_BARS", "0"))

# Weight for the delayed bar's open price when executing trades.
# 1.0 uses the open price exclusively, 0.0 uses the close price,
# and values in between compute a weighted average.
EXECUTION_PRICE_WEIGHT = float(os.getenv("EXECUTION_PRICE_WEIGHT", "1.0"))

# Baseline minimum model confidence required to consider a trade.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

# --- Trade aggressiveness controls ---
# Minimum confidence below which a predicted small-loss trade is suppressed.
SUPPRESS_CLASS1_CONF = float(os.getenv("SUPPRESS_CLASS1_CONF", "0.85"))

# Confidence at which small/big gain predictions are treated as high conviction buys.
HIGH_CONF_BUY_OVERRIDE = float(os.getenv("HIGH_CONF_BUY_OVERRIDE", "0.75"))

# Confidence required for the strongest BUY override.
VERY_HIGH_CONF_BUY_OVERRIDE = float(os.getenv("VERY_HIGH_CONF_BUY_OVERRIDE", "0.90"))

# Minimum 7-day volatility required for a symbol to be considered.
MIN_VOLATILITY_7D = float(os.getenv("MIN_VOLATILITY_7D", "0.0001"))

# Minimum historical win rate (%) required for symbols to be considered.
# Default to 60% so underperforming assets are filtered out unless
# explicitly overridden via the ``MIN_SYMBOL_WIN_RATE`` environment variable.
MIN_SYMBOL_WIN_RATE = float(os.getenv("MIN_SYMBOL_WIN_RATE", "60"))

# Minimum average PnL required for symbols to be considered.
# Any symbol with average returns at or below this threshold will be skipped
# unless ``MIN_SYMBOL_AVG_PNL`` is overridden via environment variables.
MIN_SYMBOL_AVG_PNL = float(os.getenv("MIN_SYMBOL_AVG_PNL", "0.05"))

# Minimum bars to wait after a trade before opening a new one in backtests.
HOLDING_PERIOD_BARS = int(os.getenv("HOLDING_PERIOD_BARS", "0"))

# Minimum seconds to wait after a trade before opening a new one in live trading.
# Enforce a minimum of 5 minutes to bucket trades more coarsely.
HOLDING_PERIOD_SECONDS = max(300, int(os.getenv("HOLDING_PERIOD_SECONDS", "300")))

# Minimum trade duration bucket required before exits are allowed without
# exceptional profitability. Uses labels from
# ``analytics.performance.get_duration_bucket``.
MIN_HOLD_BUCKET = os.getenv("MIN_HOLD_BUCKET", ">2h")

# Profit-to-fee multiple required to exit a trade before reaching
# ``MIN_HOLD_BUCKET``.
EARLY_EXIT_FEE_MULT = float(os.getenv("EARLY_EXIT_FEE_MULT", "3"))

# Additional confidence required before reversing an open position.
REVERSAL_CONF_DELTA = float(os.getenv("REVERSAL_CONF_DELTA", "0"))


# Seconds before refreshing performance blacklist from analytics file.
BLACKLIST_REFRESH_SEC = int(os.getenv("BLACKLIST_REFRESH_SEC", "3600"))
