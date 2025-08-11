# feature_engineer.py
# TA-Lib is optional. Import if available, otherwise proceed without it.
try:
    import talib  # type: ignore  # noqa: F401
except ImportError:
    talib = None

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from config import MOMENTUM_SCORE_CONFIG

# Minimum rows required after indicator calculations
MIN_ROWS_AFTER_INDICATORS = 60

def add_indicators(df, min_rows: int = MIN_ROWS_AFTER_INDICATORS):
    if df.empty or "Close" not in df.columns:
        print("⚠️ Cannot add indicators: DataFrame empty or missing 'Close'")
        return pd.DataFrame()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    if len(df) < 50:
        print(f"⚠️ Not enough candles to compute full indicators: {len(df)} rows")
        return pd.DataFrame()

    # RSI
    rsi = RSIIndicator(df["Close"], window=14)
    df["RSI"] = rsi.rsi()

    # MACD and signal line
    macd_ind = MACD(df["Close"])
    df["MACD"] = macd_ind.macd()
    df["Signal"] = macd_ind.macd_signal()
    df["Hist"] = macd_ind.macd_diff()

    # SMA 20 and 50
    sma_20 = SMAIndicator(df["Close"], window=20)
    sma_50 = SMAIndicator(df["Close"], window=50)
    df["SMA_20"] = sma_20.sma_indicator()
    df["SMA_50"] = sma_50.sma_indicator()

    # Daily and short-term returns
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_2d"] = df["Close"].pct_change(periods=2)
    df["Return_3d"] = df["Close"].pct_change(periods=3)
    df["Return_5d"] = df["Close"].pct_change(periods=5)
    df["Return_7d"] = df["Close"].pct_change(periods=7)

    # Short- and mid-term volatility
    df["Volatility_3d"] = df["Close"].rolling(window=3, min_periods=3).std()
    df["Volatility_7d"] = df["Close"].rolling(window=7, min_periods=7).std()
    if df["Volatility_7d"].dropna().eq(0).any():
        print("⚠️ Volatility_7d contains zero values; check OHLCV data quality")

    # Price deltas
    df["Price_Change_3d"] = df["Close"] - df["Close"].shift(3)

    # Relative position vs SMAs
    df["Price_vs_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    df["Price_vs_SMA50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]

    # Normalized MACD histogram
    df["MACD_Hist_norm"] = df["Hist"] / df["Close"]

    # Add momentum score
    df["Momentum_Score"] = df.apply(compute_momentum_score, axis=1)
    df["Momentum_Tier"] = df["Momentum_Score"].apply(classify_momentum_tier)

    df = df.dropna()
    remaining = df.shape[0]
    if remaining < min_rows:
        print(
            f"⚠️ Indicators left only {remaining} rows (<{min_rows}); skipping symbol"
        )
        # Return an empty DataFrame with expected columns so downstream code can handle gracefully
        return pd.DataFrame(columns=df.columns)

    print(f"✅ Indicators added: {remaining} rows remaining after dropna")
    return df

def momentum_signal(df):
    closes = df["Close"].tail(7).values
    if len(closes) < 7:
        return "HOLD"

    short_term_pct = (closes[-1] - closes[-3]) / closes[-3] * 100
    weekly_pct = (closes[-1] - closes[0]) / closes[0] * 100

    if short_term_pct > 2.5 and weekly_pct > 5:
        return "BUY"
    elif short_term_pct < -2.5 and weekly_pct < -5:
        return "SELL"
    else:
        return "HOLD"

def compute_momentum_score(row):
    """Compute a weighted momentum score for a row of indicator data.

    Thresholds and weights are defined in :data:`config.MOMENTUM_SCORE_CONFIG`
    and can be overridden via environment variables. This allows momentum
    scoring to be tuned without modifying code.
    """

    cfg = MOMENTUM_SCORE_CONFIG
    score = 0.0
    if row.get("Return_3d", 0) > cfg["Return_3d"]["threshold"]:
        score += cfg["Return_3d"]["weight"]
    if row.get("RSI", 0) > cfg["RSI"]["threshold"]:
        score += cfg["RSI"]["weight"]
    if (row.get("MACD", 0) - row.get("Signal", 0)) > cfg["MACD_minus_Signal"]["threshold"]:
        score += cfg["MACD_minus_Signal"]["weight"]
    if row.get("Price_vs_SMA20", 0) > cfg["Price_vs_SMA20"]["threshold"]:
        score += cfg["Price_vs_SMA20"]["weight"]
    if row.get("MACD_Hist_norm", 0) > cfg["MACD_Hist_norm"]["threshold"]:
        score += cfg["MACD_Hist_norm"]["weight"]
    return score

def classify_momentum_tier(score):
    if score >= 4:
        return "Tier 1"
    elif score >= 3:
        return "Tier 2"
    elif score >= 2:
        return "Tier 3"
    else:
        return "Tier 4"
