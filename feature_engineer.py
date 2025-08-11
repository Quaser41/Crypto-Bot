# feature_engineer.py
# TA-Lib is optional. Import if available, otherwise proceed without it.
try:
    import talib  # type: ignore  # noqa: F401
except ImportError:
    talib = None

import numpy as np
import pandas as pd
import asyncio
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange
from config import LOG_MOMENTUM_DISTRIBUTION, MOMENTUM_SCORE_CONFIG
from data_fetcher import fetch_fear_greed_index, fetch_onchain_metrics

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

    # Average True Range (ATR) as a volatility measure. If High/Low are not
    # provided by the data source, fall back to a rolling standard deviation of
    # Close prices as a rough proxy.
    if {"High", "Low"}.issubset(df.columns):
        atr_ind = AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=14
        )
        df["ATR"] = atr_ind.average_true_range()
    else:
        df["ATR"] = df["Close"].rolling(window=14, min_periods=14).std()

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

    # ==== Merge sentiment and on-chain metrics ====
    df = df.sort_values("Timestamp")

    sentiment = fetch_fear_greed_index(limit=365)
    if not sentiment.empty:
        sentiment = sentiment.sort_values("Timestamp")
        df = pd.merge_asof(df, sentiment, on="Timestamp", direction="backward")
        df["FearGreed"] = df["FearGreed"].ffill()
        std = df["FearGreed"].std()
        if std and std != 0:
            df["FearGreed_norm"] = (df["FearGreed"] - df["FearGreed"].mean()) / std
        else:
            df["FearGreed_norm"] = 0.0
        df.drop(columns=["FearGreed"], inplace=True)
    else:
        df["FearGreed_norm"] = 0.0

    onchain = fetch_onchain_metrics(days=365)
    if not onchain.empty:
        onchain = onchain.sort_values("Timestamp")
        df = pd.merge_asof(df, onchain, on="Timestamp", direction="backward")
        for col in ["TxVolume", "ActiveAddresses"]:
            df[col] = df[col].ffill()
        if "TxVolume" in df.columns:
            std = df["TxVolume"].std()
            df["TxVolume_norm"] = (
                (df["TxVolume"] - df["TxVolume"].mean()) / std if std and std != 0 else 0.0
            )
        else:
            df["TxVolume_norm"] = 0.0
        if "ActiveAddresses" in df.columns:
            std = df["ActiveAddresses"].std()
            df["ActiveAddresses_norm"] = (
                (df["ActiveAddresses"] - df["ActiveAddresses"].mean()) / std
                if std and std != 0
                else 0.0
            )
        else:
            df["ActiveAddresses_norm"] = 0.0
        df.drop(columns=[c for c in ["TxVolume", "ActiveAddresses"] if c in df.columns], inplace=True)
    else:
        df["TxVolume_norm"] = 0.0
        df["ActiveAddresses_norm"] = 0.0

    # Add momentum score
    df["Momentum_Score"] = df.apply(compute_momentum_score, axis=1)
    df["Momentum_Tier"] = df["Momentum_Score"].apply(classify_momentum_tier)

    if LOG_MOMENTUM_DISTRIBUTION:
        tier_counts = df["Momentum_Tier"].value_counts(dropna=False)
        print("Momentum tier distribution:\n" + tier_counts.to_string())

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


async def add_indicators_async(df, min_rows: int = MIN_ROWS_AFTER_INDICATORS):
    """Run :func:`add_indicators` in a background thread."""
    return await asyncio.to_thread(add_indicators, df, min_rows=min_rows)

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
