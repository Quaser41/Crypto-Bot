# feature_engineer.py
# TA-Lib is optional. Import if available, otherwise proceed without it.
try:
    import talib  # type: ignore  # noqa: F401
except ImportError:
    talib = None

import numpy as np
import pandas as pd
import asyncio
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from config import LOG_MOMENTUM_DISTRIBUTION, MOMENTUM_SCORE_CONFIG
from data_fetcher import fetch_fear_greed_index, fetch_onchain_metrics, fetch_ohlcv_smart

from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MIN_ROWS_AFTER_INDICATORS = 60

def add_indicators(df, min_rows: int = DEFAULT_MIN_ROWS_AFTER_INDICATORS):
    """Add technical and sentiment indicators to OHLCV data.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Raw OHLCV data.
    min_rows : int, default ``DEFAULT_MIN_ROWS_AFTER_INDICATORS``
        Minimum number of rows required after indicator computation. If fewer
        rows remain, an empty DataFrame is returned.
    """
    if df.empty or "Close" not in df.columns:
        logger.warning("⚠️ Cannot add indicators: DataFrame empty or missing 'Close'")
        return pd.DataFrame()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    # Ensure timestamps are timezone-aware (UTC) for downstream merges
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

    required = max(min_rows, 50)
    if len(df) < required:
        logger.warning(
            "⚠️ Skipping symbol: %d rows (<%d required)",
            len(df),
            required,
        )
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

    # Bollinger Bands (20-period)
    bb = BollingerBands(close=df["Close"], window=20)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    for col in ["BB_Upper", "BB_Middle", "BB_Lower"]:
        df[col] = df[col].ffill().bfill()

    # Exponential Moving Averages
    ema9 = EMAIndicator(df["Close"], window=9)
    ema26 = EMAIndicator(df["Close"], window=26)
    df["EMA_9"] = ema9.ema_indicator().ffill().bfill()
    df["EMA_26"] = ema26.ema_indicator().ffill().bfill()

    # On-Balance Volume and volume SMA ratios
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        obv_ind = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
        df["OBV"] = obv_ind.on_balance_volume()
        vol_sma20 = df["Volume"].rolling(window=20, min_periods=1).mean()
        df["Volume_vs_SMA20"] = (df["Volume"] - vol_sma20) / vol_sma20
        df["OBV"] = df["OBV"].ffill().bfill()
        df["Volume_vs_SMA20"] = df["Volume_vs_SMA20"].ffill().bfill()
    else:
        df["OBV"] = 0.0
        df["Volume_vs_SMA20"] = 0.0

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
    vol7 = df["Volatility_7d"].dropna()
    if vol7.eq(0).all():
        logger.warning(
            "⚠️ Volatility_7d is zero for all points; constant price data detected. Skipping symbol."
        )
        return pd.DataFrame()
    if vol7.eq(0).any():
        zero_rows = df["Volatility_7d"] == 0
        logger.warning("⚠️ Dropping %d rows with zero Volatility_7d", zero_rows.sum())
        df = df.loc[~zero_rows].copy()

    # Price deltas
    df.loc[:, "Price_Change_3d"] = df["Close"] - df["Close"].shift(3)

    # Relative position vs SMAs
    df.loc[:, "Price_vs_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    df.loc[:, "Price_vs_SMA50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]

    # Normalized MACD histogram
    df.loc[:, "MACD_Hist_norm"] = df["Hist"] / df["Close"]

    # Relative Strength vs BTC
    try:
        # Determine how many days of BTC history are needed based on the
        # earliest timestamp in ``df``.  This anchors the start date relative
        # to ``datetime.utcnow()`` instead of defaulting to the Unix epoch.
        earliest = df["Timestamp"].min().to_pydatetime().replace(tzinfo=None)
        span_days = (datetime.utcnow() - earliest).days + 1
        span_days = max(span_days, 60)
        btc = fetch_ohlcv_smart("BTC", interval="1d", coin_id="bitcoin", days=span_days)
        if not btc.empty and "Close" in btc.columns:
            btc["Timestamp"] = pd.to_datetime(btc["Timestamp"], utc=True)
            btc = btc.sort_values("Timestamp")
            df = pd.merge_asof(
                df.sort_values("Timestamp"),
                btc[["Timestamp", "Close"]].rename(columns={"Close": "BTC_Close"}),
                on="Timestamp",
                direction="backward",
            )
            df["RelStrength_BTC"] = df["Close"] / df["BTC_Close"]
            df.drop(columns=["BTC_Close"], inplace=True)
            df["RelStrength_BTC"] = df["RelStrength_BTC"].ffill().bfill()
        else:
            df["RelStrength_BTC"] = 1.0
    except Exception as e:
        logger.warning("⚠️ Failed to compute relative strength vs BTC: %s", e)
        df["RelStrength_BTC"] = 1.0

    # ==== Higher timeframe aggregates (e.g., 4-hour candles) ====
    four_h_cols = ["SMA_4h", "MACD_4h", "Signal_4h", "Hist_4h"]
    try:
        agg = (
            df.sort_values("Timestamp")
            .set_index("Timestamp")["Close"]
            .resample("4h", label="right", closed="right")
            .last()
            .ffill()
            .to_frame()
        )
        required_points = 26  # longest 4h window needed (MACD slow period)
        if len(agg) >= required_points:
            agg_sma = SMAIndicator(agg["Close"], window=20)
            agg["SMA_4h"] = agg_sma.sma_indicator()
            agg_macd = MACD(agg["Close"])
            agg["MACD_4h"] = agg_macd.macd()
            agg["Signal_4h"] = agg_macd.macd_signal()
            agg["Hist_4h"] = agg_macd.macd_diff()
            agg = agg[four_h_cols].ffill().reset_index()
            df = pd.merge_asof(
                df.sort_values("Timestamp"),
                agg.sort_values("Timestamp"),
                on="Timestamp",
                direction="backward",
                tolerance=pd.Timedelta("4h"),
            )
            df[four_h_cols] = df[four_h_cols].ffill()
        else:
            logger.warning(
                "⚠️ Not enough history for 4h aggregates: %d < %d", len(agg), required_points
            )
            for col in four_h_cols:
                df[col] = np.nan
    except Exception as e:
        logger.warning("⚠️ Failed to compute 4h aggregates: %s", e)
        for col in four_h_cols:
            df[col] = np.nan

    if set(four_h_cols).issubset(df.columns):
        if df[four_h_cols].isna().all().any():
            logger.warning(
                "⚠️ 4h aggregates contain NaNs after merge; verify timestamp alignment"
            )

    # ==== Merge sentiment and on-chain metrics ====
    df = df.sort_values("Timestamp")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)

    sentiment = fetch_fear_greed_index(limit=365)
    if not sentiment.empty:
        sentiment["Timestamp"] = pd.to_datetime(sentiment["Timestamp"]).dt.tz_localize(None)
        sentiment = sentiment.sort_values("Timestamp")
        df = pd.merge_asof(df, sentiment, on="Timestamp", direction="backward")
        df["FearGreed"] = df["FearGreed"].ffill().bfill()
        std = df["FearGreed"].std()
        if std and std != 0:
            df["FearGreed_norm"] = (df["FearGreed"] - df["FearGreed"].mean()) / std
        else:
            df["FearGreed_norm"] = 0.0
        df.drop(columns=["FearGreed"], inplace=True)
        df["FearGreed_norm"] = df["FearGreed_norm"].fillna(0.0)
    else:
        df["FearGreed_norm"] = 0.0

    # Fetch a manageable span of on-chain data.  Using the default
    # lookback avoids requests outside the provider's supported range
    # which previously resulted in 404 responses.
    onchain = fetch_onchain_metrics()
    if not onchain.empty:
        # Ensure timestamps are timezone-naive before merging so ``pd.merge_asof``
        # doesn't complain about mismatched timezone-aware vs. naive data.
        onchain["Timestamp"] = pd.to_datetime(onchain["Timestamp"]).dt.tz_localize(None)
        onchain = onchain.sort_values("Timestamp")
        df = pd.merge_asof(df, onchain, on="Timestamp", direction="backward")
        for col in ["TxVolume", "ActiveAddresses"]:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
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
        df["TxVolume_norm"] = df["TxVolume_norm"].fillna(0.0)
        df["ActiveAddresses_norm"] = df["ActiveAddresses_norm"].fillna(0.0)
    else:
        df["TxVolume_norm"] = 0.0
        df["ActiveAddresses_norm"] = 0.0

    # Add momentum score
    df["Momentum_Score"] = df.apply(compute_momentum_score, axis=1)
    df["Momentum_Tier"] = df["Momentum_Score"].apply(classify_momentum_tier)

    if LOG_MOMENTUM_DISTRIBUTION:
        tier_counts = df["Momentum_Tier"].value_counts(dropna=False)
        logger.info("Momentum tier distribution:\n%s", tier_counts.to_string())

    # Drop columns that are entirely NaN (e.g., failed on-chain metrics)
    optional_nan_cols = {"SMA_4h", "MACD_4h", "Signal_4h", "Hist_4h"}
    all_nan_cols = [
        col for col in df.columns if df[col].isna().all() and col not in optional_nan_cols
    ]
    if all_nan_cols:
        logger.debug("Dropping all-NaN columns: %s", all_nan_cols)
        df = df.drop(columns=all_nan_cols)

    # Replace infinities with NaN and drop rows missing essential indicators
    essential_cols = [
        "Close",
        "RSI",
        "MACD",
        "Signal",
        "SMA_20",
        "SMA_50",
    ]
    before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    subset_cols = [c for c in essential_cols if c in df.columns]
    df = df.dropna(subset=subset_cols)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(
            "⚠️ Dropped %d rows due to non-finite or missing essential values", dropped
        )
    remaining = df.shape[0]
    if remaining < min_rows:
        logger.warning(
            "⚠️ Indicators left only %d rows (<%d); skipping symbol",
            remaining,
            min_rows,
        )
        # Return an empty DataFrame with expected columns so downstream code can handle gracefully
        return pd.DataFrame(columns=df.columns)

    logger.info("✅ Indicators added: %d rows remaining after dropna", remaining)
    return df


async def add_indicators_async(df, min_rows: int = DEFAULT_MIN_ROWS_AFTER_INDICATORS):
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
