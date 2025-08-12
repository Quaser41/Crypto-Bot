import argparse
import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)

from data_fetcher import fetch_ohlcv_smart
from feature_engineer import add_indicators, momentum_signal
from model_predictor import predict_signal
from threshold_utils import get_dynamic_threshold
from config import MOMENTUM_TIER_THRESHOLD

# === Constants mirrored from main.py ===
CONFIDENCE_THRESHOLD = 0.65
FALLBACK_RSI_THRESHOLD = 55
FALLBACK_RETURN_3D_THRESHOLD = 0.03
FLAT_1D_THRESHOLD = 0.001
FLAT_3D_THRESHOLD = 0.003
TIER_RANKS = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3, "Tier 4": 4}


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Average True Range (ATR) and add it as a column."""
    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        return df
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=period, min_periods=period).mean()
    return df


def generate_signal(window: pd.DataFrame):
    vol_7d = window["Volatility_7d"].iloc[-1]
    if vol_7d < 1e-4:
        return "HOLD", 0.0, None

    threshold = get_dynamic_threshold(vol_7d, base=CONFIDENCE_THRESHOLD)
    momentum_tier = window["Momentum_Tier"].iloc[-1]
    if TIER_RANKS.get(momentum_tier, 4) > MOMENTUM_TIER_THRESHOLD:
        return "HOLD", 0.0, None

    try:
        signal, confidence, label = predict_signal(window, threshold)
    except Exception:
        return momentum_signal(window), 0.0, None

    if label == 1 and confidence < 0.85:
        signal = "HOLD"
    elif label in [3, 4] and confidence >= 0.90:
        signal = "BUY"
    elif signal == "HOLD" or confidence < threshold:
        if (
            abs(window["Return_1d"].iloc[-1]) < FLAT_1D_THRESHOLD
            and abs(window["Return_3d"].iloc[-1]) < FLAT_3D_THRESHOLD
        ):
            signal = "HOLD"
        else:
            signal = momentum_signal(window)
            if not (
                signal == "BUY"
                and window["Return_3d"].iloc[-1] > FALLBACK_RETURN_3D_THRESHOLD
                and window["RSI"].iloc[-1] > FALLBACK_RSI_THRESHOLD
            ):
                signal = "HOLD"
    return signal, confidence, label


def compute_metrics(returns: pd.Series, equity_curve: pd.Series, timestamps: pd.Series):
    total_return = equity_curve.iloc[-1]
    duration_years = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / (365 * 24 * 3600)
    cagr = total_return ** (1 / duration_years) - 1 if duration_years > 0 else np.nan

    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    max_drawdown = drawdown.min()

    period_seconds = np.median(np.diff(timestamps.values).astype('timedelta64[s]').astype(float))
    periods_per_year = (365 * 24 * 3600) / period_seconds if period_seconds else np.nan
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else np.nan

    return {
        "CAGR": cagr,
        "Max Drawdown": max_drawdown,
        "Sharpe": sharpe,
        "Total Return": total_return - 1,
    }


def backtest_symbol(symbol: str, days: int = 90, slippage_pct: float = 0.001):
    df = fetch_ohlcv_smart(symbol, days=days, limit=200)
    if df.empty:
        logger.error("❌ No data for %s", symbol)
        return None
    df = add_indicators(df)
    df = add_atr(df)
    df = df.dropna(subset=[
        "RSI", "MACD", "Signal", "Hist", "SMA_20", "SMA_50",
        "Return_1d", "Volatility_7d"
    ])
    if df.empty:
        logger.warning("⚠️ Indicator calculation dropped all rows for %s", symbol)
        return None

    position = 0
    equity = 1.0
    equity_curve = [equity]
    returns = []
    timestamps = df["Timestamp"].iloc[1:].reset_index(drop=True)

    for i in range(1, len(df)):
        window = df.iloc[:i]
        signal, _, _ = generate_signal(window)

        price = df["Close"].iloc[i]
        prev_price = df["Close"].iloc[i - 1]
        ret = price / prev_price - 1
        period_return = ret if position == 1 else 0

        if signal == "BUY" and position == 0:
            period_return -= slippage_pct
            position = 1
        elif signal == "SELL" and position == 1:
            period_return -= slippage_pct
            position = 0

        equity *= (1 + period_return)
        equity_curve.append(equity)
        returns.append(period_return)

    returns = pd.Series(returns)
    equity_curve = pd.Series(equity_curve[1:], index=timestamps)
    metrics = compute_metrics(returns, equity_curve, timestamps)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Backtest breakout strategy")
    parser.add_argument("--symbols", required=True, help="Comma-separated list e.g. BTC,ETH")
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch")
    parser.add_argument("--slippage", type=float, default=0.001, help="Slippage percentage per trade")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        logger.info("\n=== Backtesting %s ===", sym)
        stats = backtest_symbol(sym, days=args.days, slippage_pct=args.slippage)
        if stats:
            for k, v in stats.items():
                logger.info("%s: %.2f%s", k, v * 100 if k != "Sharpe" else v, "%" if k != "Sharpe" else "")


if __name__ == "__main__":
    main()
