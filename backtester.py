import argparse
import time
import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)

from data_fetcher import fetch_ohlcv_smart
from feature_engineer import add_indicators, momentum_signal
from model_predictor import predict_signal
from utils.prediction_class import PredictionClass
from threshold_utils import get_dynamic_threshold
from config import (
    MOMENTUM_TIER_THRESHOLD,
    SLIPPAGE_PCT,
    FEE_PCT,
    EXECUTION_DELAY_BARS,
    EXECUTION_PRICE_WEIGHT,
)

# === Constants mirrored from main.py ===
# Lower base confidence threshold slightly to allow more trades during backtests
CONFIDENCE_THRESHOLD = 0.55
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

    if label == PredictionClass.SMALL_LOSS.value and confidence < 0.85:
        signal = "HOLD"
    elif label in [PredictionClass.SMALL_GAIN.value, PredictionClass.BIG_GAIN.value] and confidence >= 0.90:
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


MIN_DURATION_YEARS = 0.1  # ~36 days
MIN_PERIODS_PER_YEAR = 50


def compute_metrics(returns: pd.Series, equity_curve: pd.Series, timestamps: pd.Series):
    total_return = equity_curve.iloc[-1]
    duration_years = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / (365 * 24 * 3600)
    if duration_years >= MIN_DURATION_YEARS:
        cagr = total_return ** (1 / duration_years) - 1
    else:
        cagr = np.nan

    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    max_drawdown = drawdown.min()

    period_seconds = np.median(np.diff(timestamps.values).astype('timedelta64[s]').astype(float))
    periods_per_year = (365 * 24 * 3600) / period_seconds if period_seconds else np.nan
    if returns.std() > 0 and not np.isnan(periods_per_year) and periods_per_year >= MIN_PERIODS_PER_YEAR:
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    else:
        sharpe = np.nan

    return {
        "CAGR": cagr,
        "Max Drawdown": max_drawdown,
        "Sharpe": sharpe,
        "Total Return": total_return - 1,
    }



def backtest_symbol(
    symbol: str,
    days: int = 90,
    slippage_pct: float = SLIPPAGE_PCT,
    fee_pct: float = FEE_PCT,
    execution_delay_bars: int = EXECUTION_DELAY_BARS,
    execution_price_weight: float = EXECUTION_PRICE_WEIGHT,
):

    """Backtest a single symbol using historical OHLCV data.

    Parameters
    ----------
    symbol: str
        Ticker to backtest.
    days: int
        Number of days of history to fetch.
    slippage_pct: float
        Percentage slippage deducted on each trade. A warning is logged if
        this value is set to zero.
    fee_pct: float

        Trading fee percentage applied when positions are opened or closed.
    execution_delay_bars: int
        Number of bars to delay order execution after a signal. The trade
        will be executed on the delayed bar's open price.
    execution_price_weight: float
        Weight applied to the delayed bar's open price when executing trades.
        ``1.0`` uses the open price, ``0.0`` uses the close price, and values in
        between use a weighted average.

    """
    if slippage_pct == 0:
        logger.warning("Slippage percentage is 0; results may be overly optimistic.")
    if fee_pct == 0:
        logger.warning("Fee percentage is 0; results may be overly optimistic.")
    start = time.time()
    df = fetch_ohlcv_smart(symbol, days=days, limit=200)
    fetch_seconds = df.attrs.get("fetch_seconds", time.time() - start)
    logger.info("⏱️ Fetched %s in %.2f seconds", symbol, fetch_seconds)
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

def _simulate_trades(df: pd.DataFrame, slippage_pct: float, fee_pct: float) -> dict:
    """Simulate trades on a prepared dataframe and return performance metrics.

    This helper contains the core backtesting loop so it can be reused by
    :func:`backtest_symbol` and stress testing utilities.


    position = 0
    equity = 1.0
    fees_paid = 0.0
    equity_curve = []
    returns = []
    timestamps = df["Timestamp"].iloc[1:].reset_index(drop=True)

    pending_orders: dict[int, str] = {}

    for i in range(1, len(df)):
        price_open = df["Open"].iloc[i]
        price_close = df["Close"].iloc[i]
        prev_close = df["Close"].iloc[i - 1]

        period_return = 0.0

        # Return from previous close to current open if holding a position
        if position == 1:
            period_return += price_open / prev_close - 1

        # Execute any pending order at this bar's open
        if i in pending_orders:
            order = pending_orders.pop(i)
            exec_price = (
                execution_price_weight * price_open
                + (1 - execution_price_weight) * price_close
            )
            if order == "BUY" and position == 0:
                period_return -= slippage_pct + fee_pct
                fees_paid += equity * fee_pct
                position = 1
                period_return += price_close / exec_price - 1
            elif order == "SELL" and position == 1:
                period_return += exec_price / price_open - 1
                period_return -= slippage_pct + fee_pct
                fees_paid += equity * fee_pct
                position = 0
        else:
            if position == 1:
                period_return += price_close / price_open - 1

        equity *= (1 + period_return)
        equity_curve.append(equity)
        returns.append(period_return)

        # Generate signal at end of bar i and schedule execution
        window = df.iloc[: i + 1]
        signal, _, _ = generate_signal(window)
        exec_index = i + 1 + execution_delay_bars
        if exec_index < len(df):
            if signal == "BUY" and position == 0:
                pending_orders[exec_index] = "BUY"
            elif signal == "SELL" and position == 1:
                pending_orders[exec_index] = "SELL"

    returns = pd.Series(returns)
    equity_curve = pd.Series(equity_curve, index=timestamps)
    metrics = compute_metrics(returns, equity_curve, timestamps)
    metrics["Fees Paid"] = fees_paid
    return metrics


def backtest_symbol(symbol: str, days: int = 90, slippage_pct: float = SLIPPAGE_PCT, fee_pct: float = FEE_PCT):
    """Backtest a single symbol using historical OHLCV data.

    Parameters
    ----------
    symbol: str
        Ticker to backtest.
    days: int
        Number of days of history to fetch.
    slippage_pct: float
        Percentage slippage deducted on each trade.
    fee_pct: float
        Trading fee percentage applied when positions are opened or closed.
    """
    start = time.time()
    df = fetch_ohlcv_smart(symbol, days=days, limit=200)
    fetch_seconds = df.attrs.get("fetch_seconds", time.time() - start)
    logger.info("⏱️ Fetched %s in %.2f seconds", symbol, fetch_seconds)
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
    return _simulate_trades(df, slippage_pct, fee_pct)


def run_stress_tests(symbol: str, windows: list, param_grid: list[dict]) -> pd.DataFrame:
    """Run backtests across multiple historical windows and parameter sets.

    Parameters
    ----------
    symbol: str
        Ticker to backtest.
    windows: list of dict
        Each dict should contain ``start`` (datetime-like) and ``days`` keys
        defining the historical slice to test.
    param_grid: list of dict
        Each dict specifies variations of parameters such as ``slippage_pct``
        and ``fee_pct``.

    Returns
    -------
    pandas.DataFrame
        Aggregated metrics for each window and parameter combination.
    """

    results = []
    for win in windows:
        start = pd.to_datetime(win["start"])
        duration = int(win.get("days", win.get("duration", 0)))
        end = start + pd.Timedelta(days=duration)

        df = fetch_ohlcv_smart(symbol, days=duration, limit=200)
        if df.empty:
            logger.warning("⚠️ No data for %s in stress window %s", symbol, win)
            continue

        df = df[(df["Timestamp"] >= start) & (df["Timestamp"] < end)]
        if df.empty:
            logger.warning("⚠️ No data after slicing for %s in window %s", symbol, win)
            continue

        df = add_indicators(df)
        df = add_atr(df)
        df = df.dropna(subset=[
            "RSI", "MACD", "Signal", "Hist", "SMA_20", "SMA_50",
            "Return_1d", "Volatility_7d",
        ])
        if df.empty:
            logger.warning("⚠️ Indicator calculation dropped all rows for %s in window %s", symbol, win)
            continue

        for params in param_grid:
            slippage = params.get("slippage_pct", SLIPPAGE_PCT)
            fee = params.get("fee_pct", FEE_PCT)
            metrics = _simulate_trades(df.copy(), slippage, fee)
            metrics.update({
                "start": start,
                "duration": duration,
                "slippage_pct": slippage,
                "fee_pct": fee,
            })
            results.append(metrics)

    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Backtest breakout strategy")
    parser.add_argument("--symbols", required=True, help="Comma-separated list e.g. BTC,ETH")
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch")
    parser.add_argument("--slippage", type=float, default=SLIPPAGE_PCT, help="Slippage percentage per trade")
    parser.add_argument("--fee", type=float, default=FEE_PCT, help="Trading fee percentage per trade")
    parser.add_argument(
        "--delay",
        type=int,
        default=EXECUTION_DELAY_BARS,
        help="Execution delay in bars before orders are filled",
    )
    parser.add_argument(
        "--delay-weight",
        type=float,
        default=EXECUTION_PRICE_WEIGHT,
        help="Weight for delayed bar's open price when executing orders",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        logger.info("\n=== Backtesting %s ===", sym)
        stats = backtest_symbol(
            sym,
            days=args.days,
            slippage_pct=args.slippage,
            fee_pct=args.fee,
            execution_delay_bars=args.delay,
            execution_price_weight=args.delay_weight,
        )
        if stats:
            for k, v in stats.items():
                logger.info("%s: %.2f%s", k, v * 100 if k != "Sharpe" else v, "%" if k != "Sharpe" else "")


if __name__ == "__main__":
    main()
