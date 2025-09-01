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
    HOLDING_PERIOD_BARS,
    REVERSAL_CONF_DELTA,
    HOLDING_PERIOD_SECONDS,
)
from trade_manager import TradeManager

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

def generate_signal(
    window: pd.DataFrame,
    last_signal=None,
    last_confidence=None,
    reversal_delta: float = 0.0,
):
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

    if (
        reversal_delta > 0
        and last_signal
        and signal != last_signal
        and last_confidence is not None
    ):
        if confidence - last_confidence < reversal_delta:
            logger.info(
                "🔁 Skipping reversal: confidence delta %.2f < %.2f",
                confidence - last_confidence,
                reversal_delta,
            )
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

def _simulate_trades(
    df: pd.DataFrame,
    slippage_pct: float,
    fee_pct: float,
    symbol: str,
    use_trade_manager: bool = False,
    execution_delay_bars: int = EXECUTION_DELAY_BARS,
    execution_price_weight: float = EXECUTION_PRICE_WEIGHT,
    holding_period_bars: int = HOLDING_PERIOD_BARS,
    reversal_conf_delta: float = REVERSAL_CONF_DELTA,
) -> dict:
    """Simulate trades on a prepared dataframe and return performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing OHLCV and indicator columns.
    slippage_pct : float
        Slippage per trade.
    fee_pct : float
        Trading fee per trade.
    symbol : str
        Symbol being tested.
    use_trade_manager : bool
        When ``True`` leverage :class:`TradeManager` for position sizing and
        risk controls. When ``False`` the legacy full-allocation simulator is
        used.
    """

    timestamps = df["Timestamp"].iloc[1:].reset_index(drop=True)

    if not use_trade_manager:
        position = 0
        equity = 1.0
        fees_paid = 0.0
        equity_curve = []
        max_exposure = 0.0
        returns = []
        pending_orders: dict[int, tuple] = {}
        last_trade_idx = -holding_period_bars
        last_signal = None
        last_conf = None

        for i in range(1, len(df)):
            price_open = df["Open"].iloc[i]
            price_close = df["Close"].iloc[i]
            prev_close = df["Close"].iloc[i - 1]

            period_return = 0.0

            if position == 1:
                period_return += price_open / prev_close - 1

            if i in pending_orders:
                order, oconf = pending_orders.pop(i)
                exec_price = (
                    execution_price_weight * price_open
                    + (1 - execution_price_weight) * price_close
                )
                if order == "BUY" and position == 0:
                    period_return -= slippage_pct + fee_pct
                    fees_paid += equity * fee_pct
                    position = 1
                    period_return += price_close / exec_price - 1
                    last_trade_idx = i
                    last_signal = "BUY"
                    last_conf = oconf
                elif order == "SELL" and position == 1:
                    period_return += exec_price / price_open - 1
                    period_return -= slippage_pct + fee_pct
                    fees_paid += equity * fee_pct
                    position = 0
                    last_trade_idx = i
                    last_signal = "SELL"
                    last_conf = oconf if oconf is not None else last_conf
            else:
                if position == 1:
                    period_return += price_close / price_open - 1

            equity *= (1 + period_return)
            exposure = equity if position == 1 else 0.0
            max_exposure = max(max_exposure, exposure)
            equity_curve.append(equity)
            returns.append(period_return)

            window = df.iloc[: i + 1]
            signal, conf, _ = generate_signal(
                window, last_signal, last_conf, reversal_conf_delta
            )
            exec_index = i + 1 + execution_delay_bars
            if exec_index < len(df):
                if signal == "BUY" and position == 0:
                    if i - last_trade_idx < holding_period_bars:
                        logger.info(
                            "⏳ Holding period active — skipping BUY for %s at bar %d",
                            symbol,
                            i,
                        )
                    else:
                        pending_orders[exec_index] = ("BUY", conf)
                elif signal == "SELL" and position == 1:
                    pending_orders[exec_index] = ("SELL", conf)

        returns = pd.Series(returns)
        equity_curve = pd.Series(equity_curve, index=timestamps)
        metrics = compute_metrics(returns, equity_curve, timestamps)
        metrics["Fees Paid"] = fees_paid
        metrics["Max Exposure"] = max_exposure
        if metrics.get("Max Drawdown") and metrics["Max Drawdown"] != 0:
            metrics["Calmar"] = metrics["CAGR"] / abs(metrics["Max Drawdown"])
        else:
            metrics["Calmar"] = np.nan
        return metrics

    # === TradeManager-powered simulation ===
    tm = TradeManager(
        starting_balance=1000,
        trade_fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        hold_period_sec=HOLDING_PERIOD_SECONDS,
    )
    equity_curve = []
    returns = []
    max_exposure = 0.0
    pending_orders: dict[int, dict] = {}
    last_trade_idx = -holding_period_bars
    last_signal = None
    last_conf = None

    for i in range(1, len(df)):
        price_open = df["Open"].iloc[i]
        price_high = df["High"].iloc[i]
        price_low = df["Low"].iloc[i]
        price_close = df["Close"].iloc[i]

        # Execute any pending orders at the bar's open
        if i in pending_orders:
            order = pending_orders.pop(i)
            if order["type"] == "BUY" and not tm.has_position(symbol):
                tm.open_trade(
                    symbol,
                    price_open,
                    confidence=order.get("confidence"),
                    label=order.get("label"),
                    atr=df["ATR"].iloc[i],
                )
                last_trade_idx = i
                last_signal = "BUY"
                last_conf = order.get("confidence")
            elif order["type"] == "SELL" and tm.has_position(symbol):
                tm.close_trade(symbol, price_open, reason="Signal Exit")
                last_trade_idx = i
                last_signal = "SELL"
                last_conf = order.get("confidence", last_conf)

        # Check stops/take-profits within the bar
        for sym, pos in list(tm.positions.items()):
            side = pos.get("side", "BUY")
            if side == "BUY":
                if price_low <= pos["stop_loss"]:
                    tm.close_trade(sym, pos["stop_loss"], reason="Stop-Loss")
                    continue
                if price_high >= pos["take_profit"]:
                    tm.close_trade(sym, pos["take_profit"], reason="Take-Profit")
                    continue
            else:
                if price_high >= pos["stop_loss"]:
                    tm.close_trade(sym, pos["stop_loss"], reason="Stop-Loss")
                    continue
                if price_low <= pos["take_profit"]:
                    tm.close_trade(sym, pos["take_profit"], reason="Take-Profit")
                    continue

        equity = tm.balance + sum(pos["qty"] * price_close for pos in tm.positions.values())
        exposure = sum(pos["qty"] * price_close for pos in tm.positions.values())
        max_exposure = max(max_exposure, exposure)
        equity_curve.append(equity)
        if len(equity_curve) > 1:
            returns.append(equity_curve[-1] / equity_curve[-2] - 1)
        else:
            returns.append(0.0)

        # Generate signal at end of bar and schedule execution
        window = df.iloc[: i + 1]
        signal, conf, label = generate_signal(
            window, last_signal, last_conf, reversal_conf_delta
        )
        exec_index = i + 1 + execution_delay_bars
        if exec_index < len(df):
            if signal == "BUY" and not tm.has_position(symbol):
                if i - last_trade_idx < holding_period_bars:
                    logger.info(
                        "⏳ Holding period active — skipping BUY for %s at bar %d",
                        symbol,
                        i,
                    )
                else:
                    pending_orders[exec_index] = {
                        "type": "BUY",
                        "confidence": conf,
                        "label": label,
                    }
            elif signal == "SELL" and tm.has_position(symbol):
                pending_orders[exec_index] = {"type": "SELL"}

    equity_series = pd.Series(equity_curve, index=timestamps)
    equity_norm = equity_series / tm.starting_balance
    returns = pd.Series(returns)
    metrics = compute_metrics(returns, equity_norm, timestamps)
    metrics["Fees Paid"] = tm.total_fees
    metrics["Max Exposure"] = max_exposure / tm.starting_balance
    if metrics.get("Max Drawdown") and metrics["Max Drawdown"] != 0:
        metrics["Calmar"] = metrics["CAGR"] / abs(metrics["Max Drawdown"])
    else:
        metrics["Calmar"] = np.nan
    return metrics


def backtest_symbol(
    symbol: str,
    days: int = 90,
    slippage_pct: float = SLIPPAGE_PCT,
    fee_pct: float = FEE_PCT,
    execution_delay_bars: int = EXECUTION_DELAY_BARS,
    execution_price_weight: float = EXECUTION_PRICE_WEIGHT,
    use_trade_manager: bool = False,
    compare: bool = False,
    holding_period_bars: int = HOLDING_PERIOD_BARS,
    reversal_conf_delta: float = REVERSAL_CONF_DELTA,
):
    """Backtest a single symbol using historical OHLCV data.

    Parameters
    ----------
    symbol : str
        Ticker to backtest.
    days : int
        Number of days of history to fetch.
    slippage_pct : float
        Slippage applied on each trade.
    fee_pct : float
        Trading fee percentage per trade.
    execution_delay_bars : int
        Bars to delay order execution after a signal.
    execution_price_weight : float
        Weight applied to the delayed bar's open price when executing trades.
    use_trade_manager : bool
        When ``True`` use :class:`TradeManager` for risk-aware execution.
    compare : bool
        If ``True`` return metrics for both legacy and risk-managed modes.
    """
    start = time.time()
    df = fetch_ohlcv_smart(symbol, days=days, limit=200)
    fetch_seconds = df.attrs.get("fetch_seconds", time.time() - start)
    logger.info("⏱️ Fetched %s in %.2f seconds", symbol, fetch_seconds)
    if df.empty:
        logger.error("❌ No data for %s", symbol)
        return None
    df = add_indicators(df)
    if df.empty:
        logger.warning("⚠️ Indicator calculation returned no data for %s", symbol)
        return None
    df = add_atr(df)
    df = df.dropna(subset=[
        "RSI", "MACD", "Signal", "Hist", "SMA_20", "SMA_50",
        "Return_1d", "Volatility_7d",
        "ATR",
    ])
    if df.empty:
        logger.warning("⚠️ Indicator calculation dropped all rows for %s", symbol)
        return None

    base_metrics = _simulate_trades(
        df.copy(),
        slippage_pct,
        fee_pct,
        symbol,
        False,
        execution_delay_bars,
        execution_price_weight,
        holding_period_bars,
        reversal_conf_delta,
    )
    if compare or use_trade_manager:
        tm_metrics = _simulate_trades(
            df.copy(),
            slippage_pct,
            fee_pct,
            symbol,
            True,
            execution_delay_bars,
            execution_price_weight,
            holding_period_bars,
            reversal_conf_delta,
        )
        if compare:
            return {"baseline": base_metrics, "risk_managed": tm_metrics}
        return tm_metrics
    return base_metrics


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
            metrics = _simulate_trades(
                df.copy(),
                slippage,
                fee,
                symbol,
                False,
                EXECUTION_DELAY_BARS,
                EXECUTION_PRICE_WEIGHT,
            )
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
            compare=True,
        )
        if stats:
            base = stats.get("baseline", {})
            rm = stats.get("risk_managed", {})
            logger.info("-- Baseline --")
            for k, v in base.items():
                logger.info("%s: %.2f%s", k, v * 100 if k not in {"Sharpe", "Calmar"} else v, "%" if k not in {"Sharpe", "Calmar"} else "")
            logger.info("-- With Risk Controls --")
            for k, v in rm.items():
                logger.info("%s: %.2f%s", k, v * 100 if k not in {"Sharpe", "Calmar"} else v, "%" if k not in {"Sharpe", "Calmar"} else "")


if __name__ == "__main__":
    main()
