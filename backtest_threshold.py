import pandas as pd
import numpy as np
from feature_engineer import add_indicators
from model_predictor import predict_signal
from data_fetcher import fetch_ohlcv_smart
from threshold_utils import get_dynamic_threshold

def backtest_thresholds(coin_id, thresholds=None, slippage_pct: float = 0.001):
    if thresholds is None:
        thresholds = np.arange(0.5, 0.75, 0.05)  # test 0.50 to 0.70

    # Fetch OHLCV and add indicators
    df = fetch_ohlcv_smart(coin_id)
    df = add_indicators(df)
    df = df.reset_index(drop=True)
    print(f"Backtesting on {len(df)} rows")

    results = []

    # We'll simulate a simple long-only strategy with buy/sell signals
    for thresh in thresholds:
        position = 0  # 0=no position, 1=long
        entry_price = 0.0
        returns = []
        trades = 0

        for i in range(20, len(df)):  # start after some rows to have features ready
            window_df = df.iloc[:i+1].copy()

            # Get ML signal and confidence for the current row
            vol_7d = window_df["Volatility_7d"].iloc[-1]
            dyn_thresh = get_dynamic_threshold(vol_7d)
            signal, conf, _ = predict_signal(window_df, threshold=dyn_thresh)

            if signal is None or conf is None:
                continue

            # Apply threshold filter
            if conf < thresh:
                # Fallback: treat as HOLD (no trade)
                signal = "HOLD"

            price = df.loc[i, "Close"]

            # Simulate trades
            if signal == "BUY" and position == 0:
                position = 1
                entry_price = price * (1 + slippage_pct)
                trades += 1
            elif signal == "SELL" and position == 1:
                # Close position, calculate return with slippage
                exit_price = price * (1 - slippage_pct)
                ret = (exit_price - entry_price) / entry_price
                returns.append(ret)
                position = 0
                entry_price = 0

        # If still holding position at end, close at last price
        if position == 1:
            exit_price = df["Close"].iloc[-1] * (1 - slippage_pct)
            ret = (exit_price - entry_price) / entry_price
            returns.append(ret)

        total_return = np.sum(returns) if returns else 0
        avg_return = np.mean(returns) if returns else 0
        win_rate = np.mean([r > 0 for r in returns]) if returns else 0

        results.append({
            "threshold": thresh,
            "total_return": total_return,
            "avg_return": avg_return,
            "trades": trades,
            "win_rate": win_rate
        })

    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    print(results_df)
    best = results_df.loc[results_df["total_return"].idxmax()]
    print(f"\nBest threshold: {best['threshold']} with total return {best['total_return']:.2f}")
    return results_df

if __name__ == "__main__":
    # Example run on bitcoin
    backtest_thresholds("bitcoin")
