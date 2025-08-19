# Crypto Bot

This repository contains utilities for analyzing cryptocurrency markets and automated trading.

## Requirements

- Python 3.10+
- Google Chrome or Chromium and the matching ChromeDriver for features that rely on Selenium (e.g., scraping CoinMarketCap). Ensure both are installed and available on your `PATH`; otherwise, scraping features will return no data.
- TA-Lib C library.

## Installation

1. Install the native TA-Lib library before installing Python packages.
   - **Debian/Ubuntu**:
     ```bash
     sudo apt-get update && sudo apt-get install -y build-essential ta-lib
     ```
   - **macOS (Homebrew)**:
     ```bash
     brew install ta-lib
     ```
   - **Windows**: Download precompiled TA-Lib binaries from [https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and ensure the library is on your `PATH`.

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

To enable machine-learning based predictions you can train an XGBoost model:

1. Ensure TA-Lib and the required Python packages (e.g., `xgboost`, `pandas`) are installed.
2. Run the training script:
   ```bash
   python train_real_model.py
   ```
   It writes the trained model to `ml_model.json` and the expected feature list to `features.json`.
3. The bot loads these files at runtime in [`model_predictor.py`](model_predictor.py).

## Running the Bot

### Unix-like systems
Use the provided shell script:

```bash
./run_bot.sh
```

### Windows
Run the batch script:

```cmd
run_bot.bat
```

Both scripts change to the project directory and execute `python main.py`.

## Configuration

Momentum scoring thresholds and weights can be adjusted without touching the
source code by setting environment variables:

- `MOMENTUM_TIER_THRESHOLD`: Lowest momentum tier allowed before an asset is
  skipped. Defaults to `3`.
- `MOMENTUM_RETURN_3D_THRESHOLD`, `MOMENTUM_RSI_THRESHOLD`,
  `MOMENTUM_MACD_DIFF_THRESHOLD`, `MOMENTUM_PRICE_SMA20_THRESHOLD`,
  `MOMENTUM_MACD_HIST_NORM_THRESHOLD`: Minimum values required for each
  indicator to contribute to the momentum score. Defaults are `0.015`, `45`,
  and `0` for the remaining indicators respectively.
- `MOMENTUM_RETURN_3D_WEIGHT`, `MOMENTUM_RSI_WEIGHT`,
  `MOMENTUM_MACD_DIFF_WEIGHT`, `MOMENTUM_PRICE_SMA20_WEIGHT`,
  `MOMENTUM_MACD_HIST_NORM_WEIGHT`: Weight applied when an indicator exceeds
  its threshold. All default to `1`.
- `SLIPPAGE_PCT`: Estimated slippage percentage applied to each trade. Defaults to `0`.
- `FEE_PCT`: Trading fee percentage deducted on entry and exit. Defaults to `0.001` (0.1%).
- `MIN_HOLD_BUCKET`: Minimum trade duration bucket that must be reached before a
  position can be closed without exceptional profit. Accepts labels such as
  `<1m`, `1-5m`, `5-30m`, `30m-2h`, and `>2h`. Defaults to `">2h"`.
- `EARLY_EXIT_FEE_MULT`: Profit-to-fee multiple required to exit a trade before
  reaching `MIN_HOLD_BUCKET`. Defaults to `3`.

- `PREDICT_SIGNAL_LOG_FREQ`: How often `predict_signal` emits info-level logs
  of the predicted class. Defaults to `100`. Set to `0` to silence per-iteration
  logs during backtests.

These options allow fine-tuning of momentum evaluation without code changes.

### Tuning Trade Aggressiveness

Several environment variables control how aggressively the bot enters trades.
Lowering these thresholds allows more trades, while raising them makes the bot
more selective:

- `MIN_VOLATILITY_7D`: Minimum 7‑day volatility required to analyze a symbol.
  Defaults to `0.0001`.
- `SUPPRESS_CLASS1_CONF`: If a prediction indicates a small loss and the
  confidence is below this value (default `0.85`), the trade is suppressed.
- `HIGH_CONF_BUY_OVERRIDE`: Confidence needed to upgrade small/big gain
  predictions to BUY signals. Defaults to `0.75`.
- `VERY_HIGH_CONF_BUY_OVERRIDE`: Strongest BUY override for gain predictions.
  Defaults to `0.90`.

By adjusting these values, users can tune the bot's sensitivity to predictions
and market volatility without modifying the source code.

