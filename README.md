
# Crypto Bot

This repository contains utilities for analyzing cryptocurrency markets and automated trading.

## Requirements

- Python 3.10+
- Google Chrome or Chromium and the matching ChromeDriver for features that rely on Selenium (e.g., scraping CoinMarketCap). Without Chrome or Chromium installed, those features will return no data.

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

=======
# Crypto-Bot


## Installation

1. Install the native TA-Lib library **before** installing Python packages.
   - **Debian/Ubuntu**:
     ```bash
     sudo apt-get update && sudo apt-get install -y build-essential ta-lib
     ```
   - **macOS (Homebrew)**:
     ```bash
     brew install ta-lib
     ```
   - **Windows**: Download precompiled TA-Lib binaries from [https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and install them, ensuring the library is on your PATH.

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

=======
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
- `FEE_PCT`: Trading fee percentage deducted on entry and exit. Defaults to `0`.

- `PREDICT_SIGNAL_LOG_FREQ`: How often `predict_signal` emits info-level logs
  of the predicted class. Defaults to `100`. Set to `0` to silence per-iteration
  logs during backtests.

These options allow fine-tuning of momentum evaluation without code changes.

