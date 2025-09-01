# Crypto Bot

This repository contains utilities for analyzing cryptocurrency markets and automated trading.

## Requirements

- Python 3.10+
- Google Chrome or Chromium and the matching ChromeDriver for features that rely on Selenium (e.g., scraping CoinMarketCap). Ensure both are installed and available on your `PATH`; otherwise, scraping features will return no data.
- TA-Lib C library.
- imbalanced-learn>=0.11 for optional oversampling techniques.

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
   If the repository is updated later, reinstall dependencies to ensure pinned
   versions are used:
   ```bash
   pip install --upgrade -r requirements.txt
   ```
   The `update_repo.bat` script performs this step automatically.

## Training the Model

To enable machine-learning based predictions you can train an XGBoost model:

1. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure the TA-Lib library is installed before running this command.
2. Run the training script:
   ```bash
   python train_real_model.py
   ```
It writes the trained model to `ml_model.json` and overwrites `features.json` with the exact
feature names used, keeping inference and training in sync.
3. The bot loads these files at runtime in [`model_predictor.py`](model_predictor.py).

By default the script only trains on symbols whose 24‑hour quote volume exceeds
`MIN_24H_VOLUME` (50 000 USD). You can override this threshold at runtime with
`--min-volume` or set the `MIN_24H_VOLUME` environment variable for
configuration‑file style control.

The feature‑engineering pipeline scales its minimum history requirement with
the amount of data fetched (60 % by default). This adaptive threshold can be
customised via the ``min_rows_ratio`` argument to
``train_real_model.prepare_training_data`` when integrating the training
utilities programmatically.  The prediction horizon defaults to three future
15‑minute bars (45 minutes) but can be adjusted with the ``horizon`` argument
or ``--horizon`` CLI flag (e.g. ``--horizon 288`` for roughly three days).

The label preparation step drops rows whose future return is smaller than
0.5 % in magnitude.  This threshold can be tweaked with ``--min-return`` on the
command line or the corresponding ``min_return`` argument when calling
``train_real_model.prepare_training_data`` directly.  Set it to ``0`` to keep
all rows for experimentation.

### Handling Class Imbalance

Time‑series data makes synthetic sampling tricky. By default the training
script applies **RandomOverSampler**, which simply duplicates existing rows and
preserves temporal order. This requires the
[`imbalanced-learn`](https://imbalanced-learn.org/) package.

To rely solely on class weighting (no oversampling):

```bash
python train_real_model.py --oversampler none
```

Other strategies like **SMOTE** or **ADASYN** are available but may introduce
temporal leakage because they interpolate between points. Use them only if the
time dependence is negligible:

```bash
python train_real_model.py --oversampler smote  # or adasyn/borderline
```

Choose a strategy based on how strictly you need to preserve chronology in
your data.

## Engineered Features

The feature engineering pipeline expands raw OHLCV data with a variety of
technical indicators and sentiment/on‑chain metrics.  Notable features
include:

- RSI, MACD (and signal/histogram)
- SMA 20 & 50, plus 4‑hour equivalents
- Bollinger Bands (20‑period)
- EMA(9) and EMA(26)
- On‑Balance Volume and volume vs. 20‑day SMA ratios
- Daily through 7‑day returns and volatility measures
- Price position vs. moving averages and normalized MACD histogram
- Fear & Greed index and normalized on‑chain activity
- Relative strength versus Bitcoin

The full set is listed in `features.json` and mirrored in
`train_real_model.DEFAULT_FEATURES`.

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
- `COINGECKO_API_KEY`: CoinGecko API key used for authenticated requests.
  If unset, a demo key is used. Requests include the key via the
  `x-cg-demo-api-key` header, and Coingecko OHLCV lookups automatically retry
  once with this header if the first attempt returns `401`.
- `BLOCKCHAIN_API_KEY`: API key for Blockchain.com charts used to fetch
  on-chain metrics. When absent, the bot skips network calls and returns
  placeholder metrics.
- `GLASSNODE_API_KEY`: Optional Glassnode API key for enhanced on-chain
  metrics. If unset, only public data sources are queried.

These options allow fine-tuning of momentum evaluation without code changes.

### OHLCV Data Source Fallback

Candlestick data is pulled from several providers in order of preference:

1. **Coinbase** – primary source
2. **Binance.US**
3. **Binance**
4. **Yahoo Finance**
5. **CoinGecko**
6. **DexScreener**

If Coinbase returns fewer than 60 rows for a request, the Binance sources are
moved to the front of the list so they are tried first on subsequent attempts.

### Tuning Trade Aggressiveness

Several environment variables control how aggressively the bot enters trades.
Lowering these thresholds allows more trades, while raising them makes the bot
more selective:

- `CONFIDENCE_THRESHOLD`: Baseline confidence required before a prediction is
  considered tradable. Defaults to `0.78`.
- `MIN_VOLATILITY_7D`: Minimum 7‑day volatility required to analyze a symbol.
  Defaults to `0.0001`.
- `SUPPRESS_CLASS1_CONF`: If a prediction indicates a small loss and the
  confidence is below this value (default `0.88`), the trade is suppressed.
- `HIGH_CONF_BUY_OVERRIDE`: Confidence needed to upgrade small/big gain
  predictions to BUY signals. Defaults to `0.84`.
- `VERY_HIGH_CONF_BUY_OVERRIDE`: Strongest BUY override for gain predictions.
  Defaults to `0.90`.

These defaults were derived from isotonic probability calibration and analysis
of ROC and precision-recall curves on the expanded validation set. By adjusting
them, users can further tune the bot's sensitivity to predictions and market
volatility without modifying the source code.

