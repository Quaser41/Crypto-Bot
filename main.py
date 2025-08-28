import time
import os
import json
import pandas as pd
import threading
import asyncio
import argparse
import signal
import sys

# ✅ Now use refactored fetchers
from data_fetcher import (
    get_top_gainers,
    fetch_ohlcv_smart,
    fetch_ohlcv_smart_async,
    clear_old_cache,
)
from feature_engineer import add_indicators, momentum_signal
from model_predictor import predict_signal
from utils.prediction_class import PredictionClass
from trade_manager import TradeManager
from config import (
    MOMENTUM_TIER_THRESHOLD,
    ERROR_DELAY,
    CONFIDENCE_THRESHOLD,
    TRADING_MODE,
    MIN_SYMBOL_WIN_RATE,
    MIN_SYMBOL_AVG_PNL,
    WIN_RATE_WEIGHT,
    SUPPRESS_CLASS1_CONF,
    HIGH_CONF_BUY_OVERRIDE,
    VERY_HIGH_CONF_BUY_OVERRIDE,
    MIN_VOLATILITY_7D,
    HOLDING_PERIOD_SECONDS,
    CORRELATION_THRESHOLD,
)
from exchange_adapter import BinancePaperTradeAdapter
from threshold_utils import get_dynamic_threshold
from utils.logging import get_logger
from symbol_resolver import filter_candidates

logger = get_logger(__name__)

# Load symbol performance statistics
try:
    _stats_df = pd.read_csv("analytics/trade_stats.csv")
    SYMBOL_PERFORMANCE = (
        _stats_df.groupby("symbol")[['avg_pnl', 'win_rate']].mean().to_dict("index")
    )
    logger.info(f"Loaded trade stats for {len(SYMBOL_PERFORMANCE)} symbols")
except Exception as e:
    logger.warning(f"Could not load trade stats: {e}")
    SYMBOL_PERFORMANCE = {}

# ✅ Global thresholds
# CONFIDENCE_THRESHOLD imported from config
ROTATE_CONF_THRESHOLD = 0.05        # new trade must have +5% higher confidence to rotate
MOMENTUM_ADV_THRESHOLD = 0.5        # candidate must exceed current momentum score by 0.5
STAGNATION_THRESHOLD = 0.01         # <1% price movement = stagnation
ROTATION_AUDIT_LOG = []  # 📘 In-memory rotation history
ROTATION_LOG_LIMIT = 10  # How many to keep
# Protects access to the rotation audit log
ROTATION_AUDIT_LOCK = threading.Lock()
# Correlation filter threshold for new candidates (imported from config)
# CORRELATION_THRESHOLD is imported above

# Momentum tier mapping for gating logic
TIER_RANKS = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3, "Tier 4": 4}

# ✅ Fallback momentum thresholds
FALLBACK_RSI_THRESHOLD = 55
FALLBACK_RETURN_3D_THRESHOLD = 0.03
FLAT_1D_THRESHOLD = 0.001
FLAT_3D_THRESHOLD = 0.003

# 🔍 Rotation audit logging (toggleable)
ENABLE_ROTATION_AUDIT = os.getenv("ENABLE_ROTATION_AUDIT", "1") not in ("0", "false", "False")
PERSIST_ROTATION_AUDIT = os.getenv("PERSIST_ROTATION_AUDIT", "1") not in (
    "0",
    "false",
    "False",
)

def log_rotation_decision(current, candidate):
    logger.info("\n🔁 ROTATION DECISION:")
    logger.info(f" - Current: {current['symbol']} | Conf {current['confidence']:.2f} | Label {current['label']} | Entry ${current['entry_price']:.4f} | Movement {current['movement']:.2%}")
    logger.info(f" - Candidate: {candidate['symbol']} | Conf {candidate['confidence']:.2f} | Label {candidate['label']} | Price ${candidate['price']:.4f}")

def record_rotation_audit(current, candidate, pnl_before, pnl_after=None):
    with ROTATION_AUDIT_LOCK:
        ROTATION_AUDIT_LOG.append({
            "timestamp": time.time(),
            "current": current,
            "candidate": candidate,
            "pnl_before": pnl_before,
            "pnl_after": pnl_after,
        })
        if len(ROTATION_AUDIT_LOG) > ROTATION_LOG_LIMIT:
            ROTATION_AUDIT_LOG.pop(0)


def save_rotation_audit(filepath="rotation_audit.json", max_entries=100):
    """Persist in-memory rotation audit to JSON, keeping the newest entries."""
    if not ROTATION_AUDIT_LOG:
        return
    try:
        existing = []
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                existing = json.load(f)
        existing.extend(ROTATION_AUDIT_LOG)
        if len(existing) > max_entries:
            existing = existing[-max_entries:]
        with open(filepath, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save rotation audit: {e}")
    finally:
        ROTATION_AUDIT_LOG.clear()

# ✅ TradeManager instance
if TRADING_MODE == "paper":
    try:
        exchange = BinancePaperTradeAdapter()
        logger.info("📈 Paper trading mode enabled (Binance testnet)")
    except Exception as e:
        logger.warning(f"Paper adapter unavailable, reverting to simulation: {e}")
        exchange = None
else:
    exchange = None

tm = TradeManager(exchange=exchange, hold_period_sec=HOLDING_PERIOD_SECONDS)
tm.load_state()

# ✅ Background thread for monitoring existing trades
def monitor_thread():
    while True:
        if tm.positions:
            tm.monitor_open_trades(single_run=True)
            time.sleep(60)
        else:
            time.sleep(180)

t = threading.Thread(target=monitor_thread, daemon=True)
t.start()


def handle_exit(signum, frame):
    logger.info(f"Received signal {signum}, exiting...")
    tm.save_state()
    if ENABLE_ROTATION_AUDIT and PERSIST_ROTATION_AUDIT:
        save_rotation_audit()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


def scan_for_breakouts():
    if not tm.can_trade():
        logger.warning("🚫 Risk thresholds hit — skipping scan for new trades.")
        return

    logger.info(f"⚠️ Currently open trades before scanning: {list(tm.positions.keys())}")

    movers = get_top_gainers(limit=15)

    if not movers:
        logger.error("❌ No valid gainers found on Coinbase — skipping scan.")
        return

    logger.info("\n🔥 TOP GAINERS:")
    for cid, sym, name, chg, vol in movers:
        logger.info(f" - {name} ({sym}) {chg:.2f}% | vol {vol:,.0f}")

    candidates = []
    open_symbols = list(tm.positions.keys())
    suppressed, fallbacks = 0, 0

    fetch_meta = filter_candidates(
        movers, open_symbols, SYMBOL_PERFORMANCE, win_rate_weight=WIN_RATE_WEIGHT
    )

    logger.info("📡 Fetching OHLCV data for candidates...")
    ohlcvs = [
        fetch_ohlcv_smart(symbol, coin_id=coin_id, days=10, limit=500)
        for coin_id, symbol, _ in fetch_meta
    ] if fetch_meta else []

    for (coin_id, symbol, name), df in zip(fetch_meta, ohlcvs):
        logger.info(f"\n🔎 Analyzing {name} ({symbol})...")

        if df.empty or len(df) < 60:
            logger.warning(f"⚠️ Not enough OHLCV for {symbol}, skipping.")
            if ERROR_DELAY:
                time.sleep(ERROR_DELAY)
            continue

        df = add_indicators(df).dropna(subset=[
            "RSI", "MACD", "Signal", "Hist", "SMA_20", "SMA_50",
            "Return_1d", "Volatility_7d"
        ])
        if df.empty:
            logger.warning("⚠️ Indicator calc dropped all rows, skipping...")
            if ERROR_DELAY:
                time.sleep(ERROR_DELAY)
            continue

        vol_7d = df["Volatility_7d"].iloc[-1]
        if vol_7d < MIN_VOLATILITY_7D:
            logger.warning(f"⚠️ Skipping {symbol}: 7d volatility too low ({vol_7d:.6f})")
            continue

        threshold = get_dynamic_threshold(vol_7d, base=CONFIDENCE_THRESHOLD)

        # ✅ NEW: Momentum filter
        momentum_tier = df["Momentum_Tier"].iloc[-1]
        momentum_score = df["Momentum_Score"].iloc[-1]
        if TIER_RANKS.get(momentum_tier, 4) > MOMENTUM_TIER_THRESHOLD:
            logger.error(
                f"❌ Skipping {symbol}: weak momentum ({momentum_tier}, score={momentum_score})"
            )
            continue

        try:
            signal, confidence, label = predict_signal(df, threshold)
            logger.info(f"🤖 ML Signal: {signal} (conf={confidence:.2f}, label={label})")
            logger.info(f"🧠 Threshold: {threshold:.2f} (7d vol={vol_7d:.3f})")

            if label == PredictionClass.SMALL_LOSS.value and confidence < SUPPRESS_CLASS1_CONF:
                logger.info(
                    f"🚫 Suppressing weak Class 1 pick: {symbol} (conf={confidence:.2f})"
                )
                signal = "HOLD"
                suppressed += 1

            if label in [PredictionClass.SMALL_GAIN.value, PredictionClass.BIG_GAIN.value] and confidence >= HIGH_CONF_BUY_OVERRIDE:
                logger.info("🔥 High Conviction BUY override active")
                signal = "BUY"

            # === Skip flat coins early (no momentum) ===
            if (
                abs(df["Return_1d"].iloc[-1]) < FLAT_1D_THRESHOLD
                and abs(df["Return_3d"].iloc[-1]) < FLAT_3D_THRESHOLD
            ):
                logger.info(
                    f"⛔ Skipping {symbol}: too flat for fallback trigger (1d={df['Return_1d'].iloc[-1]:.2%}, 3d={df['Return_3d'].iloc[-1]:.2%})"
                )
                continue

            # === High-confidence override for Class 3 or 4 ===
            if label in [PredictionClass.SMALL_GAIN.value, PredictionClass.BIG_GAIN.value] and confidence >= VERY_HIGH_CONF_BUY_OVERRIDE:
                logger.info(
                    f"🟢 High-conviction BUY override: {symbol} → label={label}, conf={confidence:.2f}"
                )
                signal = "BUY"

            # === Suppress weak Class 1 signals ===
            elif label == PredictionClass.SMALL_LOSS.value and confidence < SUPPRESS_CLASS1_CONF:
                logger.info(
                    f"🚫 Suppressing weak Class 1 pick: {symbol} (conf={confidence:.2f})"
                )
                signal = "HOLD"
                suppressed += 1

            # === ML HOLD → fallback strategy ===
            if signal == "HOLD" or confidence < threshold:
                logger.warning(
                    f"⚠️ No ML trigger → Using fallback momentum strategy for {symbol}"
                )
                signal = momentum_signal(df)
                logger.info(f"📉 Fallback momentum signal: {signal}")
                label = PredictionClass.NEUTRAL.value
                fallbacks += 1

                # === Fallback breakout trigger ===
                if (
                    signal == "BUY"
                    and df["Return_3d"].iloc[-1] > FALLBACK_RETURN_3D_THRESHOLD
                    and df["RSI"].iloc[-1] > FALLBACK_RSI_THRESHOLD
                ):
                    logger.info(
                        f"🔥 Fallback BUY trigger: {symbol} shows 3d return {df['Return_3d'].iloc[-1]:.2%} with RSI {df['RSI'].iloc[-1]:.1f}"
                    )
                else:
                    signal = "HOLD"

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            signal = momentum_signal(df)
            confidence = 0.0
            label = PredictionClass.NEUTRAL.value
            fallbacks += 1

        last_price = df["Close"].iloc[-1]

        if signal == "BUY" and label in [PredictionClass.SMALL_GAIN.value, PredictionClass.BIG_GAIN.value] and last_price > 0:
            if confidence < CONFIDENCE_THRESHOLD:
                logger.error(
                    f"❌ Skipping {symbol}: confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"
                )
                continue
            candidates.append((symbol, last_price, confidence, coin_id, signal, label, momentum_score))
        else:
            logger.error(
                f"❌ Skipping {symbol}: signal={signal}, label={label}, conf={confidence:.2f}"
            )

    logger.info(f"📉 {suppressed} suppressed | 🔄 {fallbacks} fallback-triggered")

    if not candidates:
        logger.error("\n❌ No trade candidates found.")
        logger.info("✅ Scan cycle complete\n" + "="*40)
        return

    # Sort candidates by confidence (desc)
    candidates.sort(key=lambda x: x[2], reverse=True)

    selected = None
    for symbol, price, conf, coin_id, sig, label, mom_score in candidates:
        skip_due_to_corr = False
        if tm.positions:
            try:
                cand_df = fetch_ohlcv_smart(symbol, coin_id=coin_id, days=10)
                cand_returns = cand_df["Close"].pct_change().dropna().tail(7)
            except Exception as e:
                logger.warning(f"Correlation check failed for {symbol}: {e}")
                cand_returns = pd.Series(dtype=float)

            for sym, pos in tm.positions.items():
                try:
                    open_df = fetch_ohlcv_smart(sym, coin_id=pos.get("coin_id"), days=10)
                    open_returns = open_df["Close"].pct_change().dropna().tail(7)
                    if len(cand_returns) >= 7 and len(open_returns) >= 7:
                        corr = cand_returns.corr(open_returns)
                        logger.info(
                            f"📈 7d return correlation {symbol}-{sym}: {corr:.2f}"
                        )
                        if corr >= CORRELATION_THRESHOLD:
                            logger.info(
                                f"🚫 Skipping {symbol}: correlation {corr:.2f} with open position {sym} exceeds {CORRELATION_THRESHOLD}"
                            )
                            skip_due_to_corr = True
                            break
                except Exception as e:
                    logger.warning(
                        f"Correlation check failed for {symbol}-{sym}: {e}"
                    )

        if not skip_due_to_corr:
            selected = (symbol, price, conf, coin_id, sig, label, mom_score)
            break

    if not selected:
        logger.error("❌ All trade candidates are too correlated with open positions.")
        logger.info("✅ Scan cycle complete\n" + "="*40)
        return

    best_symbol, best_price, best_conf, best_coin_id, _, best_label, best_mom_score = selected
    logger.info(f"\n🏆 BEST PICK: {best_symbol} BUY with confidence {best_conf:.2f} (label={best_label})")

    if not tm.positions:
        tm.open_trade(best_symbol, best_price, coin_id=best_coin_id, confidence=best_conf, label=best_label, side="BUY")
        tm.save_state()
        tm.summary()
        logger.info("✅ Scan cycle complete\n" + "="*40)
        return

    open_symbol = list(tm.positions.keys())[0]
    open_pos = tm.positions[open_symbol]
    open_conf = open_pos.get("confidence", 0.5)
    entry_price = open_pos["entry_price"]

    df_open = fetch_ohlcv_smart(open_symbol, coin_id=open_pos["coin_id"], days=10)
    df_open = add_indicators(df_open).dropna(subset=[
        "RSI", "MACD", "Signal", "Hist", "SMA_20", "SMA_50",
        "Return_1d", "Volatility_7d"
    ])

    if not df_open.empty:
        current_price = df_open["Close"].iloc[-1]
        movement = abs(current_price - entry_price) / entry_price
        is_stagnant = movement < STAGNATION_THRESHOLD
        open_momentum_score = df_open["Momentum_Score"].iloc[-1]

        if is_stagnant:
            recent_return = df_open["Return_3d"].iloc[-1]
            rsi = df_open["RSI"].iloc[-1]
            macd = df_open["MACD"].iloc[-1]
            signal_line = df_open["Signal"].iloc[-1]
            bullish_momentum = recent_return > 0.03 and rsi > 55 and macd > signal_line

            logger.info(f"📉 Stagnation check: return_3d={recent_return:.2%}, RSI={rsi:.1f}, MACD crossover={macd > signal_line}")
            if bullish_momentum:
                logger.info("🚫 Blocking rotation: current trade has bullish momentum despite price stagnation.")
                is_stagnant = False
    else:
        current_price = entry_price
        is_stagnant = True
        open_momentum_score = 0
        movement = 0

    logger.info(f"ℹ️ Open trade: {open_symbol} conf={open_conf:.2f}, stagnant={is_stagnant}")

    rotate_conf_gap = best_conf > open_conf + ROTATE_CONF_THRESHOLD
    rotate_mom_adv = best_mom_score > open_momentum_score + MOMENTUM_ADV_THRESHOLD

    if (
        best_symbol != open_symbol
        and is_stagnant
        and (rotate_conf_gap or rotate_mom_adv)
    ):
        qty = open_pos.get("qty", 0)
        entry_fee = open_pos.get("entry_fee", 0)
        exit_fee_est = current_price * qty * tm.trade_fee_pct
        gross_unrealized = (current_price - entry_price) * qty
        net_unrealized = gross_unrealized - entry_fee - exit_fee_est
        total_fees = entry_fee + exit_fee_est

        if net_unrealized <= 0:
            logger.info(
                f"🚫 Rotation blocked: unrealized PnL ${net_unrealized:.2f} does not cover fees ${total_fees:.2f}"
            )
            logger.info(
                f"✅ Keeping current trade {open_symbol} (conf={open_conf:.2f}) - insufficient profit for rotation."
            )
        else:
            if ENABLE_ROTATION_AUDIT:
                log_rotation_decision(
                    current={
                        "symbol": open_symbol,
                        "confidence": open_conf,
                        "label": open_pos.get("label"),
                        "entry_price": entry_price,
                        "movement": movement,
                    },
                    candidate={
                        "symbol": best_symbol,
                        "confidence": best_conf,
                        "label": best_label,
                        "price": best_price,
                    },
                )

                record_rotation_audit(
                    current={
                        "symbol": open_symbol,
                        "confidence": open_conf,
                        "label": open_pos.get("label"),
                        "entry_price": entry_price,
                        "movement": movement,
                    },
                    candidate={
                        "symbol": best_symbol,
                        "confidence": best_conf,
                        "label": best_label,
                        "price": best_price,
                    },
                    pnl_before=net_unrealized,
                )
                if PERSIST_ROTATION_AUDIT:
                    with ROTATION_AUDIT_LOCK:
                        save_rotation_audit()

            logger.info(
                f"💡 Rotation decision: {open_symbol} current=${current_price:.4f}, new pick={best_symbol} @${best_price:.4f}"
            )
            logger.info(f"🔄 Rotating {open_symbol} → {best_symbol}")
            closed = tm.close_trade(
                open_symbol,
                current_price,
                reason="Rotated to better candidate",
                candidate={
                    "symbol": best_symbol,
                    "price": best_price,
                    "confidence": best_conf,
                    "label": best_label,
                    "side": "BUY",
                },
            )
            if closed:
                tm.open_trade(
                    best_symbol,
                    best_price,
                    coin_id=best_coin_id,
                    confidence=best_conf,
                    label=best_label,
                    side="BUY",
                )
                tm.save_state()
                tm.summary()
            else:
                logger.info("❌ Rotation aborted: insufficient expected improvement")
    else:
        logger.info(
            f"✅ Keeping current trade {open_symbol} (conf={open_conf:.2f}) - no better candidate yet."
        )

    logger.info("✅ Scan cycle complete\n" + "="*40)


def main_loop():
    while True:
        try:
            clear_old_cache(cache_dir="cache", max_age=600)  # Purge cache >10 min old
            scan_for_breakouts()
        except Exception:
            logger.exception("Scan iteration failed")
        time.sleep(210)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-audit-file",
        action="store_true",
        help="Disable persistent rotation audit logging",
    )
    args = parser.parse_args()
    if args.no_audit_file:
        PERSIST_ROTATION_AUDIT = False
    try:
        main_loop()
    except Exception:
        logger.exception("❌ FATAL ERROR")
        input("Press Enter to exit...")
    finally:
        # Regenerate trade_stats.csv so future sessions pick up new blacklist data
        tm.summary()
        if ENABLE_ROTATION_AUDIT and PERSIST_ROTATION_AUDIT:
            with ROTATION_AUDIT_LOCK:
                save_rotation_audit()


