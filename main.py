import time 
import pandas as pd
import threading

# ✅ Now use refactored fetchers
from data_fetcher import get_top_gainers, fetch_ohlcv_smart, clear_old_cache
from feature_engineer import add_indicators, momentum_signal
from model_predictor import predict_signal
from trade_manager import TradeManager
from config import MOMENTUM_TIER_THRESHOLD, ERROR_DELAY, CONFIDENCE_THRESHOLD
from threshold_utils import get_dynamic_threshold
from utils.logging import get_logger

logger = get_logger(__name__)

# ✅ Global thresholds
# CONFIDENCE_THRESHOLD imported from config
ROTATE_CONF_THRESHOLD = 0.03        # new trade must have +3% higher confidence to rotate
STAGNATION_THRESHOLD = 0.01         # <1% price movement = stagnation
ROTATION_AUDIT_LOG = []  # 📘 In-memory rotation history
ROTATION_LOG_LIMIT = 10  # How many to keep

# Momentum tier mapping for gating logic
TIER_RANKS = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3, "Tier 4": 4}

# ✅ Fallback momentum thresholds
FALLBACK_RSI_THRESHOLD = 55
FALLBACK_RETURN_3D_THRESHOLD = 0.03
FLAT_1D_THRESHOLD = 0.001
FLAT_3D_THRESHOLD = 0.003

# 🔍 Rotation audit logging (toggleable)
ENABLE_ROTATION_AUDIT = True

def log_rotation_decision(current, candidate):
    logger.info("\n🔁 ROTATION DECISION:")
    logger.info(f" - Current: {current['symbol']} | Conf {current['confidence']:.2f} | Label {current['label']} | Entry ${current['entry_price']:.4f} | Movement {current['movement']:.2%}")
    logger.info(f" - Candidate: {candidate['symbol']} | Conf {candidate['confidence']:.2f} | Label {candidate['label']} | Price ${candidate['price']:.4f}")

def record_rotation_audit(current, candidate):
    ROTATION_AUDIT_LOG.append({
        "timestamp": time.time(),
        "current": current,
        "candidate": candidate
    })
    if len(ROTATION_AUDIT_LOG) > ROTATION_LOG_LIMIT:
        ROTATION_AUDIT_LOG.pop(0)

# ✅ TradeManager instance
tm = TradeManager()
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

def scan_for_breakouts():
    logger.warning(f"⚠️ Currently open trades before scanning: {list(tm.positions.keys())}")

    movers = get_top_gainers(limit=15)
    if not movers:
        logger.error("❌ No valid gainers found on Coinbase — skipping scan.")
        return

    logger.info("\n🔥 TOP GAINERS:")
    for cid, sym, name, chg in movers:
        logger.info(f" - {name} ({sym}) {chg:.2f}%")

    candidates = []
    open_symbols = list(tm.positions.keys())
    suppressed, fallbacks = 0, 0

    for coin_id, symbol, name, _ in movers:
        if symbol in open_symbols:
            logger.info(f"⏭️ Skipping {symbol} (already open trade)")
            continue

        logger.info(f"\n🔎 Analyzing {name} ({symbol})...")
        # Fetch a wider OHLCV window to ensure enough data remains after indicator dropna
        df = fetch_ohlcv_smart(symbol, coin_id=coin_id, days=10, limit=200)

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
        if vol_7d < 1e-4:
            logger.warning(f"⚠️ Skipping {symbol}: 7d volatility too low ({vol_7d:.6f})")
            continue

        threshold = get_dynamic_threshold(vol_7d, base=CONFIDENCE_THRESHOLD)

        # ✅ NEW: Momentum filter
        momentum_tier = df["Momentum_Tier"].iloc[-1]
        momentum_score = df["Momentum_Score"].iloc[-1]
        if TIER_RANKS.get(momentum_tier, 4) > MOMENTUM_TIER_THRESHOLD:
            logger.error(f"❌ Skipping {symbol}: weak momentum ({momentum_tier}, score={momentum_score})")
            continue

        try:
            signal, confidence, label = predict_signal(df, threshold)
            logger.info(f"🤖 ML Signal: {signal} (conf={confidence:.2f}, label={label})")
            logger.info(f"🧠 Threshold: {threshold:.2f} (7d vol={vol_7d:.3f})")

            if label == 1 and confidence < 0.85:
                logger.info(f"🚫 Suppressing weak Class 1 pick: {symbol} (conf={confidence:.2f})")
                signal = "HOLD"
                suppressed += 1

            if label in [3, 4] and confidence >= 0.75:
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
            if label in [3, 4] and confidence >= 0.90:
                logger.info(f"🟢 High-conviction BUY override: {symbol} → label={label}, conf={confidence:.2f}")
                signal = "BUY"

            # === Suppress weak Class 1 signals ===
            elif label == 1 and confidence < 0.85:
                logger.info(f"🚫 Suppressing weak Class 1 pick: {symbol} (conf={confidence:.2f})")
                signal = "HOLD"
                suppressed += 1

            # === ML HOLD → fallback strategy ===
            if signal == "HOLD" or confidence < threshold:
                logger.warning(f"⚠️ No ML trigger → Using fallback momentum strategy for {symbol}")
                signal = momentum_signal(df)
                logger.info(f"📉 Fallback momentum signal: {signal}")
                label = 2
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
            label = 2
            fallbacks += 1

        last_price = df["Close"].iloc[-1]

        if signal == "BUY" and label in [3, 4] and last_price > 0:
            if confidence < CONFIDENCE_THRESHOLD:
                logger.error(
                    f"❌ Skipping {symbol}: confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"
                )
                continue
            candidates.append((symbol, last_price, confidence, coin_id, signal, label))
        else:
            logger.error(f"❌ Skipping {symbol}: signal={signal}, label={label}, conf={confidence:.2f}")

        # API calls are rate-limited within fetch_ohlcv_smart and other
        # data_fetcher utilities, so no additional per-symbol delay is
        # required here.

    logger.info(f"📉 {suppressed} suppressed | 🔄 {fallbacks} fallback-triggered")

    if not candidates:
        logger.error("\n❌ No trade candidates found.")
        logger.info("✅ Scan cycle complete\n" + "="*40)
        return

    # ✅ Select best BUY candidate
    best = max(candidates, key=lambda x: x[2])
    best_symbol, best_price, best_conf, best_coin_id, _, best_label = best
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

    logger.info(f"ℹ️ Open trade: {open_symbol} conf={open_conf:.2f}, stagnant={is_stagnant}")

    if (
        best_symbol != open_symbol and
        best_conf > open_conf + ROTATE_CONF_THRESHOLD and
        is_stagnant
    ):
        if ENABLE_ROTATION_AUDIT:
            log_rotation_decision(
                current={
                    "symbol": open_symbol,
                    "confidence": open_conf,
                    "label": open_pos.get("label"),
                    "entry_price": entry_price,
                    "movement": movement
                },
                candidate={
                    "symbol": best_symbol,
                    "confidence": best_conf,
                    "label": best_label,
                    "price": best_price
                }
            )

            # ✅ ADD THIS HERE
            record_rotation_audit(
                current={
                    "symbol": open_symbol,
                    "confidence": open_conf,
                    "label": open_pos.get("label"),
                    "entry_price": entry_price,
                    "movement": movement
                },
                candidate={
                    "symbol": best_symbol,
                    "confidence": best_conf,
                    "label": best_label,
                    "price": best_price
                }
            )

        logger.info(f"💡 Rotation decision: {open_symbol} current=${current_price:.4f}, new pick={best_symbol} @${best_price:.4f}")
        logger.info(f"🔄 Rotating {open_symbol} → {best_symbol}")
        tm.close_trade(open_symbol, current_price, reason="Rotated to better candidate")
        tm.open_trade(best_symbol, best_price, coin_id=best_coin_id, confidence=best_conf, label=best_label, side="BUY")
        tm.save_state()
        tm.summary()
    else:
        logger.info(f"✅ Keeping current trade {open_symbol} (conf={open_conf:.2f}) - no better candidate yet.")

    logger.info("✅ Scan cycle complete\n" + "="*40)


def main_loop():
    while True:
        clear_old_cache(cache_dir="cache", max_age=600)  # Purge cache >10 min old
        scan_for_breakouts()
        time.sleep(210)

if __name__ == "__main__":
    try:
        main_loop()
    except Exception:
        logger.exception("❌ FATAL ERROR")
        input("Press Enter to exit...")

