
import json
import os
import time

import numpy as np

from data_fetcher import fetch_live_price
from utils.prediction_class import PredictionClass
from analytics.performance import is_blacklisted, get_duration_bucket, get_avg_fee_ratio

from config import (
    ATR_MULT_SL,
    ATR_MULT_TP,
    SL_BUFFER_ATR_MULT,
    BREAKEVEN_BUFFER_MULT,
)

from config import (
    RISK_PER_TRADE,
    MIN_TRADE_USD,
    SLIPPAGE_PCT,
    FEE_PCT,
    MAX_DRAWDOWN_PCT,
    MAX_DAILY_LOSS_PCT,
    HOLDING_PERIOD_SECONDS,
    REVERSAL_CONF_DELTA,
    MIN_PROFIT_FEE_RATIO,  # Higher profit-to-fee requirement
    CONFIDENCE_THRESHOLD,
    BLACKLIST_REFRESH_SEC,
    MIN_HOLD_BUCKET,
    EARLY_EXIT_FEE_MULT,
    ALLOCATION_MAX_DD,
    ALLOCATION_MIN_FACTOR,

    INCLUDE_UNREALIZED_PNL,

    STAGNATION_THRESHOLD_PCT,
    STAGNATION_DURATION_SEC,

    TRAIL_VOL_MULT,
    ADAPTIVE_STAGNATION,
    STAGNATION_VOL_MULT,
    SYMBOL_PNL_THRESHOLD,
)

from utils.logging import get_logger

logger = get_logger(__name__)

# Order of duration buckets for comparison with ``MIN_HOLD_BUCKET``
DURATION_BUCKETS = ["<1m", "1-5m", "5-30m", "30m-2h", ">2h"]

# Max number of recent closed-trade PnL values to retain
PNL_HISTORY_LIMIT = 20


def _bucket_index(bucket: str) -> int:
    try:
        return DURATION_BUCKETS.index(bucket)
    except ValueError:
        return len(DURATION_BUCKETS)



def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


class TradeManager:
    STATE_FILE = "trade_manager_state.json"

    def __init__(self, starting_balance=500, max_allocation=0.20,
                 sl_pct=0.06, tp_pct=0.10, trade_fee_pct=FEE_PCT,
                 trail_pct=0.03, trail_atr_mult=None,
                 trail_vol_mult=TRAIL_VOL_MULT,
                 trail_activation_pct=5.0,
                 trail_activation_atr_mult=1.0,
                 atr_mult_sl=ATR_MULT_SL,
                 atr_mult_tp=ATR_MULT_TP,
                 sl_buffer_atr_mult=SL_BUFFER_ATR_MULT,
                 breakeven_buffer_mult=BREAKEVEN_BUFFER_MULT,
                 max_drawdown_pct=MAX_DRAWDOWN_PCT, max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
                 slippage_pct=SLIPPAGE_PCT,
                 hold_period_sec=HOLDING_PERIOD_SECONDS,
                 reverse_conf_delta=REVERSAL_CONF_DELTA,
                 min_profit_fee_ratio=MIN_PROFIT_FEE_RATIO,  # Enforce stricter profit threshold
                 trail_profit_fee_ratio=2.0,
                 exchange=None,
                 blacklist_refresh_sec=BLACKLIST_REFRESH_SEC,
                 min_hold_bucket=MIN_HOLD_BUCKET,
                 early_exit_fee_mult=EARLY_EXIT_FEE_MULT,
                 stagnation_threshold_pct=STAGNATION_THRESHOLD_PCT,
                 stagnation_duration_sec=STAGNATION_DURATION_SEC,
                 adaptive_stagnation=ADAPTIVE_STAGNATION,
                 stagnation_vol_mult=STAGNATION_VOL_MULT,
                 include_unrealized_pnl=INCLUDE_UNREALIZED_PNL,
                 symbol_pnl_threshold=SYMBOL_PNL_THRESHOLD):

        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.max_allocation = max_allocation
        self.stop_loss_pct = sl_pct
        self.take_profit_pct = tp_pct
        self.trade_fee_pct = trade_fee_pct
        self.trail_pct = trail_pct
        self.trail_atr_mult = trail_atr_mult
        self.trail_vol_mult = trail_vol_mult
        self.trail_activation_pct = trail_activation_pct
        self.trail_activation_atr_mult = trail_activation_atr_mult
        self.atr_mult_sl = atr_mult_sl
        self.atr_mult_tp = atr_mult_tp
        self.sl_buffer_atr_mult = sl_buffer_atr_mult
        self.breakeven_buffer_mult = breakeven_buffer_mult
        self.slippage_pct = slippage_pct
        self.min_profit_fee_ratio = min_profit_fee_ratio
        self.trail_profit_fee_ratio = trail_profit_fee_ratio
        self.blacklist_refresh_sec = blacklist_refresh_sec
        self.min_hold_bucket = min_hold_bucket
        self.early_exit_fee_mult = early_exit_fee_mult
        self.stagnation_threshold_pct = stagnation_threshold_pct
        self.stagnation_duration_sec = stagnation_duration_sec
        self.adaptive_stagnation = adaptive_stagnation
        self.stagnation_vol_mult = stagnation_vol_mult
        self.symbol_pnl_threshold = symbol_pnl_threshold

        # Holding period and reversal requirements
        # Add a small buffer to the minimum hold time to better absorb
        # slippage-induced execution delays.
        extra_hold = max(1, int(hold_period_sec * self.slippage_pct)) if hold_period_sec > 0 else 0
        self.min_hold_time = hold_period_sec + extra_hold
        self.reverse_conf_delta = reverse_conf_delta
        self.last_trade_time = 0.0
        self.last_trade_side = None
        self.last_trade_confidence = None

        # Risk parameters
        self.risk_per_trade = RISK_PER_TRADE
        self.min_trade_usd = MIN_TRADE_USD
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct

        # Optional exchange adapter (paper or live trading)
        self.exchange = exchange

        # Whether to factor open trade PnL into equity calculations
        self.include_unrealized_pnl = include_unrealized_pnl

        # Equity tracking
        self.peak_equity = starting_balance
        self.drawdown_pct = 0.0
        self.current_day = time.strftime("%Y-%m-%d")
        self.daily_start_balance = starting_balance
        self.daily_loss_pct = 0.0

        self.positions = {}
        self.trade_history = []
        self.total_fees = 0.0

        # Recent closed-trade PnL history
        self.closed_pnl_history = []

        # Cumulative realized PnL per symbol
        self.symbol_pnl = {}

    def has_position(self, symbol):
        return symbol in self.positions

    def fmt_price(self, p):
        return f"{p:.6f}" if p < 1 else f"{p:.2f}"

    def calculate_allocation(self, confidence=1.0):
        """Determine trade size based on current balance and risk settings.

        Allocation is derived from a fixed fraction of equity (``risk_per_trade``)
        and optionally scaled by the model's confidence score to favor
        higher-conviction entries.  Recent loss streaks further reduce the
        allocation to slow down trading after consecutive losing trades.
        """
        base = self.balance * self.risk_per_trade
        if confidence is not None:
            base *= confidence

        # Apply a penalty for consecutive losing trades based on recent
        # closed-trade PnL history.  Count the number of negative PnL values
        # starting from the most recent trade and scale the allocation
        # accordingly.
        loss_streak = 0
        for pnl in reversed(self.closed_pnl_history):
            if pnl < 0:
                loss_streak += 1
            else:
                break

        if loss_streak >= 3:
            base *= 0.6
        elif loss_streak >= 2:
            base *= 0.8

        # Scale allocation based on current drawdown
        self._update_equity_metrics()
        dd = self.drawdown_pct
        if dd <= 0:
            factor = 1.0
        else:
            factor = max(
                ALLOCATION_MIN_FACTOR,
                1 - (dd / ALLOCATION_MAX_DD) * (1 - ALLOCATION_MIN_FACTOR),
            )

        return base * factor

    def _estimate_rotation_gain(self, symbol, price, confidence, side="BUY", balance_override=None):
        """Estimate net profit of a prospective trade for rotation decisions.

        The minimum profit-to-fee threshold is scaled by the symbol's historical
        fee ratio to avoid rotating into pairs with consistently high fees.
        """
        base_balance = balance_override if balance_override is not None else self.balance
        allocation = base_balance * self.risk_per_trade
        if confidence is not None:
            allocation *= confidence
        if allocation <= 0:
            return 0.0
        entry_fee_est = allocation * self.trade_fee_pct
        net_alloc_est = allocation - entry_fee_est
        if side == "BUY":
            est_exec_price = price * (1 + self.slippage_pct)
            est_qty = net_alloc_est / est_exec_price
            est_take_profit = (
                est_exec_price * (1 + self.take_profit_pct) * (1 + self.trade_fee_pct)
            )
            est_exit_fee = est_take_profit * est_qty * self.trade_fee_pct
            expected_profit = (est_take_profit - est_exec_price) * est_qty
        else:
            est_exec_price = price * (1 - self.slippage_pct)
            est_qty = net_alloc_est / est_exec_price
            est_take_profit = (
                est_exec_price * (1 - self.take_profit_pct) * (1 - self.trade_fee_pct)
            )
            est_exit_fee = est_take_profit * est_qty * self.trade_fee_pct
            expected_profit = (est_exec_price - est_take_profit) * est_qty

        total_est_fees = entry_fee_est + est_exit_fee
        profit_fee_ratio = (
            expected_profit / total_est_fees if total_est_fees > 0 else float("inf")
        )
        hist_fee_ratio = get_avg_fee_ratio(symbol, self.min_hold_bucket)
        adjusted_threshold = self.min_profit_fee_ratio * (1 + hist_fee_ratio)
        logger.info(
            "Adjusted min_profit_fee_ratio for %s: %.2f (fee_ratio %.2f)",
            symbol,
            adjusted_threshold,
            hist_fee_ratio,
        )
        if profit_fee_ratio < adjusted_threshold:
            return 0.0

        return expected_profit - total_est_fees

    def _update_equity_metrics(self):
        """Recalculate drawdown and daily loss percentages."""
        equity = self.balance
        if self.include_unrealized_pnl and self.positions:
            for symbol, pos in self.positions.items():
                price = fetch_live_price(symbol, pos.get("coin_id"))
                if price is None or price <= 0:
                    continue
                qty = pos.get("qty", 0)
                entry = pos.get("entry_price", 0)
                if pos.get("side") == "SELL":
                    unrealized = (entry - price) * qty
                else:
                    unrealized = (price - entry) * qty
                equity += unrealized

        today = time.strftime("%Y-%m-%d")
        if self.current_day != today:
            self.current_day = today
            self.daily_start_balance = equity
            self.daily_loss_pct = 0.0

        # In realized-only mode, defer drawdown updates until all trades closed
        if not self.include_unrealized_pnl and self.positions:
            return

        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.peak_equity > 0:
            self.drawdown_pct = max(0.0, (self.peak_equity - equity) / self.peak_equity)

        if self.daily_start_balance > 0:
            self.daily_loss_pct = max(
                0.0, (self.daily_start_balance - equity) / self.daily_start_balance
            )

    def can_trade(self):
        """Return True if drawdown and daily loss are within limits."""
        # Refresh metrics to account for new day boundaries
        self._update_equity_metrics()
        if self.drawdown_pct >= self.max_drawdown_pct:
            return False
        if self.daily_loss_pct >= self.max_daily_loss_pct:
            return False
        return True

    def _compute_trail_offset(self, pos, current_price):
        """Return trailing stop distance using ATR or volatility scaling.

        Trailing stops should only activate on trades that are net profitable
        after accounting for fees *and* have moved sufficiently in our favor.
        If the current unrealized PnL does not exceed the estimated fees or
        the price move fails to clear a profit/ATR threshold, ``None`` is
        returned to signal that a trailing stop should not yet be applied.

        If ``trail_atr_mult`` is set and an ATR value is available, the
        distance is ``ATR * trail_atr_mult``.  Otherwise fall back to the
        percentage-based distance ``current_price * trail_pct`` but adjust it
        by recent price volatility (standard deviation) if enough data exists.
        This allows the trailing stop to widen during volatile periods and
        tighten when price action is quiet.
        """

        entry = pos.get("entry_price")
        qty = pos.get("qty", 0)
        side = pos.get("side", "BUY")
        if entry is not None and qty > 0:
            if side == "BUY":
                unrealized = (current_price - entry) * qty
                price_move = current_price - entry
            else:
                unrealized = (entry - current_price) * qty
                price_move = entry - current_price
            est_exit_fee = current_price * qty * self.trade_fee_pct
            total_fees = pos.get("entry_fee", 0) + est_exit_fee
            atr_val = pos.get("atr")
            atr_thresh = atr_val * self.trail_activation_atr_mult if atr_val else 0
            profit_thresh = entry * self.trail_activation_pct / 100
            if unrealized <= total_fees or price_move < max(profit_thresh, atr_thresh):
                return None

        if self.trail_atr_mult:
            atr_val = pos.get("atr")
            if not atr_val:
                prices = pos.get("recent_prices", [])
                if len(prices) >= 2:
                    atr_val = float(np.std(prices))
            if atr_val and atr_val > 0:
                prices = pos.get("recent_prices", [])
                vol_pct = 0.0
                if len(prices) >= 2 and current_price > 0:
                    vol_pct = float(np.std(prices)) / current_price
                mult = max(1.0, vol_pct * self.trail_vol_mult)
                return atr_val * self.trail_atr_mult * mult

        prices = pos.get("recent_prices", [])
        if len(prices) >= 2 and current_price > 0:
            vol = float(np.std(prices))
            vol_pct = vol / current_price
            dynamic_pct = max(self.trail_pct, vol_pct * self.trail_vol_mult)
            return current_price * dynamic_pct

        return current_price * self.trail_pct

    def _compute_stagnation_params(self, pos, current_price):
        """Return (threshold_pct, duration_sec) accounting for volatility."""
        threshold = self.stagnation_threshold_pct
        duration = self.stagnation_duration_sec
        if self.adaptive_stagnation:
            prices = pos.get("recent_prices", [])
            if len(prices) >= 2 and current_price > 0:
                vol_pct = float(np.std(prices)) / current_price
                mult = 1 + vol_pct * self.stagnation_vol_mult
                threshold *= mult
                duration *= mult
        return threshold, duration

    def open_trade(self, symbol, price, coin_id=None, confidence=0.5,
                   label=None, side="BUY", atr=None):
        if not self.can_trade():
            logger.warning("🚫 Risk limits exceeded — cannot open new trades.")
            return

        now = time.time()
        if self.min_hold_time > 0 and now - self.last_trade_time < self.min_hold_time:
            remaining = self.min_hold_time - (now - self.last_trade_time)
            logger.info(
                f"⏳ Holding period active — skipping trade for {symbol} ({remaining:.0f}s remaining)"
            )
            return

        if confidence is None or confidence < CONFIDENCE_THRESHOLD:
            conf_val = confidence if confidence is not None else float("nan")
            logger.info(
                "🔮 Skipping %s: confidence %.2f below threshold %.2f",
                symbol,
                conf_val,
                CONFIDENCE_THRESHOLD,
            )
            return

        if (
            self.reverse_conf_delta > 0
            and self.last_trade_side
            and side != self.last_trade_side
            and self.last_trade_confidence is not None
            and confidence is not None
            and confidence - self.last_trade_confidence < self.reverse_conf_delta
        ):
            logger.info(
                "🔁 Skipping reversal trade for %s: confidence delta %.2f < %.2f",
                symbol,
                confidence - self.last_trade_confidence,
                self.reverse_conf_delta,
            )
            return

        duration_bucket = self.min_hold_bucket
        # `is_blacklisted` now also screens out combinations where trading fees
        # have historically outweighed profits via the fee-ratio rule.
        if is_blacklisted(symbol, duration_bucket, refresh_seconds=self.blacklist_refresh_sec):
            logger.info(
                "🚫 Skipping %s: blacklisted for bucket %s",
                symbol,
                duration_bucket,
            )
            return

        if (
            self.symbol_pnl_threshold is not None
            and self.symbol_pnl.get(symbol, 0.0) < self.symbol_pnl_threshold
        ):
            logger.info(
                "🚫 Skipping %s: cumulative PnL $%.2f below threshold $%.2f",
                symbol,
                self.symbol_pnl.get(symbol, 0.0),
                self.symbol_pnl_threshold,
            )
            return

        if self.has_position(symbol):
            logger.warning(f"⚠️ Already have position in {symbol}")
            return

        allocation = self.calculate_allocation(confidence)
        if allocation > self.balance:
            logger.warning(
                f"⚠️ Insufficient margin for {symbol}: required ${allocation:.2f} but only ${self.balance:.2f} available, skipping",
            )
            return

        if allocation < self.min_trade_usd:

            logger.warning(
                f"💸 Allocation ${allocation:.2f} below minimum {self.min_trade_usd} for {symbol}, skipping",
            )
            return

        atr_val = atr
        if atr_val is None:
            try:
                from data_fetcher import fetch_ohlcv_smart
                from feature_engineer import add_indicators

                df = fetch_ohlcv_smart(symbol, coin_id=coin_id, days=10)
                df = add_indicators(df)
                if "ATR" in df.columns and not df["ATR"].dropna().empty:
                    atr_val = float(df["ATR"].iloc[-1])
            except Exception as e:
                logger.warning(f"⚠️ Could not compute ATR for {symbol}: {e}")

        entry_fee_est = allocation * self.trade_fee_pct
        net_alloc_est = allocation - entry_fee_est
        if side == "BUY":
            est_exec_price = price * (1 + self.slippage_pct)
        else:
            est_exec_price = price * (1 - self.slippage_pct)
        est_qty = net_alloc_est / est_exec_price

        if atr_val and atr_val > 0:
            tp_offset = self.atr_mult_tp * atr_val
            if side == "SELL":
                est_take_profit = (
                    est_exec_price - tp_offset
                ) * (1 - self.trade_fee_pct)
            else:
                est_take_profit = (
                    est_exec_price + tp_offset
                ) * (1 + self.trade_fee_pct)
        else:
            tp_pct = self.take_profit_pct
            if side == "SELL":
                est_take_profit = (
                    est_exec_price * (1 - tp_pct) * (1 - self.trade_fee_pct)
                )
            else:
                est_take_profit = (
                    est_exec_price * (1 + tp_pct) * (1 + self.trade_fee_pct)
                )

        est_exit_fee = est_take_profit * est_qty * self.trade_fee_pct
        total_est_fees = entry_fee_est + est_exit_fee
        if side == "SELL":
            expected_profit = (est_exec_price - est_take_profit) * est_qty
        else:
            expected_profit = (est_take_profit - est_exec_price) * est_qty

        if expected_profit <= 0:
            logger.info(
                "📉 Skipping %s: non-positive expected profit ($%.2f)",
                symbol,
                expected_profit,
            )
            return

        profit_fee_ratio = (
            expected_profit / total_est_fees if total_est_fees > 0 else float("inf")
        )
        if profit_fee_ratio < self.min_profit_fee_ratio:
            logger.info(
                "💰 Trade rejected for %s: expected profit $%.2f vs fees $%.2f (ratio %.2f < %.2f)",
                symbol,
                expected_profit,
                total_est_fees,
                profit_fee_ratio,
                self.min_profit_fee_ratio,
            )
            return

        order = None
        try:
            if self.exchange:
                order = self.exchange.place_market_order(
                    symbol, side, quote_quantity=allocation
                )
                exec_price = order.get("price", price)
                qty = order.get("executed_qty", 0)
                entry_cost = exec_price * qty
                entry_fee = entry_cost * self.trade_fee_pct
                allocation = entry_cost + entry_fee
            else:
                entry_fee = allocation * self.trade_fee_pct
                net_allocation = allocation - entry_fee
                if side == "BUY":
                    exec_price = price * (1 + self.slippage_pct)
                else:
                    exec_price = price * (1 - self.slippage_pct)
                qty = net_allocation / exec_price
        except Exception as e:
            logger.error(f"❌ Order placement failed for {symbol}: {e}")
            entry_fee = allocation * self.trade_fee_pct
            net_allocation = allocation - entry_fee
            if side == "BUY":
                exec_price = price * (1 + self.slippage_pct)
            else:
                exec_price = price * (1 - self.slippage_pct)
            qty = net_allocation / exec_price

        entry_slippage_pct = abs(exec_price - price) / price if price else 0

        if qty < 0.0001:

            logger.warning(f"⚠️ Trade qty too small for {symbol}, skipping")
            return

        if label == PredictionClass.BIG_LOSS.value and side == "BUY":
            logger.warning(
                f"⚠️ Model predicts loss for {symbol} (label {PredictionClass.BIG_LOSS.value}), skipping long trade."
            )
            return


        self.balance -= allocation
        self.total_fees += entry_fee

        if atr_val and atr_val > 0:
            sl_offset = self.atr_mult_sl * atr_val + exec_price * self.slippage_pct
            tp_offset = self.atr_mult_tp * atr_val
            if side == "SELL":
                stop_loss = (exec_price + sl_offset) * (1 + self.trade_fee_pct)
                take_profit = (exec_price - tp_offset) * (1 - self.trade_fee_pct)
            else:
                stop_loss = (exec_price - sl_offset) * (1 - self.trade_fee_pct)
                take_profit = (exec_price + tp_offset) * (1 + self.trade_fee_pct)
        else:

            logger.warning(f"⚠️ ATR unavailable for {symbol}; falling back to percentage-based SL/TP.")

            tp_pct = self.take_profit_pct
            sl_pct = self.stop_loss_pct + self.slippage_pct
            if side == "SELL":
                stop_loss = exec_price * (1 + sl_pct) * (1 + self.trade_fee_pct)
                take_profit = exec_price * (1 - tp_pct) * (1 - self.trade_fee_pct)
            else:
                stop_loss = exec_price * (1 - sl_pct) * (1 - self.trade_fee_pct)
                take_profit = exec_price * (1 + tp_pct) * (1 + self.trade_fee_pct)

        self.positions[symbol] = {
            "coin_id": coin_id or symbol.lower(),
            "entry_price": exec_price,
            "qty": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_fee": entry_fee,
            "highest_price": exec_price,
            "confidence": confidence,
            "label": label,
            "side": side,
            "entry_time": now,
            "last_movement_time": now,
            "atr": atr_val,
            "order_id": order.get("order_id") if order else None,
            "entry_slippage_pct": entry_slippage_pct,
        }

        msg = (
            f"🚀 OPEN {side.upper()} {symbol}: qty={qty:.4f} @ ${self.fmt_price(exec_price)} | "
            f"Slippage {entry_slippage_pct * 100:.2f}% | "
            f"Allocated ${allocation:.2f} (fee ${entry_fee:.2f}) | Balance left ${self.balance:.2f} | "
            f"Label={label}"
        )
        if atr_val:
            msg += f" | ATR={atr_val:.6f}"
        logger.info(msg)

        self.last_trade_time = now
        self.last_trade_side = side
        self.last_trade_confidence = confidence

    def close_trade(self, symbol, current_price, reason="", candidate=None):
        if not self.has_position(symbol):
            logger.warning(f"⚠️ No open position to close for {symbol}")
            return False

        pos = self.positions[symbol]
        elapsed = time.time() - pos.get("entry_time", 0)
        sl_tp_reasons = {
            "Stop-Loss",
            "Take-Profit",
            "Trailing Stop",
            "Take Profit Hit",
        }
        if elapsed < self.min_hold_time and reason in sl_tp_reasons:
            logger.info(
                f"⏱️ Minimum hold time not met for {symbol} ({elapsed:.0f}s < {self.min_hold_time}s). Skipping close."
            )
            return False

        # Verify profitability before closing
        qty = pos.get("qty", 0)
        entry_price = pos["entry_price"]
        side = pos.get("side", "BUY")
        entry_fee = pos.get("entry_fee", 0)
        exit_fee_est = current_price * qty * self.trade_fee_pct
        if side == "BUY":
            expected_profit = (current_price - entry_price) * qty
        else:
            expected_profit = (entry_price - current_price) * qty
        total_fees = entry_fee + exit_fee_est
        profit_fee_ratio = (
            expected_profit / total_fees if total_fees > 0 else float("inf")
        )
        duration_bucket = get_duration_bucket(elapsed)
        if _bucket_index(duration_bucket) < _bucket_index(self.min_hold_bucket) and profit_fee_ratio < self.early_exit_fee_mult:
            logger.info(
                f"⏱️ Hold bucket {duration_bucket} below {self.min_hold_bucket} for {symbol}. "
                f"Profit/fee ratio {profit_fee_ratio:.2f} < {self.early_exit_fee_mult:.2f}. Skipping close."
            )
            return False
        if (
            expected_profit > 0
            and total_fees > 0
            and profit_fee_ratio < self.min_profit_fee_ratio
            and reason != "Stop-Loss"
        ):
            logger.info(
                "💰 Exit for %s rejected: expected profit $%.2f vs fees $%.2f (ratio %.2f < %.2f)",
                symbol,
                expected_profit,
                total_fees,
                profit_fee_ratio,
                self.min_profit_fee_ratio,
            )
            return False

        rotation_projected_gain = None
        rotation_cost_est = None
        rotation_net_gain_est = None
        if reason == "Rotated to better candidate" and candidate:
            rotation_projected_gain = self._estimate_rotation_gain(
                candidate.get("symbol"),
                candidate.get("price"),
                candidate.get("confidence", 1.0),
                candidate.get("side", "BUY"),
                balance_override=self.balance + current_price * qty,
            )
            if side == "BUY":
                raw_loss = max(0, (entry_price - current_price) * qty)
            else:
                raw_loss = max(0, (current_price - entry_price) * qty)
            rotation_cost_est = raw_loss + entry_fee + exit_fee_est
            rotation_net_gain_est = rotation_projected_gain - rotation_cost_est
            logger.info(
                "🔄 Rotation check: projected gain $%.2f vs cost $%.2f => net $%.2f",
                rotation_projected_gain,
                rotation_cost_est,
                rotation_net_gain_est,
            )
            if rotation_net_gain_est <= 0:
                logger.info(
                    "❌ Rotation aborted: net gain $%.2f <= 0",
                    rotation_net_gain_est,
                )
                return False

        pos = self.positions.pop(symbol)
        side = pos.get("side", "BUY")

        if pos["qty"] <= 0:
            logger.warning(f"❌ Invalid quantity for {symbol}, skipping close.")
            return False

        qty = pos["qty"]
        exit_order = None
        if self.exchange:
            exit_side = "SELL" if side == "BUY" else "BUY"
            try:
                exit_order = self.exchange.place_market_order(
                    symbol, exit_side, quantity=qty
                )
                exec_price = exit_order.get("price", current_price)
                qty = exit_order.get("executed_qty", qty)
            except Exception as e:
                logger.error(f"❌ Close order failed for {symbol}: {e}")
                if side == "BUY":
                    exec_price = current_price * (1 - self.slippage_pct)
                else:
                    exec_price = current_price * (1 + self.slippage_pct)
        else:
            if side == "BUY":
                exec_price = current_price * (1 - self.slippage_pct)
            else:
                exec_price = current_price * (1 + self.slippage_pct)

        slippage_pct = abs(exec_price - current_price) / current_price if current_price else 0

        entry_val = pos["entry_price"] * qty
        exit_val = exec_price * qty

        exit_fee = exit_val * self.trade_fee_pct
        net_exit = exit_val - exit_fee
        self.total_fees += exit_fee

        pnl = net_exit - (entry_val + pos.get("entry_fee", 0))
        self.balance += net_exit

        # Track cumulative PnL per symbol
        self.symbol_pnl[symbol] = self.symbol_pnl.get(symbol, 0.0) + pnl

        self.last_trade_time = time.time()

        # Update drawdown and daily loss metrics after realizing PnL
        self._update_equity_metrics()

        duration = time.time() - pos.get("entry_time", time.time())

        rotation_cost_actual = None
        if reason == "Rotated to better candidate":
            rotation_cost_actual = max(0, -pnl)

        # ✅ Exit momentum capture BEFORE building trade_record
        exit_momentum = {}
        try:
            from feature_engineer import add_indicators
            from data_fetcher import fetch_ohlcv_smart

            df = fetch_ohlcv_smart(symbol, coin_id=pos.get("coin_id"), days=10)
            df = add_indicators(df).dropna(subset=["Return_1d", "Return_3d", "RSI", "Hist"])
            if not df.empty:
                row = df.iloc[-1]
                exit_momentum = {
                    "return_1d": row["Return_1d"],
                    "return_3d": row["Return_3d"],
                    "rsi": row["RSI"],
                    "macd_hist": row["Hist"]
                }
        except Exception as e:
            logger.warning(f"⚠️ Could not capture exit momentum for {symbol}: {e}")

        # ✅ Build trade record AFTER collecting momentum
        trade_record = {
            "symbol": symbol,
            "entry_price": pos["entry_price"],
            "exit_price": exec_price,
            "qty": pos["qty"],
            "pnl": pnl,
            "reason": reason,
            "entry_fee": pos.get("entry_fee", 0),
            "exit_fee": exit_fee,
            "confidence": pos.get("confidence"),
            "label": pos.get("label"),
            "trail_triggered": reason == "Trailing Stop",
            "duration": round(duration, 2),
            "exit_momentum": exit_momentum,
            "order_id": pos.get("order_id"),
            "exit_order_id": exit_order.get("order_id") if exit_order else None,
            "entry_slippage_pct": pos.get("entry_slippage_pct"),
            "exit_slippage_pct": slippage_pct,
        }

        if reason == "Rotated to better candidate":
            trade_record["rotation_exit_price"] = current_price
            if rotation_projected_gain is not None:
                trade_record["rotation_projected_gain"] = rotation_projected_gain
                trade_record["rotation_cost"] = rotation_cost_actual
                trade_record["rotation_net_gain"] = rotation_projected_gain - rotation_cost_actual
                trade_record["rotation_candidate"] = candidate.get("symbol") if candidate else None

        self.trade_history.append(trade_record)

        # Maintain short history of closed-trade PnL
        self.closed_pnl_history.append(pnl)
        if len(self.closed_pnl_history) > PNL_HISTORY_LIMIT:
            self.closed_pnl_history.pop(0)

        logger.info(
            f"🔐 CLOSE {symbol} | Exit ${self.fmt_price(exec_price)} | "
            f"PnL: ${pnl:.2f} | Fee ${exit_fee:.2f} | "
            f"Slippage {slippage_pct * 100:.2f}% | Duration {duration:.0f}s | "
            f"Balance now ${self.balance:.2f}"
        )
        self.save_state()

        return True


    def monitor_open_trades(self, check_interval=60, single_run=False):
        logger.info(f"🚦 Monitoring {len(self.positions)} open trade(s)...")

        if not self.positions:
            self.load_state()
            logger.info(f"🔄 Reloaded state → {len(self.positions)} positions")

        def check_once():
            for symbol, pos in list(self.positions.items()):
                coin_id = pos.get("coin_id", symbol.lower())
                last_price = fetch_live_price(symbol, coin_id)
                if last_price is None or last_price <= 0:
                    logger.warning(f"⚠️ Could not fetch live price for {symbol}, skipping this check.")
                    continue

                entry_price = pos["entry_price"]
                logger.info(
                    f"🔄 Monitoring {symbol}: entry ${self.fmt_price(entry_price)} | current ${self.fmt_price(last_price)}"
                )

                if last_price < entry_price * 0.01:
                    logger.warning(
                        f"⚠️ Suspicious price ({last_price}), skipping to prevent false SL trigger."
                    )
                    continue

                recent = pos.setdefault("recent_prices", [])
                recent.append(last_price)
                if len(recent) > 20:
                    recent.pop(0)

                price_change = abs(last_price - entry_price) / entry_price
                threshold, duration = self._compute_stagnation_params(pos, last_price)
                if price_change >= threshold:
                    pos["last_movement_time"] = time.time()

                # ✅ Hybrid profit protection
                pnl_pct = (last_price - entry_price) / entry_price * 100
                elapsed = time.time() - pos.get("entry_time", 0)

                if elapsed < self.min_hold_time:
                    logger.debug(
                        f"⏱️ Holding period active for {symbol} ({elapsed:.0f}s < {self.min_hold_time}s)"
                    )
                else:
                    atr_val = pos.get("atr")
                    atr_trigger_pct = (
                        (atr_val / last_price) * 100 * self.trail_activation_atr_mult
                        if atr_val and last_price > 0
                        else 0
                    )
                    activation_pct = max(self.trail_activation_pct, atr_trigger_pct)
                    # Hard take-profit at 10–12%
                    if pnl_pct >= 10:
                        logger.info(
                            f"🎯 Take-profit hit: {symbol} is up {pnl_pct:.2f}% — closing trade."
                        )
                        self.close_trade(symbol, last_price, reason="Take Profit Hit")
                        continue  # skip further checks on this trade

                    # Trailing stop trigger using dynamic gain/ATR confirmation
                    elif pnl_pct >= activation_pct:
                        qty = pos.get("qty", 0)
                        side = pos.get("side", "BUY")
                        if qty > 0:
                            if side == "BUY":
                                unrealized = (last_price - entry_price) * qty
                            else:
                                unrealized = (entry_price - last_price) * qty
                            est_exit_fee = last_price * qty * self.trade_fee_pct
                            total_fees = pos.get("entry_fee", 0) + est_exit_fee
                        else:
                            unrealized = 0
                            total_fees = pos.get("entry_fee", 0)

                        if unrealized <= total_fees:
                            logger.debug(
                                f"📉 PnL ${unrealized:.2f} has not covered fees ${total_fees:.2f} for {symbol}"
                            )
                            continue

                        if side == "BUY":
                            current_sl = pos.get("stop_loss", 0)
                            if current_sl < entry_price:
                                pos["stop_loss"] = entry_price
                                logger.info(
                                    f"🛡️ Fees covered for {symbol}; moved stop loss to entry {self.fmt_price(entry_price)}"
                                )
                                self.save_state()
                        else:
                            current_sl = pos.get("stop_loss", 0)
                            if current_sl > entry_price:
                                pos["stop_loss"] = entry_price
                                logger.info(
                                    f"🛡️ Fees covered for {symbol}; moved stop loss to entry {self.fmt_price(entry_price)}"
                                )
                                self.save_state()

                        threshold = self.trail_profit_fee_ratio * total_fees
                        if total_fees > 0 and unrealized < threshold:
                            logger.debug(
                                f"📉 PnL ${unrealized:.2f} below trailing threshold ${threshold:.2f} for {symbol}"
                            )
                            continue

                        if not pos.get("trail_triggered"):
                            offset = self._compute_trail_offset(pos, last_price)
                            if offset is None:
                                continue
                            pos["trail_triggered"] = True
                            if side == "BUY":
                                pos["trail_price"] = last_price - offset
                            else:
                                pos["trail_price"] = last_price + offset
                            logger.info(
                                f"📉 Trailing stop activated for {symbol} at {self.fmt_price(last_price)} (PnL ${unrealized:.2f} vs fees ${total_fees:.2f})"
                            )
                        else:
                            hit = (
                                last_price < pos.get("trail_price", 0)
                                if side == "BUY"
                                else last_price > pos.get("trail_price", 0)
                            )
                            if hit:
                                logger.info(f"🔻 Trailing stop hit for {symbol} — closing trade.")
                                self.close_trade(symbol, last_price, reason="Trailing Stop")
                                continue
                            offset = self._compute_trail_offset(pos, last_price)
                            if offset is None:
                                continue
                            if side == "BUY":
                                new_trail = last_price - offset
                                if new_trail > pos["trail_price"]:
                                    logger.info(
                                        f"🔼 Updating trail stop for {symbol}: {pos['trail_price']:.4f} → {new_trail:.4f}"
                                    )
                                    pos["trail_price"] = new_trail
                            else:
                                new_trail = last_price + offset
                                if new_trail < pos["trail_price"]:
                                    logger.info(
                                        f"🔽 Updating trail stop for {symbol}: {pos['trail_price']:.4f} → {new_trail:.4f}"
                                    )
                                    pos["trail_price"] = new_trail

                self.manage(symbol, last_price)

            self.summary()
            self.save_state()

        if single_run:
            check_once()
            return

        while self.positions:
            check_once()
            logger.info(f"⏳ Sleeping {check_interval} seconds before next check...\n")
            time.sleep(check_interval)

        logger.info("✅ No more open trades. Exiting monitoring.")

    def manage(self, symbol, current_price):
        if not self.has_position(symbol):
            return

        pos = self.positions[symbol]
        side = pos.get("side", "BUY")
        entry_price = pos["entry_price"]
        highest_price = pos["highest_price"]

        if side == "BUY":
            if current_price > highest_price:
                pos["highest_price"] = current_price
        else:
            if current_price < highest_price:
                pos["highest_price"] = current_price

        now = time.time()
        price_change = abs(current_price - entry_price) / entry_price
        threshold, duration = self._compute_stagnation_params(pos, current_price)
        if price_change >= threshold:
            pos["last_movement_time"] = now

        # Respect minimum holding period before evaluating exits
        elapsed = now - pos.get("entry_time", 0)
        if elapsed < self.min_hold_time:
            logger.debug(
                f"⏱️ Holding period active for {symbol} ({elapsed:.0f}s < {self.min_hold_time}s)"
            )
            return

        if price_change < threshold:
            stagnant_for = now - pos.get("last_movement_time", pos.get("entry_time", now))
            if stagnant_for >= duration:
                logger.info(
                    f"🔕 Price stagnation exit for {symbol}: below {threshold * 100:.2f}% for {stagnant_for:.0f}s"
                )
                self.close_trade(symbol, current_price, reason="Stagnant Price")
                return

        trail_stop = None
        atr = pos.get("atr")
        if atr and atr > 0:
            if side == "BUY" and pos["highest_price"] > entry_price:
                trail_stop = pos["highest_price"] - atr * self.atr_mult_sl
            elif side == "SELL" and pos["highest_price"] < entry_price:
                trail_stop = pos["highest_price"] + atr * self.atr_mult_sl
        else:
            if side == "BUY" and pos["highest_price"] > entry_price:
                trail_stop = pos["highest_price"] * (1 - self.trail_pct)
            elif side == "SELL" and pos["highest_price"] < entry_price:
                trail_stop = pos["highest_price"] * (1 + self.trail_pct)

        atr_val = pos.get("atr")
        if atr_val and atr_val > 0:
            sl_buffer = atr_val * self.sl_buffer_atr_mult
        else:
            prices = pos.get("recent_prices", [])
            sl_buffer = 0.0
            if len(prices) >= 2:
                sl_buffer = float(np.std(prices)) * self.sl_buffer_atr_mult

        entry_price = pos.get("entry_price")
        if entry_price is not None:
            if side == "BUY" and pos["stop_loss"] >= entry_price:
                sl_buffer *= self.breakeven_buffer_mult
            elif side == "SELL" and pos["stop_loss"] <= entry_price:
                sl_buffer *= self.breakeven_buffer_mult

        if side == "BUY":
            if current_price < pos["stop_loss"] - sl_buffer:
                logger.warning(
                    f"⚠️ STOP-LOSS hit for {symbol} at price {current_price:.2f} (SL was {self.fmt_price(pos['stop_loss'])})"
                )
                self.close_trade(symbol, current_price, reason="Stop-Loss")
                return
        else:
            if current_price > pos["stop_loss"] + sl_buffer:
                logger.warning(
                    f"⚠️ STOP-LOSS hit for {symbol} at price {current_price:.2f} (SL was {self.fmt_price(pos['stop_loss'])})"
                )
                self.close_trade(symbol, current_price, reason="Stop-Loss")
                return

        if side == "BUY":
            if current_price >= pos["take_profit"]:
                logger.info(f"🎯 TAKE-PROFIT hit for {symbol} at price {current_price:.2f}")
                self.close_trade(symbol, current_price, reason="Take-Profit")
                return
        else:
            if current_price <= pos["take_profit"]:
                logger.info(f"🎯 TAKE-PROFIT hit for {symbol} at price {current_price:.2f}")
                self.close_trade(symbol, current_price, reason="Take-Profit")
                return

        if trail_stop:
            if side == "BUY" and current_price <= trail_stop:
                logger.info(
                    f"🏃 TRAILING STOP hit for {symbol} at {self.fmt_price(current_price)} (highest was {self.fmt_price(pos['highest_price'])})"
                )
                self.close_trade(symbol, current_price, reason="Trailing Stop")
            elif side == "SELL" and current_price >= trail_stop:
                logger.info(
                    f"🏃 TRAILING STOP hit for {symbol} at {self.fmt_price(current_price)} (lowest was {self.fmt_price(pos['highest_price'])})"
                )
                self.close_trade(symbol, current_price, reason="Trailing Stop")

    def summary(self):
        from collections import Counter, defaultdict

        open_trades = len(self.positions)
        total_pnl = sum(t["pnl"] for t in self.trade_history)
        logger.info("\n📊 ACCOUNT SUMMARY")
        logger.info(f"💰 Current Balance: ${self.balance:.2f}")
        logger.info(f"📈 Open Trades: {open_trades}")
        logger.info(
            f"📉 Drawdown: {self.drawdown_pct*100:.2f}% | Daily Loss: {self.daily_loss_pct*100:.2f}%"
        )
        logger.info(
            f"✅ Closed Trades: {len(self.trade_history)} | Total PnL: ${total_pnl:.2f} | Fees Paid: ${self.total_fees:.2f}"
        )

        # ➕ Average duration and PnL
        durations = [t["duration"] for t in self.trade_history if "duration" in t]
        if durations:
            avg_dur = sum(durations) / len(durations)
            logger.info(
                f"⏱️ Avg Trade Duration: {avg_dur/60:.1f} min ({avg_dur/3600:.2f} hrs)"
            )

        if self.trade_history:
            avg_pnl = total_pnl / len(self.trade_history)
            wins = [t for t in self.trade_history if t["pnl"] > 0]
            win_rate = len(wins) / len(self.trade_history) * 100
            logger.info(f"📈 Win Rate: {win_rate:.1f}% | Avg PnL: ${avg_pnl:.2f}")

        # ➕ Label-level performance breakdown
        label_stats = defaultdict(list)
        for t in self.trade_history:
            if t.get("label") is not None:
                label_stats[t["label"]].append(t["pnl"])

        for label, pnl_list in label_stats.items():
            avg = sum(pnl_list) / len(pnl_list)
            logger.info(f"🔢 Label {label}: {len(pnl_list)} trades")

        # 📈 Group performance by symbol & duration bucket
        group_stats = defaultdict(lambda: {"pnl": 0, "wins": 0, "count": 0, "fees": 0})

        def bucket(seconds):
            if seconds < 60:
                return "<1m"
            elif seconds < 5 * 60:
                return "1-5m"
            elif seconds < 30 * 60:
                return "5-30m"
            elif seconds < 2 * 3600:
                return "30m-2h"
            else:
                return ">2h"

        for t in self.trade_history:
            symbol = t.get("symbol", "?")
            dur = t.get("duration", 0)
            b = bucket(dur)
            pnl = t.get("pnl", 0)
            fees = t.get("entry_fee", 0) + t.get("exit_fee", 0)
            g = group_stats[(symbol, b)]
            g["pnl"] += pnl
            g["fees"] += fees
            g["count"] += 1
            if pnl > 0:
                g["wins"] += 1

        rows = []
        if group_stats:
            logger.info("\n📊 Performance by Symbol & Duration:")
            for (sym, b), s in group_stats.items():
                win_rate = s["wins"] / s["count"] * 100 if s["count"] else 0
                avg_pnl = s["pnl"] / s["count"] if s["count"] else 0
                fee_ratio = s["fees"] / abs(s["pnl"]) if s["pnl"] else 0
                logger.info(
                    f" - {sym} [{b}]: {s['count']} trades | Win {win_rate:.1f}% | Avg PnL ${avg_pnl:.2f} | Fee/PnL {fee_ratio:.2f}"
                )
                rows.append({
                    "symbol": sym,
                    "duration_bucket": b,
                    "trade_count": s["count"],
                    "win_rate": round(win_rate, 2),
                    "avg_pnl": round(avg_pnl, 2),
                    "fee_ratio": round(fee_ratio, 4)
                })

            # Save analytics to CSV
            import csv, os
            os.makedirs("analytics", exist_ok=True)
            csv_path = os.path.join("analytics", "trade_stats.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["symbol", "duration_bucket", "trade_count", "win_rate", "avg_pnl", "fee_ratio"])
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"📁 Trade analytics saved to {csv_path}")

        # 🔁 Recent trades overview
        if self.trade_history:
            logger.info("\n📌 Recent Trades:")
            for t in self.trade_history[-5:]:
                fees = t.get("entry_fee", 0) + t.get("exit_fee", 0)
                dur = t.get("duration", 0)
                rotation_note = f" | Rotated at ${t['rotation_exit_price']:.2f}" if "rotation_exit_price" in t else ""
                logger.info(
                    f" - {t['symbol']} | Entry ${t['entry_price']:.6f} → Exit ${t['exit_price']:.6f} | "
                    f"PnL ${t['pnl']:.2f} (Fees ${fees:.2f}, {t['reason']}) | Dur {dur/60:.1f} min{rotation_note}"
                )

        # 📊 Label frequency summary
        label_counts = Counter(t.get("label") for t in self.trade_history if t.get("label") is not None)
        logger.info(
            f"📊 Prediction label breakdown (closed trades): {dict(label_counts)}"
        )


    def save_state(self):
        MAX_HISTORY = 500
        if len(self.trade_history) > MAX_HISTORY:
            self.trade_history = self.trade_history[-MAX_HISTORY:]


        state = {
            "balance": self.balance,
            "positions": self.positions,
            "trade_history": self.trade_history,
            "closed_pnl_history": self.closed_pnl_history,
            "total_fees": self.total_fees,
            "trade_fee_pct": self.trade_fee_pct,
            "peak_equity": self.peak_equity,
            "drawdown_pct": self.drawdown_pct,
            "current_day": self.current_day,
            "daily_start_balance": self.daily_start_balance,
            "daily_loss_pct": self.daily_loss_pct,
            "last_trade_time": self.last_trade_time,
            "last_trade_side": self.last_trade_side,
            "last_trade_confidence": self.last_trade_confidence,
            "trail_pct": self.trail_pct,
            "trail_atr_mult": self.trail_atr_mult,
            "symbol_pnl": self.symbol_pnl,
        }
        state = convert_numpy_types(state)
        with open(self.STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("💾 TradeManager state saved.")

    def load_state(self):
        if os.path.exists(self.STATE_FILE):
            with open(self.STATE_FILE, "r") as f:
                state = json.load(f)

            self.balance = state.get("balance", self.starting_balance)
            self.positions = state.get("positions", {})
            self.trade_history = state.get("trade_history", [])
            self.closed_pnl_history = state.get("closed_pnl_history", [])
            self.total_fees = state.get("total_fees", 0.0)
            self.trade_fee_pct = state.get("trade_fee_pct", self.trade_fee_pct)
            self.peak_equity = state.get("peak_equity", self.balance)
            self.drawdown_pct = state.get("drawdown_pct", 0.0)
            self.current_day = state.get("current_day", time.strftime("%Y-%m-%d"))
            self.daily_start_balance = state.get("daily_start_balance", self.balance)
            self.daily_loss_pct = state.get("daily_loss_pct", 0.0)
            self.last_trade_time = state.get("last_trade_time", 0.0)
            self.last_trade_side = state.get("last_trade_side")
            self.last_trade_confidence = state.get("last_trade_confidence")
            self.trail_pct = state.get("trail_pct", self.trail_pct)
            self.trail_atr_mult = state.get("trail_atr_mult", self.trail_atr_mult)
            self.symbol_pnl = state.get("symbol_pnl", {})
            logger.info("📂 TradeManager state loaded.")

            for sym, pos in self.positions.items():
                cid = pos.get("coin_id")
                if not cid or cid.upper() == sym.upper():
                    self.positions[sym]["coin_id"] = sym.lower()

            # WebSocket subscription removed; no live feed setup needed
            self._update_equity_metrics()
        else:
            logger.info("ℹ️ No saved state found, starting fresh.")

