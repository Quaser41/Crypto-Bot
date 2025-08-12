
import json
import os
import time

import numpy as np

from data_fetcher import fetch_live_price
from utils.prediction_class import PredictionClass

from config import ATR_MULT_SL, ATR_MULT_TP

from config import (
    RISK_PER_TRADE,
    MIN_TRADE_USD,
    SLIPPAGE_PCT,
    HOLDING_PERIOD_SECONDS,
    REVERSAL_CONF_DELTA,
)

from utils.logging import get_logger

logger = get_logger(__name__)



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
                 sl_pct=0.06, tp_pct=0.10, trade_fee_pct=0.005,
                 trail_pct=0.03, atr_mult_sl=ATR_MULT_SL,
                 atr_mult_tp=ATR_MULT_TP,
                 max_drawdown_pct=0.20, max_daily_loss_pct=0.05,
                 slippage_pct=SLIPPAGE_PCT,
                 hold_period_sec=HOLDING_PERIOD_SECONDS,
                 reverse_conf_delta=REVERSAL_CONF_DELTA,
                 exchange=None):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.max_allocation = max_allocation
        self.stop_loss_pct = sl_pct
        self.take_profit_pct = tp_pct
        self.trade_fee_pct = trade_fee_pct
        self.trail_pct = trail_pct
        self.atr_mult_sl = atr_mult_sl
        self.atr_mult_tp = atr_mult_tp
        self.slippage_pct = slippage_pct

        # Holding period and reversal requirements
        self.min_hold_time = hold_period_sec
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

        # Equity tracking
        self.peak_equity = starting_balance
        self.drawdown_pct = 0.0
        self.current_day = time.strftime("%Y-%m-%d")
        self.daily_start_balance = starting_balance
        self.daily_loss_pct = 0.0

        self.positions = {}
        self.trade_history = []
        self.total_fees = 0.0

    def has_position(self, symbol):
        return symbol in self.positions

    def fmt_price(self, p):
        return f"{p:.6f}" if p < 1 else f"{p:.2f}"

    def calculate_allocation(self, confidence=1.0):
        """Determine trade size based on current balance and risk settings.

        Allocation is derived from a fixed fraction of equity (``risk_per_trade``)
        and optionally scaled by the model's confidence score to favor
        higher-conviction entries.
        """
        base = self.balance * self.risk_per_trade
        if confidence is not None:
            base *= confidence
        return base

    def _update_equity_metrics(self):
        """Recalculate drawdown and daily loss percentages."""
        today = time.strftime("%Y-%m-%d")
        if self.current_day != today:
            self.current_day = today
            self.daily_start_balance = self.balance
            self.daily_loss_pct = 0.0

        # Only update drawdown using realized equity when no open positions
        if self.positions:
            return

        if self.balance > self.peak_equity:
            self.peak_equity = self.balance

        if self.peak_equity > 0:
            self.drawdown_pct = max(0.0, (self.peak_equity - self.balance) / self.peak_equity)

        if self.daily_start_balance > 0:
            self.daily_loss_pct = max(0.0, (self.daily_start_balance - self.balance) / self.daily_start_balance)

    def can_trade(self):
        """Return True if drawdown and daily loss are within limits."""
        # Refresh metrics to account for new day boundaries
        self._update_equity_metrics()
        if self.drawdown_pct >= self.max_drawdown_pct:
            return False
        if self.daily_loss_pct >= self.max_daily_loss_pct:
            return False
        return True

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

        if self.has_position(symbol):
            logger.warning(f"⚠️ Already have position in {symbol}")
            return

        allocation = self.calculate_allocation(confidence)
        if allocation < self.min_trade_usd:

            logger.warning(
                f"💸 Allocation ${allocation:.2f} below minimum {self.min_trade_usd} for {symbol}, skipping",
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


        if atr_val and atr_val > 0:
            sl_offset = self.atr_mult_sl * atr_val
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
            sl_pct = self.stop_loss_pct
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
            "entry_time": time.time(),
            "last_movement_time": time.time(),
            "atr": atr_val,
            "order_id": order.get("order_id") if order else None,
        }

        msg = (
            f"🚀 OPEN {side.upper()} {symbol}: qty={qty:.4f} @ ${self.fmt_price(exec_price)} | "
            f"Allocated ${allocation:.2f} (fee ${entry_fee:.2f}) | Balance left ${self.balance:.2f} | "
            f"Label={label}"
        )
        if atr_val:
            msg += f" | ATR={atr_val:.6f}"
        logger.info(msg)

        self.last_trade_time = now
        self.last_trade_side = side
        self.last_trade_confidence = confidence

    def close_trade(self, symbol, current_price, reason=""):
        if not self.has_position(symbol):
            logger.warning(f"⚠️ No open position to close for {symbol}")
            return

        pos = self.positions.pop(symbol)
        side = pos.get("side", "BUY")

        if pos["qty"] <= 0:
            logger.warning(f"❌ Invalid quantity for {symbol}, skipping close.")
            return

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

        entry_val = pos["entry_price"] * qty
        exit_val = exec_price * qty

        exit_fee = exit_val * self.trade_fee_pct
        net_exit = exit_val - exit_fee
        self.total_fees += exit_fee

        pnl = net_exit - (entry_val + pos.get("entry_fee", 0))
        self.balance += net_exit

        self.last_trade_time = time.time()

        # Update drawdown and daily loss metrics after realizing PnL
        self._update_equity_metrics()

        duration = time.time() - pos.get("entry_time", time.time())

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
        }

        if reason == "Rotated to better candidate":
            trade_record["rotation_exit_price"] = current_price

        self.trade_history.append(trade_record)

        logger.info(
            f"🔐 CLOSE {symbol} | Exit ${self.fmt_price(exec_price)} | "
            f"PnL: ${pnl:.2f} | Fee ${exit_fee:.2f} | Balance now ${self.balance:.2f}"
        )
        self.save_state()


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

                price_change = abs(last_price - entry_price) / entry_price
                if price_change >= 0.005:
                    pos["last_movement_time"] = time.time()

                # ✅ Hybrid profit protection
                pnl_pct = (last_price - entry_price) / entry_price * 100

                # Hard take-profit at 10–12%
                if pnl_pct >= 10:
                    logger.info(
                        f"🎯 Take-profit hit: {symbol} is up {pnl_pct:.2f}% — closing trade."
                    )
                    self.close_trade(symbol, last_price, reason="Take Profit Hit")
                    continue  # skip further checks on this trade

                # Trailing stop trigger at 3% gain with 2% trail
                elif pnl_pct >= 3:
                    if not pos.get("trail_triggered"):
                        pos["trail_triggered"] = True
                        pos["trail_price"] = last_price * 0.98
                        logger.info(
                            f"📉 Trailing stop activated at {last_price:.4f} for {symbol}"
                        )
                    elif last_price < pos.get("trail_price", 0):
                        logger.info(f"🔻 Trailing stop hit for {symbol} — closing trade.")
                        self.close_trade(symbol, last_price, reason="Trailing Stop")
                        continue
                    else:
                        # Update trail upward if price climbs
                        new_trail = last_price * 0.98
                        if new_trail > pos["trail_price"]:
                            logger.info(
                                f"🔼 Updating trail stop for {symbol}: {pos['trail_price']:.4f} → {new_trail:.4f}"
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

        sl_buffer = 0.999
        if side == "BUY":
            if current_price < pos["stop_loss"] * sl_buffer:
                logger.warning(
                    f"⚠️ STOP-LOSS hit for {symbol} at price {current_price:.2f} (SL was {self.fmt_price(pos['stop_loss'])})"
                )
                self.close_trade(symbol, current_price, reason="Stop-Loss")
                return
        else:
            if current_price > pos["stop_loss"] / sl_buffer:
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
            "total_fees": self.total_fees,
            "peak_equity": self.peak_equity,
            "drawdown_pct": self.drawdown_pct,
            "current_day": self.current_day,
            "daily_start_balance": self.daily_start_balance,
            "daily_loss_pct": self.daily_loss_pct,
            "last_trade_time": self.last_trade_time,
            "last_trade_side": self.last_trade_side,
            "last_trade_confidence": self.last_trade_confidence,
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
            self.total_fees = state.get("total_fees", 0.0)
            self.peak_equity = state.get("peak_equity", self.balance)
            self.drawdown_pct = state.get("drawdown_pct", 0.0)
            self.current_day = state.get("current_day", time.strftime("%Y-%m-%d"))
            self.daily_start_balance = state.get("daily_start_balance", self.balance)
            self.daily_loss_pct = state.get("daily_loss_pct", 0.0)
            self.last_trade_time = state.get("last_trade_time", 0.0)
            self.last_trade_side = state.get("last_trade_side")
            self.last_trade_confidence = state.get("last_trade_confidence")
            logger.info("📂 TradeManager state loaded.")

            for sym, pos in self.positions.items():
                cid = pos.get("coin_id")
                if not cid or cid.upper() == sym.upper():
                    self.positions[sym]["coin_id"] = sym.lower()

            # WebSocket subscription removed; no live feed setup needed
            self._update_equity_metrics()
        else:
            logger.info("ℹ️ No saved state found, starting fresh.")

