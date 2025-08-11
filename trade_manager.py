import json
import os
import time
import numpy as np

from data_fetcher import fetch_live_price
from config import RISK_PER_TRADE, MIN_TRADE_USD

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
                 trail_pct=0.03, risk_per_trade=RISK_PER_TRADE):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.max_allocation = max_allocation
        self.stop_loss_pct = sl_pct
        self.take_profit_pct = tp_pct
        self.trade_fee_pct = trade_fee_pct
        self.trail_pct = trail_pct
        self.risk_per_trade = risk_per_trade
        self.min_trade_usd = MIN_TRADE_USD

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

    def open_trade(self, symbol, price, coin_id=None, confidence=0.5, label=None, side="BUY"):
        if self.has_position(symbol):
            print(f"⚠️ Already have position in {symbol}")
            return

        allocation = self.calculate_allocation(confidence)
        if allocation < self.min_trade_usd:
            print(
                f"💸 Allocation ${allocation:.2f} below minimum {self.min_trade_usd} for {symbol}, skipping"
            )
            return

        entry_fee = allocation * self.trade_fee_pct
        net_allocation = allocation - entry_fee
        qty = net_allocation / price

        if qty < 0.0001:
            print(f"⚠️ Trade qty too small for {symbol}, skipping")
            return

        if label == 0 and side == "BUY":
            print(f"⚠️ Model predicts loss for {symbol} (label 0), skipping long trade.")
            return

        self.balance -= allocation
        self.total_fees += entry_fee

        tp_pct = self.take_profit_pct
        sl_pct = self.stop_loss_pct

        if label == 4:
            tp_pct = 0.15
            sl_pct = 0.05
        elif label == 3:
            tp_pct = 0.10
            sl_pct = 0.06
        elif label == 2:
            tp_pct = 0.06
            sl_pct = 0.06
        elif label == 1:
            tp_pct = 0.04
            sl_pct = 0.08

        if side == "SELL":
            stop_loss = price * (1 + sl_pct) * (1 + self.trade_fee_pct)
            take_profit = price * (1 - tp_pct) * (1 - self.trade_fee_pct)
        else:
            stop_loss = price * (1 - sl_pct) * (1 - self.trade_fee_pct)
            take_profit = price * (1 + tp_pct) * (1 + self.trade_fee_pct)

        self.positions[symbol] = {
            "coin_id": coin_id or symbol.lower(),
            "entry_price": price,
            "qty": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_fee": entry_fee,
            "highest_price": price,
            "confidence": confidence,
            "label": label,
            "side": side,
            "entry_time": time.time(),
            "last_movement_time": time.time()
        }

        print(f"🚀 OPEN {side.upper()} {symbol}: qty={qty:.4f} @ ${self.fmt_price(price)} | "
              f"Allocated ${allocation:.2f} (fee ${entry_fee:.2f}) | Balance left ${self.balance:.2f} | "
              f"Label={label}")

    def close_trade(self, symbol, current_price, reason=""):
        if not self.has_position(symbol):
            print(f"⚠️ No open position to close for {symbol}")
            return

        pos = self.positions.pop(symbol)

        if pos["qty"] <= 0:
            print(f"❌ Invalid quantity for {symbol}, skipping close.")
            return

        entry_val = pos["entry_price"] * pos["qty"]
        exit_val = current_price * pos["qty"]

        exit_fee = exit_val * self.trade_fee_pct
        net_exit = exit_val - exit_fee
        self.total_fees += exit_fee

        pnl = net_exit - (entry_val + pos.get("entry_fee", 0))
        self.balance += net_exit

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
            print(f"⚠️ Could not capture exit momentum for {symbol}: {e}")

        # ✅ Build trade record AFTER collecting momentum
        trade_record = {
            "symbol": symbol,
            "entry_price": pos["entry_price"],
            "exit_price": current_price,
            "qty": pos["qty"],
            "pnl": pnl,
            "reason": reason,
            "entry_fee": pos.get("entry_fee", 0),
            "exit_fee": exit_fee,
            "confidence": pos.get("confidence"),
            "label": pos.get("label"),
            "trail_triggered": reason == "Trailing Stop",
            "duration": round(duration, 2),
            "exit_momentum": exit_momentum
        }

        if reason == "Rotated to better candidate":
            trade_record["rotation_exit_price"] = current_price

        self.trade_history.append(trade_record)

        print(f"🔐 CLOSE {symbol} | Exit ${self.fmt_price(current_price)} | "
            f"PnL: ${pnl:.2f} | Fee ${exit_fee:.2f} | Balance now ${self.balance:.2f}")
        self.save_state()

    def monitor_open_trades(self, check_interval=60, single_run=False):
        print(f"🚦 Monitoring {len(self.positions)} open trade(s)...")

        if not self.positions:
            self.load_state()
            print(f"🔄 Reloaded state → {len(self.positions)} positions")

        def check_once():
            for symbol, pos in list(self.positions.items()):
                coin_id = pos.get("coin_id", symbol.lower())
                last_price = fetch_live_price(symbol, coin_id)
                if last_price is None or last_price <= 0:
                    print(f"⚠️ Could not fetch live price for {symbol}, skipping this check.")
                    continue

                entry_price = pos["entry_price"]
                print(f"🔄 Monitoring {symbol}: entry ${self.fmt_price(entry_price)} | "
                      f"current ${self.fmt_price(last_price)}")

                if last_price < entry_price * 0.01:
                    print(f"⚠️ Suspicious price ({last_price}), skipping to prevent false SL trigger.")
                    continue

                price_change = abs(last_price - entry_price) / entry_price
                if price_change >= 0.005:
                    pos["last_movement_time"] = time.time()

                # ✅ Hybrid profit protection
                pnl_pct = (last_price - entry_price) / entry_price * 100

                # Hard take-profit at 10–12%
                if pnl_pct >= 10:
                    print(f"🎯 Take-profit hit: {symbol} is up {pnl_pct:.2f}% — closing trade.")
                    self.close_trade(symbol, last_price, reason="Take Profit Hit")
                    continue  # skip further checks on this trade

                # Trailing stop trigger at 3% gain with 2% trail
                elif pnl_pct >= 3:
                    if not pos.get("trail_triggered"):
                        pos["trail_triggered"] = True
                        pos["trail_price"] = last_price * 0.98
                        print(f"📉 Trailing stop activated at {last_price:.4f} for {symbol}")
                    elif last_price < pos.get("trail_price", 0):
                        print(f"🔻 Trailing stop hit for {symbol} — closing trade.")
                        self.close_trade(symbol, last_price, reason="Trailing Stop")
                        continue
                    else:
                        # Update trail upward if price climbs
                        new_trail = last_price * 0.98
                        if new_trail > pos["trail_price"]:
                            print(f"🔼 Updating trail stop for {symbol}: {pos['trail_price']:.4f} → {new_trail:.4f}")
                            pos["trail_price"] = new_trail

                self.manage(symbol, last_price)

            self.summary()
            self.save_state()

        if single_run:
            check_once()
            return

        while self.positions:
            check_once()
            print(f"⏳ Sleeping {check_interval} seconds before next check...\n")
            time.sleep(check_interval)

        print("✅ No more open trades. Exiting monitoring.")

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
        if side == "BUY" and pos["highest_price"] > entry_price:
            trail_stop = pos["highest_price"] * (1 - self.trail_pct)
        elif side == "SELL" and pos["highest_price"] < entry_price:
            trail_stop = pos["highest_price"] * (1 + self.trail_pct)

        sl_buffer = 0.999
        if side == "BUY":
            if current_price < pos["stop_loss"] * sl_buffer:
                print(f"⚠️ STOP-LOSS hit for {symbol} at price {current_price:.2f} (SL was {self.fmt_price(pos['stop_loss'])})")
                self.close_trade(symbol, current_price, reason="Stop-Loss")
                return
        else:
            if current_price > pos["stop_loss"] / sl_buffer:
                print(f"⚠️ STOP-LOSS hit for {symbol} at price {current_price:.2f} (SL was {self.fmt_price(pos['stop_loss'])})")
                self.close_trade(symbol, current_price, reason="Stop-Loss")
                return

        if side == "BUY":
            if current_price >= pos["take_profit"]:
                print(f"🎯 TAKE-PROFIT hit for {symbol} at price {current_price:.2f}")
                self.close_trade(symbol, current_price, reason="Take-Profit")
                return
        else:
            if current_price <= pos["take_profit"]:
                print(f"🎯 TAKE-PROFIT hit for {symbol} at price {current_price:.2f}")
                self.close_trade(symbol, current_price, reason="Take-Profit")
                return

        if trail_stop:
            if side == "BUY" and current_price <= trail_stop:
                print(f"🏃 TRAILING STOP hit for {symbol} at {self.fmt_price(current_price)} (highest was {self.fmt_price(pos['highest_price'])})")
                self.close_trade(symbol, current_price, reason="Trailing Stop")
            elif side == "SELL" and current_price >= trail_stop:
                print(f"🏃 TRAILING STOP hit for {symbol} at {self.fmt_price(current_price)} (lowest was {self.fmt_price(pos['highest_price'])})")
                self.close_trade(symbol, current_price, reason="Trailing Stop")

    def summary(self):
        from collections import Counter, defaultdict

        open_trades = len(self.positions)
        total_pnl = sum(t["pnl"] for t in self.trade_history)

        print("\n📊 ACCOUNT SUMMARY")
        print(f"💰 Current Balance: ${self.balance:.2f}")
        print(f"📈 Open Trades: {open_trades}")
        print(f"✅ Closed Trades: {len(self.trade_history)} | "
            f"Total PnL: ${total_pnl:.2f} | Fees Paid: ${self.total_fees:.2f}")

        # ➕ Average duration and PnL
        durations = [t["duration"] for t in self.trade_history if "duration" in t]
        if durations:
            avg_dur = sum(durations) / len(durations)
            print(f"⏱️ Avg Trade Duration: {avg_dur/60:.1f} min ({avg_dur/3600:.2f} hrs)")

        if self.trade_history:
            avg_pnl = total_pnl / len(self.trade_history)
            wins = [t for t in self.trade_history if t["pnl"] > 0]
            win_rate = len(wins) / len(self.trade_history) * 100
            print(f"📈 Win Rate: {win_rate:.1f}% | Avg PnL: ${avg_pnl:.2f}")

        # ➕ Label-level performance breakdown
        label_stats = defaultdict(list)
        for t in self.trade_history:
            if t.get("label") is not None:
                label_stats[t["label"]].append(t["pnl"])

        for label, pnl_list in label_stats.items():
            avg = sum(pnl_list) / len(pnl_list)
            print(f"🔢 Label {label}: {len(pnl_list)} trades")

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
            print("\n📊 Performance by Symbol & Duration:")
            for (sym, b), s in group_stats.items():
                win_rate = s["wins"] / s["count"] * 100 if s["count"] else 0
                avg_pnl = s["pnl"] / s["count"] if s["count"] else 0
                fee_ratio = s["fees"] / abs(s["pnl"]) if s["pnl"] else 0
                print(f" - {sym} [{b}]: {s['count']} trades | Win {win_rate:.1f}% | Avg PnL ${avg_pnl:.2f} | Fee/PnL {fee_ratio:.2f}")
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
            print(f"📁 Trade analytics saved to {csv_path}")

        # 🔁 Recent trades overview
        if self.trade_history:
            print("\n📌 Recent Trades:")
            for t in self.trade_history[-5:]:
                fees = t.get("entry_fee", 0) + t.get("exit_fee", 0)
                dur = t.get("duration", 0)
                rotation_note = f" | Rotated at ${t['rotation_exit_price']:.2f}" if "rotation_exit_price" in t else ""
                print(f" - {t['symbol']} | Entry ${t['entry_price']:.6f} → Exit ${t['exit_price']:.6f} | "
                    f"PnL ${t['pnl']:.2f} (Fees ${fees:.2f}, {t['reason']}) | Dur {dur/60:.1f} min{rotation_note}")

        # 📊 Label frequency summary
        label_counts = Counter(t.get("label") for t in self.trade_history if t.get("label") is not None)
        print(f"📊 Prediction label breakdown (closed trades): {dict(label_counts)}")


    def save_state(self):
        MAX_HISTORY = 500
        if len(self.trade_history) > MAX_HISTORY:
            self.trade_history = self.trade_history[-MAX_HISTORY:]

        state = {
            "balance": self.balance,
            "positions": self.positions,
            "trade_history": self.trade_history,
            "total_fees": self.total_fees
        }
        state = convert_numpy_types(state)
        with open(self.STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        print("💾 TradeManager state saved.")

    def load_state(self):
        if os.path.exists(self.STATE_FILE):
            with open(self.STATE_FILE, "r") as f:
                state = json.load(f)
            self.balance = state.get("balance", self.starting_balance)
            self.positions = state.get("positions", {})
            self.trade_history = state.get("trade_history", [])
            self.total_fees = state.get("total_fees", 0.0)
            print("📂 TradeManager state loaded.")

            for sym, pos in self.positions.items():
                cid = pos.get("coin_id")
                if not cid or cid.upper() == sym.upper():
                    self.positions[sym]["coin_id"] = sym.lower()

            # WebSocket subscription removed; no live feed setup needed
        else:
            print("ℹ️ No saved state found, starting fresh.")
