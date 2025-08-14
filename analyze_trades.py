import argparse
import csv
import os
import time
import json
from collections import defaultdict

from utils.logging import get_logger

logger = get_logger(__name__)


def export_trade_stats(state_file: str, out_path: str) -> None:
    """Aggregate trade performance from saved state and export to CSV."""
    if not os.path.exists(state_file):
        logger.warning("No trade manager state found at %s.", state_file)
        return

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
    except Exception as e:
        logger.error("Failed to read state file %s: %s", state_file, e)
        return

    trades = state.get("trade_history", [])
    if not trades:
        logger.warning("No trades found in state file.")
        return

    stats = defaultdict(lambda: {"pnl": 0.0, "wins": 0, "count": 0})
    for t in trades:
        sym = t.get("symbol")
        pnl = float(t.get("pnl", 0))
        s = stats[sym]
        s["pnl"] += pnl
        s["count"] += 1
        if pnl > 0:
            s["wins"] += 1

    rows = []
    for sym, s in stats.items():
        win_rate = s["wins"] / s["count"] * 100 if s["count"] else 0
        avg_pnl = s["pnl"] / s["count"] if s["count"] else 0
        rows.append(
            {
                "symbol": sym,
                "duration_bucket": "all",
                "trade_count": s["count"],
                "win_rate": round(win_rate, 2),
                "avg_pnl": round(avg_pnl, 2),
                "fee_ratio": 0,
            }
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "symbol",
                "duration_bucket",
                "trade_count",
                "win_rate",
                "avg_pnl",
                "fee_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("📁 Exported trade stats to %s", out_path)


def analyze(file_path: str) -> None:
    if not os.path.exists(file_path):
        logger.warning("No trade stats found at %s.", file_path)
        return

    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.warning("Trade stats file is empty.")
        return

    for r in rows:
        # convert numeric fields
        r['avg_pnl'] = float(r.get('avg_pnl', 0))
        r['win_rate'] = float(r.get('win_rate', 0))
        r['fee_ratio'] = float(r.get('fee_ratio', 0))
        r['trade_count'] = int(r.get('trade_count', 0))

    best = max(rows, key=lambda r: r['avg_pnl'])
    worst = min(rows, key=lambda r: r['avg_pnl'])

    logger.info("Best performer:")
    logger.info(
        "  %s [%s] | Avg PnL $%.2f | Win %.1f%% | Fee/PnL %.2f",
        best['symbol'],
        best['duration_bucket'],
        best['avg_pnl'],
        best['win_rate'],
        best['fee_ratio'],
    )

    logger.info("\nWorst performer:")
    logger.info(
        "  %s [%s] | Avg PnL $%.2f | Win %.1f%% | Fee/PnL %.2f",
        worst['symbol'],
        worst['duration_bucket'],
        worst['avg_pnl'],
        worst['win_rate'],
        worst['fee_ratio'],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trade statistics")
    parser.add_argument(
        "-s",
        "--state-file",
        default="trade_manager_state.json",
        help="Path to TradeManager state JSON",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="analytics/trade_stats.csv",
        help="Where to write aggregated stats",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between exports; 0 for one-time run",
    )
    args = parser.parse_args()

    while True:
        export_trade_stats(args.state_file, args.out)
        analyze(args.out)
        if args.interval <= 0:
            break
        time.sleep(args.interval)
