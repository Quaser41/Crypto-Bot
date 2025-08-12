import argparse
import csv
import os

from utils.logging import get_logger

logger = get_logger(__name__)


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
    parser.add_argument('-f', '--file', default='analytics/trade_stats.csv', help='Path to trade stats CSV')
    args = parser.parse_args()
    analyze(args.file)
