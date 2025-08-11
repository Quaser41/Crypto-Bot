import argparse
import csv
import os


def analyze(file_path: str) -> None:
    if not os.path.exists(file_path):
        print(f"No trade stats found at {file_path}.")
        return

    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Trade stats file is empty.")
        return

    for r in rows:
        # convert numeric fields
        r['avg_pnl'] = float(r.get('avg_pnl', 0))
        r['win_rate'] = float(r.get('win_rate', 0))
        r['fee_ratio'] = float(r.get('fee_ratio', 0))
        r['trade_count'] = int(r.get('trade_count', 0))

    best = max(rows, key=lambda r: r['avg_pnl'])
    worst = min(rows, key=lambda r: r['avg_pnl'])

    print("Best performer:")
    print(f"  {best['symbol']} [{best['duration_bucket']}] | Avg PnL ${best['avg_pnl']:.2f} | Win {best['win_rate']:.1f}% | Fee/PnL {best['fee_ratio']:.2f}")

    print("\nWorst performer:")
    print(f"  {worst['symbol']} [{worst['duration_bucket']}] | Avg PnL ${worst['avg_pnl']:.2f} | Win {worst['win_rate']:.1f}% | Fee/PnL {worst['fee_ratio']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trade statistics")
    parser.add_argument('-f', '--file', default='analytics/trade_stats.csv', help='Path to trade stats CSV')
    args = parser.parse_args()
    analyze(args.file)
