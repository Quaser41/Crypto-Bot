import analytics.performance as perf
from config import PERF_MIN_TRADE_COUNT
import pytest

def test_blacklist_respects_trade_count_threshold(tmp_path):
    csv_path = tmp_path / "stats.csv"
    csv_path.write_text(
        "symbol,duration_bucket,trade_count,win_rate,avg_pnl,fee_ratio\n"
        f"AAA,1-5m,{PERF_MIN_TRADE_COUNT - 1},0.0,-0.5,0.0\n"
        f"BBB,1-5m,{PERF_MIN_TRADE_COUNT},0.0,-0.5,0.0\n"
    )
    perf.reset_cache()
    assert not perf.is_blacklisted("AAA", "1-5m", path=str(csv_path), refresh_seconds=0)
    assert perf.is_blacklisted("BBB", "1-5m", path=str(csv_path), refresh_seconds=0)


def test_get_trade_count_reads_from_stats():
    perf.reset_cache()
    assert perf.get_trade_count("INJ", "5-30m", refresh_seconds=0) == 3


def test_get_avg_fee_ratio_reads_from_stats():
    perf.reset_cache()
    ratio = perf.get_avg_fee_ratio("LINK", "30m-2h", refresh_seconds=0)
    assert ratio == pytest.approx(6.519, rel=1e-3)
