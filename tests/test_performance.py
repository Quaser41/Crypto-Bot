import analytics.performance as perf

def test_blacklist_respects_trade_count_threshold(tmp_path):
    csv_path = tmp_path / "stats.csv"
    csv_path.write_text(
        "symbol,duration_bucket,trade_count,win_rate,avg_pnl,fee_ratio\n"
        "AAA,1-5m,2,0.0,-0.5,0.0\n"
        "BBB,1-5m,3,0.0,-0.5,0.0\n"
    )
    perf.reset_cache()
    assert not perf.is_blacklisted("AAA", "1-5m", path=str(csv_path), refresh_seconds=0)
    assert perf.is_blacklisted("BBB", "1-5m", path=str(csv_path), refresh_seconds=0)


def test_get_trade_count_reads_from_stats():
    perf.reset_cache()
    assert perf.get_trade_count("INJ", "5-30m", refresh_seconds=0) == 3
