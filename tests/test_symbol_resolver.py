import pytest

from config import MIN_SYMBOL_AVG_PNL, MIN_SYMBOL_WIN_RATE
from symbol_resolver import filter_candidates
import analytics.performance as perf_utils


def test_filter_candidates_enforces_thresholds():
    movers = [
        ("id1", "AAA", "AAA Coin", 10.0),
        ("id2", "BBB", "BBB Coin", 5.0),
        ("id3", "CCC", "CCC Coin", 3.0),
    ]
    performance = {
        # Meets both thresholds
        "AAA": {"avg_pnl": MIN_SYMBOL_AVG_PNL + 0.01, "win_rate": MIN_SYMBOL_WIN_RATE + 5},
        # Fails avg_pnl threshold
        "BBB": {"avg_pnl": MIN_SYMBOL_AVG_PNL - 0.01, "win_rate": MIN_SYMBOL_WIN_RATE + 5},
        # Fails win_rate threshold
        "CCC": {"avg_pnl": MIN_SYMBOL_AVG_PNL + 0.01, "win_rate": MIN_SYMBOL_WIN_RATE - 5},
    }

    result = filter_candidates(movers, set(), performance)
    assert result == [("id1", "AAA", "AAA Coin")]


def test_filter_candidates_skips_blacklisted(monkeypatch):
    movers = [
        ("id1", "AAA", "AAA Coin", 10.0),
        ("id2", "BBB", "BBB Coin", 5.0),
    ]
    performance = {
        "AAA": {
            "avg_pnl": MIN_SYMBOL_AVG_PNL + 0.1,
            "win_rate": MIN_SYMBOL_WIN_RATE + 10,
            "holding_times": [120],
        },
        "BBB": {
            "avg_pnl": MIN_SYMBOL_AVG_PNL + 0.1,
            "win_rate": MIN_SYMBOL_WIN_RATE + 10,
            "holding_times": [120],
        },
    }

    calls = []

    def fake_blacklisted(symbol, bucket):
        calls.append((symbol, bucket))
        return symbol == "AAA" and bucket == "1-5m"

    monkeypatch.setattr(perf_utils, "is_blacklisted", fake_blacklisted)

    result = filter_candidates(movers, set(), performance)
    assert result == [("id2", "BBB", "BBB Coin")]
    expected_bucket = perf_utils.get_duration_bucket(120)
    assert ("AAA", expected_bucket) in calls
