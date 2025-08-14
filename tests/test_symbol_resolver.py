import pytest

from config import MIN_SYMBOL_AVG_PNL, MIN_SYMBOL_WIN_RATE
from symbol_resolver import filter_candidates


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
