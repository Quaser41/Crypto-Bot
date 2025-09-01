import pytest

from config import MIN_SYMBOL_AVG_PNL, MIN_SYMBOL_WIN_RATE
import symbol_resolver
from symbol_resolver import filter_candidates
import analytics.performance as perf_utils


def test_binance_global_451_warning_logged_once(monkeypatch, caplog):
    """Ensure 451 status is logged only once and subsequent calls stop retrying."""

    class MockResp:
        status_code = 451

        def raise_for_status(self):
            pass

    calls = {}

    def mock_get(*args, **kwargs):
        calls["count"] = calls.get("count", 0) + 1
        return MockResp()

    # Reset globals and patch request
    monkeypatch.setattr(symbol_resolver, "BINANCE_GLOBAL_SYMBOLS", {})
    monkeypatch.setattr(symbol_resolver, "BINANCE_GLOBAL_UNAVAILABLE", False)
    monkeypatch.setattr(symbol_resolver.requests, "get", mock_get)

    with caplog.at_level("WARNING"):
        symbol_resolver.resolve_symbol_binance_global("BTC")
        symbol_resolver.resolve_symbol_binance_global("ETH")

    warnings = [
        rec
        for rec in caplog.records
        if "Binance Global blocked (451)" in rec.message
    ]
    assert len(warnings) == 1
    assert calls.get("count", 0) == 1


def test_filter_candidates_enforces_thresholds():
    movers = [
        ("id1", "AAA", "AAA Coin", 10.0, 2_000_000),
        ("id2", "BBB", "BBB Coin", 5.0, 2_000_000),
        ("id3", "CCC", "CCC Coin", 3.0, 2_000_000),
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
        ("id1", "AAA", "AAA Coin", 10.0, 2_000_000),
        ("id2", "BBB", "BBB Coin", 5.0, 2_000_000),
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

    monkeypatch.setattr(symbol_resolver, "MIN_HOLD_BUCKET", "<1m")

    calls = []

    def fake_blacklisted(symbol, bucket):
        calls.append((symbol, bucket))
        return symbol == "AAA" and bucket == "1-5m"

    monkeypatch.setattr(perf_utils, "is_blacklisted", fake_blacklisted)

    result = filter_candidates(movers, set(), performance)
    assert result == [("id2", "BBB", "BBB Coin")]
    expected_bucket = perf_utils.get_duration_bucket(120)
    assert ("AAA", expected_bucket) in calls


def test_filter_candidates_respects_min_hold_bucket(monkeypatch):
    movers = [("id1", "AAA", "AAA Coin", 10.0, 2_000_000)]
    performance = {
        "AAA": {
            "avg_pnl": MIN_SYMBOL_AVG_PNL + 0.1,
            "win_rate": MIN_SYMBOL_WIN_RATE + 10,
            "holding_times": [120],
        }
    }

    monkeypatch.setattr(symbol_resolver, "MIN_HOLD_BUCKET", "30m-2h")
    monkeypatch.setattr(perf_utils, "is_blacklisted", lambda *a, **k: False)

    result = filter_candidates(movers, set(), performance)
    assert result == []


def test_filter_candidates_skips_low_volume(monkeypatch):
    movers = [("id1", "AAA", "AAA Coin", 10.0, 500)]
    performance = {
        "AAA": {
            "avg_pnl": MIN_SYMBOL_AVG_PNL + 0.1,
            "win_rate": MIN_SYMBOL_WIN_RATE + 10,
        }
    }
    monkeypatch.setattr(symbol_resolver, "MIN_24H_VOLUME", 1000)

    result = filter_candidates(movers, set(), performance)
    assert result == []


def test_filter_candidates_sorts_by_score():
    movers = [
        ("id1", "AAA", "AAA Coin", 10.0, 2_000_000),
        ("id2", "BBB", "BBB Coin", 5.0, 2_000_000),
    ]
    performance = {
        "AAA": {"avg_pnl": 0.07, "win_rate": 80},
        "BBB": {"avg_pnl": 0.2, "win_rate": 70},
    }

    # Default weighting favors higher avg PnL when win rates are close
    result = filter_candidates(movers, set(), performance)
    assert [sym for _, sym, _ in result] == ["BBB", "AAA"]

    # Heavier weighting toward win rate flips the order
    result = filter_candidates(movers, set(), performance, win_rate_weight=0.8)
    assert [sym for _, sym, _ in result] == ["AAA", "BBB"]
