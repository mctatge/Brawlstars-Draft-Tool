"""Unit tests for the data-driven free/"boosted" brawler detector (bsdraft.engine.freebrawlers)
and its integration into DraftStats + the stats artifact roundtrip.

    PYTHONPATH=backend python -m pytest backend/tests/test_freebrawlers.py    # or run directly
"""
from __future__ import annotations

from bsdraft.engine import freebrawlers as F
from bsdraft.engine.stats import DraftStats
from bsdraft.engine.stats_store import stats_payload, load_payload


def _hist(p11, sub):
    """A power histogram with ``p11`` maxed slots and ``sub`` sub-max ones (split across P9/P10)."""
    return {11: p11, 10: sub // 2, 9: sub - sub // 2}


def test_detect_free_separates_free_from_maxed_owners():
    # Free (handed out maxed) = ~all Power 11; a heavily-maxed but non-free brawler still carries a
    # levelling tail. The gate sits in the wide gap between them.
    counts = {
        1: _hist(5000, 0),      # free: zero sub-max
        2: _hist(4996, 4),      # free: 0.08% sub-max, under the 0.5% gate
        3: _hist(4900, 100),    # non-free: 2% sub-max (typical maxed-but-owned brawler)
        4: _hist(9000, 80),     # non-free: ~0.9%, the nearest-non-free band — must NOT qualify
    }
    assert F.detect_free(counts) == frozenset({1, 2})


def test_detect_free_requires_min_sample():
    # A tiny sample can read 0% sub-max by luck — the volume gate rejects it.
    counts = {1: _hist(50, 0), 2: _hist(5000, 0)}
    assert F.detect_free(counts) == frozenset({2})


def test_detect_free_empty_and_all_submax():
    assert F.detect_free({}) == frozenset()
    assert F.detect_free({1: _hist(0, 5000)}) == frozenset()   # everyone levelling → not free
    assert F.submax_fraction({}) == 0.0


def test_draftstats_accumulates_and_detects_from_matches():
    # Two brawlers, same recent window: id 100 always Power 11 (free), id 200 always Power 9.
    ts = 1_800_000_000
    rows = []
    for i in range(400):
        rows.append({
            "ts": ts, "a_won": True, "map_id": 15000072,
            "team_a": [{"brawler_id": 100, "power": 11}],
            "team_b": [{"brawler_id": 200, "power": 9}],
        })
    g = DraftStats(rows)
    assert 100 in g.free_brawler_ids
    assert 200 not in g.free_brawler_ids
    # bracket tables do not carry the signal (global-only)
    assert g.pw_recent[100][11] == 400.0


def test_free_window_excludes_old_matches():
    # A brawler that was maxed-only long ago but is being levelled now must not read as free: only
    # the recent window (anchored to the newest match) counts.
    anchor = 1_800_000_000
    old = anchor - int((F.FREE_WINDOW_DAYS + 5) * 86400)
    rows = []
    for _ in range(400):   # OLD: all Power 11 (outside the window — ignored)
        rows.append({"ts": old, "a_won": True, "map_id": 1,
                     "team_a": [{"brawler_id": 100, "power": 11}], "team_b": [{"brawler_id": 999, "power": 11}]})
    for _ in range(400):   # RECENT: all Power 9 (inside the window — the real signal)
        rows.append({"ts": anchor, "a_won": True, "map_id": 1,
                     "team_a": [{"brawler_id": 100, "power": 9}], "team_b": [{"brawler_id": 999, "power": 11}]})
    g = DraftStats(rows)
    assert 100 not in g.free_brawler_ids       # recent window is all sub-max
    assert 999 in g.free_brawler_ids           # maxed across the recent window


def test_artifact_roundtrip_preserves_free_set():
    ts = 1_800_000_000
    rows = [{"ts": ts, "a_won": True, "map_id": 1,
             "team_a": [{"brawler_id": 100, "power": 11}], "team_b": [{"brawler_id": 200, "power": 9}]}
            for _ in range(400)]
    g = DraftStats(rows)
    g2, _ = load_payload(stats_payload(g, {}))
    assert g2.free_brawler_ids == g.free_brawler_ids
    assert 100 in g2.free_brawler_ids


def test_artifact_without_free_field_loads_empty():
    # An artifact published before the detector existed has no 'free_brawler_ids' key → empty set,
    # never a KeyError (same optional-field discipline as the rest of the loader).
    payload = stats_payload(DraftStats([]), {})
    del payload["global"]["free_brawler_ids"]
    g, _ = load_payload(payload)
    assert g.free_brawler_ids == frozenset()


if __name__ == "__main__":
    import sys
    sys.exit(__import__("pytest").main([__file__, "-q"]))
