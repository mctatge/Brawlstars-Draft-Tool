"""The stats artifact's save/load round-trip, which the deployed API depends on entirely.

The synergy tables are keyed by ``frozenset{b1, b2}``. A battle-log glitch can put the same
brawler twice on one team, collapsing that frozenset to a single element — which serializes
without a ``"_"`` separator. The loader used to demand exactly two parts, so ONE such key made
``load_stats`` raise on the whole artifact and the API silently fell back to its capped 60k
rebuild (it served that for weeks before anyone noticed). These tests pin that a singleton
synergy key round-trips instead of poisoning the load.
"""
from __future__ import annotations

from bsdraft.engine.stats import RECENT_WINDOW_DAYS, DraftStats, build_bracketed
from bsdraft.engine.stats_store import load_stats, save_stats, stats_payload, load_payload
from bsdraft.engine.tiers import bracket_of_tier


def _match(team_a, team_b, a_won=True, map_id=7, ts=1_700_000_000, tier=None):
    player = ({"brawler_id": 0, "trophies": tier} if tier is not None
              else {"brawler_id": 0})
    return {
        "team_a": [dict(player, brawler_id=b) for b in team_a],
        "team_b": [dict(player, brawler_id=b) for b in team_b],
        "a_won": a_won,
        "map_id": map_id,
        "ts": ts,
    }


def _tables(s: DraftStats) -> dict:
    return {name: dict(getattr(s, name))
            for name in ("b_games", "b_wins", "bm_games", "bm_wins", "map_games",
                         "map_games_recent", "syn_games", "syn_wins", "cnt_games", "cnt_wins")}


def test_roundtrip_is_lossless(tmp_path):
    stats = DraftStats(matches=[
        _match([1, 2, 3], [4, 5, 6]),
        _match([1, 2, 4], [3, 5, 6], a_won=False),
    ], halflife_days=0)  # no decay -> exact float comparison is safe
    path = save_stats(stats, {}, tmp_path / "stats.json.gz")
    loaded, brackets = load_stats(path)
    assert loaded.n == stats.n
    assert _tables(loaded) == _tables(stats)
    assert brackets == {}


def test_recent_window_counts_only_fresh_matches():
    # `map_games_recent` is the rotation-liveness signal: raw games within RECENT_WINDOW_DAYS
    # of the newest match. A game from an earlier rotation (the 3 stray Beach Ball games from
    # March 2026) must not count, and a timestamp-less row can't claim liveness.
    now = 1_800_000_000
    stats = DraftStats(matches=[
        _match([1, 2, 3], [4, 5, 6], map_id=7, ts=now),
        _match([1, 2, 3], [4, 5, 6], map_id=7, ts=now - 86400),        # 1 day old: in window
        _match([1, 2, 3], [4, 5, 6], map_id=8, ts=now - 30 * 86400),   # 30 days old: out
        _match([1, 2, 3], [4, 5, 6], map_id=9, ts=0),                  # no timestamp: out
    ])
    assert stats.map_games_recent[7] == 2.0
    assert 8 not in stats.map_games_recent
    assert 9 not in stats.map_games_recent
    assert stats.map_games[8] > 0    # still known to the slow table, just not "live"


def test_recent_window_boundary_is_inclusive():
    # A match exactly RECENT_WINDOW_DAYS old is the window's edge — in, not out. Pins the
    # anchoring arithmetic (newest match, not wall clock) independent of the calendar.
    now = 1_800_000_000
    edge = now - int(RECENT_WINDOW_DAYS * 86400)
    stats = DraftStats(matches=[
        _match([1, 2, 3], [4, 5, 6], map_id=7, ts=now),
        _match([1, 2, 3], [4, 5, 6], map_id=8, ts=edge),
        _match([1, 2, 3], [4, 5, 6], map_id=9, ts=edge - 1),
    ])
    assert stats.map_games_recent[8] == 1.0
    assert 9 not in stats.map_games_recent


def test_bracket_recent_windows_are_anchored_to_the_dataset_not_the_bracket():
    # A thin bracket's own newest match can predate a rotation flip by days. Self-anchoring
    # would make that bracket's table report the pre-flip rotation as currently live — and the
    # artifact serializes bracket tables, so the stale window would ship. The window anchor is
    # the DATASET's newest match for every table.
    now = 1_800_000_000
    stale = now - 10 * 86400
    low_bracket, high_bracket = bracket_of_tier(2), bracket_of_tier(16)
    rows = ([_match([1, 2, 3], [4, 5, 6], map_id=200, ts=now, tier=16) for _ in range(3)]
            + [_match([1, 2, 3], [4, 5, 6], map_id=100, ts=stale, tier=2) for _ in range(3)])
    g, br = build_bracketed(matches=rows, min_matches=1)
    assert g.map_games_recent.get(200) == 3.0
    assert 100 not in g.map_games_recent
    assert 100 not in br[low_bracket].map_games_recent   # its own games are 10 days old
    assert br[high_bracket].map_games_recent.get(200) == 3.0


def test_artifact_without_the_recent_table_still_loads():
    # Artifacts published before 2026-08-25 have no `map_games_recent`. The loader must default
    # it empty (consumers fall back to the cumulative cut) — not raise, which would silently
    # drop the API to its capped 60k rebuild, the exact failure the syn-key bug caused.
    stats = DraftStats(matches=[_match([1, 2, 3], [4, 5, 6])], halflife_days=0)
    payload = stats_payload(stats, {})
    del payload["global"]["map_games_recent"]
    loaded, _ = load_payload(payload)
    assert dict(loaded.map_games_recent) == {}
    assert dict(loaded.map_games) == dict(stats.map_games)


def test_duplicate_brawler_on_a_team_roundtrips(tmp_path):
    # The glitch case: brawler 9 fielded twice on team_a -> frozenset{9}, a singleton syn key.
    stats = DraftStats(matches=[_match([9, 9, 3], [4, 5, 6])], halflife_days=0)
    assert frozenset({9}) in stats.syn_games  # the build really does produce the singleton
    path = save_stats(stats, {}, tmp_path / "stats.json.gz")
    loaded, _ = load_stats(path)  # must not raise
    assert _tables(loaded) == _tables(stats)
    assert loaded.syn_games[frozenset({9})] == stats.syn_games[frozenset({9})]
