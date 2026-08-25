"""The stats artifact's save/load round-trip, which the deployed API depends on entirely.

The synergy tables are keyed by ``frozenset{b1, b2}``. A battle-log glitch can put the same
brawler twice on one team, collapsing that frozenset to a single element — which serializes
without a ``"_"`` separator. The loader used to demand exactly two parts, so ONE such key made
``load_stats`` raise on the whole artifact and the API silently fell back to its capped 60k
rebuild (it served that for weeks before anyone noticed). These tests pin that a singleton
synergy key round-trips instead of poisoning the load.
"""
from __future__ import annotations

from bsdraft.engine.stats import DraftStats
from bsdraft.engine.stats_store import load_stats, save_stats


def _match(team_a, team_b, a_won=True, map_id=7, ts=1_700_000_000):
    return {
        "team_a": [{"brawler_id": b} for b in team_a],
        "team_b": [{"brawler_id": b} for b in team_b],
        "a_won": a_won,
        "map_id": map_id,
        "ts": ts,
    }


def _tables(s: DraftStats) -> dict:
    return {name: dict(getattr(s, name))
            for name in ("b_games", "b_wins", "bm_games", "bm_wins", "map_games",
                         "syn_games", "syn_wins", "cnt_games", "cnt_wins")}


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


def test_duplicate_brawler_on_a_team_roundtrips(tmp_path):
    # The glitch case: brawler 9 fielded twice on team_a -> frozenset{9}, a singleton syn key.
    stats = DraftStats(matches=[_match([9, 9, 3], [4, 5, 6])], halflife_days=0)
    assert frozenset({9}) in stats.syn_games  # the build really does produce the singleton
    path = save_stats(stats, {}, tmp_path / "stats.json.gz")
    loaded, _ = load_stats(path)  # must not raise
    assert _tables(loaded) == _tables(stats)
    assert loaded.syn_games[frozenset({9})] == stats.syn_games[frozenset({9})]
