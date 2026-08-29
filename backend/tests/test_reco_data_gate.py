"""A brawler the data has never observed is withheld from pick recommendations.

The failure this guards: a brawler with zero collected games (an unreleased catalog leak, or a
brand-new released brawler the crawler hasn't logged yet) is scored purely from priors — a
baseline-shrunk map winrate, a class-level role prior, and an untrained model embedding — and can
float to the top of the pick list at "CONFIDENCE 0%". ``DraftEngine._data_backed`` drops such
candidates once the table is populated enough for a zero to be a real signal, while never hiding a
brawler we actually have data on and never emptying the board on a cold start.
"""
from __future__ import annotations

from bsdraft.data import reference as R
from bsdraft.engine.engine import DraftEngine, MIN_RECO_GAMES, MIN_TABLE_MATCHES
from bsdraft.engine.state import DraftState
from bsdraft.engine.stats import DraftStats


def _ids(n: int):
    return [b.id for b in R.pickable_brawlers()][:n]


def _matches(seen_ids, map_id, count):
    """`count` synthetic matches that only ever field `seen_ids`, so every other real brawler
    ends the build with zero games. Uniform (halflife-disabled) and deterministic."""
    rows = []
    for i in range(count):
        a = [seen_ids[i % len(seen_ids)], seen_ids[(i + 1) % len(seen_ids)],
             seen_ids[(i + 2) % len(seen_ids)]]
        b = [seen_ids[(i + 3) % len(seen_ids)], seen_ids[(i + 4) % len(seen_ids)],
             seen_ids[(i + 5) % len(seen_ids)]]
        rows.append({
            "a_won": (i % 2 == 0),
            "team_a": [{"brawler_id": x, "power": 11} for x in a],
            "team_b": [{"brawler_id": x, "power": 11} for x in b],
            "map_id": map_id,
            "ts": 1_700_000_000 + i,
        })
    return rows


def _populated_engine():
    seen = _ids(8)
    a_map = R.load_ranked_maps()[0]
    stats = DraftStats(_matches(seen, a_map.id, MIN_TABLE_MATCHES + 1000), halflife_days=0)
    return DraftEngine(stats=stats, model=None), stats, seen, a_map


def test_unseen_brawler_is_withheld_from_recommendations():
    engine, stats, seen, a_map = _populated_engine()
    assert stats.n >= MIN_TABLE_MATCHES
    unseen = [i for i in _ids(60) if i not in seen]           # real brawlers with zero games here
    assert unseen and all(stats.brawler_rate(i, None).games == 0 for i in unseen)

    st = DraftState(map_id=a_map.id, mode=a_map.mode, we_pick_first=True)
    recommended = {s.brawler_id for s in engine.recommend_picks(st, top=200)}

    assert recommended.isdisjoint(unseen)                     # never recommends a zero-data brawler
    assert set(seen) <= recommended                           # the observed ones still appear


def test_cold_start_gates_nothing():
    """Below MIN_TABLE_MATCHES the whole table is too thin for "unseen" to mean anything, so the
    gate is inert and a rebuilding backend still shows the full board."""
    seen = _ids(8)
    a_map = R.load_ranked_maps()[0]
    thin = DraftStats(_matches(seen, a_map.id, 50), halflife_days=0)
    assert thin.n < MIN_TABLE_MATCHES
    engine = DraftEngine(stats=thin, model=None)
    cands = engine.candidates(DraftState(map_id=a_map.id, mode=a_map.mode, we_pick_first=True))
    assert engine._data_backed(cands) == cands                # nothing dropped


def test_gate_never_returns_empty():
    """Even above the table threshold, if every candidate is below the floor the gate falls back
    to the ungated list rather than returning nothing to rank."""
    engine, stats, seen, a_map = _populated_engine()
    all_unseen = [i for i in _ids(60) if i not in seen][:5]
    assert engine._data_backed(all_unseen) == all_unseen      # all-thin → ungated fallback


def test_floor_is_the_smoothing_prior():
    """The recommend floor is tied to the stats smoothing prior: a brawler must have at least as
    many games as the prior's pseudo-count before its own record outweighs the neutral prior."""
    from bsdraft.engine.stats import PRIOR
    assert MIN_RECO_GAMES == float(PRIOR)
