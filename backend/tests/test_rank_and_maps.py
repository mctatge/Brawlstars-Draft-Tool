"""Rank resolution across a season reset, and the ranked-map list the board offers.

Both regressions surfaced on the 2026-08-20 season flip: the board showed a player their
*previous* season's tier as fact, and offered map/mode pairs (famously "Heist: Pit Stop") that
Ranked hasn't rotated in.
"""
from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

from bsdraft.api import main as M
from bsdraft.data import reference as R


class _FakeClient:
    """Stands in for BrawlStarsClient: an async context manager whose get_player either
    returns a profile dict or raises (the IP-locked-key / offline case)."""

    def __init__(self, player=None, exc=None):
        self._player, self._exc = player, exc

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get_player(self, tag):
        if self._exc is not None:
            raise self._exc
        return self._player


@pytest.fixture(autouse=True)
def _clear_rank_cache():
    M._rank_cache.clear()
    yield
    M._rank_cache.clear()


def _client(monkeypatch, *, token, player=None, exc=None, dataset_tier=None):
    monkeypatch.setattr(M.settings, "brawlstars_api_token", token)
    monkeypatch.setattr(M, "BrawlStarsClient", _FakeClient(player=player, exc=exc))
    monkeypatch.setattr(M, "_rank_index",
                        lambda: types.SimpleNamespace(get=lambda t: dataset_tier))
    return TestClient(M.app)


def test_live_tier_wins_and_is_not_marked_stale(monkeypatch):
    c = _client(monkeypatch, token="k", player={"rankedRank": 4}, dataset_tier=16)
    r = c.get("/api/rank", params={"tag": "#ABC123"}).json()
    assert r["found"] and r["tier"] == 4 and r["source"] == "live"
    assert r["stale"] is False          # a live read is current by construction


def test_unplaced_this_season_does_not_fall_back_to_the_pre_reset_row(monkeypatch):
    # The 2026-08-20 bug: the profile loads fine and carries no rankedRank (the season just
    # reset and the player hasn't placed), and the crawl still holds their old Legendary I row.
    # Falling through to that row reports a tier they no longer hold — the live lookup already
    # answered the question, so the dataset must not get a say.
    c = _client(monkeypatch, token="k", player={"name": "x"}, dataset_tier=16)
    r = c.get("/api/rank", params={"tag": "#ABC123"}).json()
    assert r["found"] is False
    assert r["source"] == "live"
    assert r["tier"] is None and r["bracket"] is None
    assert "season" in (r["error"] or "")


def test_unreachable_live_lookup_serves_the_dataset_but_flags_it(monkeypatch):
    # An IP-lock 403 teaches us nothing about the player, so the crawl row is still the best
    # guess — but it may predate a reset, so it goes out marked rather than as fact.
    c = _client(monkeypatch, token="k", exc=RuntimeError("HTTP 403: auth/IP error"),
                dataset_tier=16)
    r = c.get("/api/rank", params={"tag": "#ABC123"}).json()
    assert r["found"] and r["tier"] == 16 and r["source"] == "dataset"
    assert r["stale"] is True


def test_keyless_host_serves_the_dataset_unflagged(monkeypatch):
    # On the public host there is no live check to miss, so the dataset isn't "stale" in the
    # sense the flag means — nothing failed.
    c = _client(monkeypatch, token="", dataset_tier=16)
    r = c.get("/api/rank", params={"tag": "#ABC123"}).json()
    assert r["found"] and r["source"] == "dataset" and r["stale"] is False


def test_blank_tag_is_a_prompt_not_an_error(monkeypatch):
    c = _client(monkeypatch, token="k", dataset_tier=16)
    r = c.get("/api/rank", params={"tag": "  "}).json()
    assert r["found"] is False and r["error"]


# --------------------------------------------------------------------------- ranked map list

def _engine_with_map_games(map_games, recent=None):
    return types.SimpleNamespace(
        stats=types.SimpleNamespace(map_games=map_games, map_games_recent=recent or {}),
        bracket_stats={},
        roster=None,
    )


def test_reference_offers_only_maps_ranked_actually_rotates(monkeypatch):
    # The catalog's not-retired set spans every mode's whole map pool; Ranked rotates a handful.
    # Collected games are the only rotation signal we have.
    all_maps = R.load_ranked_maps()
    played = {m.id: 500 for m in all_maps[:6]}
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(played))
    served = TestClient(M.app).get("/api/reference").json()["maps"]
    assert {m["id"] for m in served} == set(played)
    assert len(served) < len(all_maps)
    assert all(m["games"] > 0 for m in served)


def test_a_retired_map_with_leftover_games_is_still_dropped(monkeypatch):
    # The bug in the first cut: "any games at all" readmits maps from earlier rotations, which
    # keep a decaying residue in stats that span more history than one season. Observed
    # 2026-08-20 — Heist's live four sat at 1954-2026 games while retired Pit Stop had 90, and
    # a games > 0 test happily served it back.
    by_mode = {}
    for m in R.load_ranked_maps():
        by_mode.setdefault(m.mode, []).append(m)
    mode = next(k for k, v in by_mode.items() if len(v) >= 5)
    live, retired = by_mode[mode][:4], by_mode[mode][4]
    games = {m.id: 2000 for m in live}
    games[retired.id] = 90          # ~4% of the leader: a previous rotation's leftovers
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(games))
    served = {m["id"] for m in TestClient(M.app).get("/api/reference").json()["maps"]}
    assert served == {m.id for m in live}
    assert retired.id not in served


def test_a_quiet_but_current_map_survives_the_cut(monkeypatch):
    # The threshold has to tolerate ordinary imbalance within a live rotation — some maps in the
    # pool genuinely draw fewer games. Only an order-of-magnitude gap should read as "retired".
    by_mode = {}
    for m in R.load_ranked_maps():
        by_mode.setdefault(m.mode, []).append(m)
    mode = next(k for k, v in by_mode.items() if len(v) >= 4)
    pool = by_mode[mode][:4]
    games = {m.id: 2000 for m in pool}
    games[pool[-1].id] = 700        # 35% of the leader — quiet, not gone
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(games))
    served = {m["id"] for m in TestClient(M.app).get("/api/reference").json()["maps"]}
    assert served == {m.id for m in pool}


def test_the_cut_is_per_mode_not_global(monkeypatch):
    # Modes draw wildly different volumes. A single global threshold would wipe out an entire
    # quiet mode's rotation while keeping a busy mode's leftovers.
    by_mode = {}
    for m in R.load_ranked_maps():
        by_mode.setdefault(m.mode, []).append(m)
    busy, quiet = list(by_mode)[:2]
    games = {m.id: 5000 for m in by_mode[busy][:3]}
    games.update({m.id: 300 for m in by_mode[quiet][:3]})   # 6% of the busy mode's leader
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(games))
    served = {m["id"] for m in TestClient(M.app).get("/api/reference").json()["maps"]}
    assert served == set(games)      # the quiet mode keeps its whole rotation


def test_a_map_added_mid_rotation_appears_on_the_recent_signal(monkeypatch):
    # The 2026-08-25 flip: Brawl Ball gained Beach Ball + Pinhole Punt at ~18:00 UTC. Their
    # cumulative games were ~0.03% of the leader's 24k — the cumulative cut alone would hide
    # them for days — but a few hours of crawl already fills the recent window.
    by_mode = {}
    for m in R.load_ranked_maps():
        by_mode.setdefault(m.mode, []).append(m)
    mode = next(k for k, v in by_mode.items() if len(v) >= 6)
    old, added = by_mode[mode][:5], by_mode[mode][5]
    games = {m.id: 24000 for m in old}
    games[added.id] = 40                                  # hours old: nothing accumulated yet
    recent = {m.id: 5000 for m in old}
    recent[added.id] = 300                                # ~6% of the window leader — ramping
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(games, recent))
    served = {m["id"] for m in TestClient(M.app).get("/api/reference").json()["maps"]}
    assert added.id in served
    assert {m.id for m in old} <= served


def test_a_map_dropped_from_rotation_leaves_with_the_window(monkeypatch):
    # The reverse direction: a dropped map keeps a large cumulative residue (half-life 21 days —
    # ~8 weeks to decay below the cumulative cut) but its recent window empties within days.
    by_mode = {}
    for m in R.load_ranked_maps():
        by_mode.setdefault(m.mode, []).append(m)
    mode = next(k for k, v in by_mode.items() if len(v) >= 5)
    live, dropped = by_mode[mode][:4], by_mode[mode][4]
    games = {m.id: 20000 for m in live}
    games[dropped.id] = 18000                             # still ~90% of the leader cumulatively
    recent = {m.id: 5000 for m in live}
    recent[dropped.id] = 12                               # under the absolute floor: gone
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(games, recent))
    served = {m["id"] for m in TestClient(M.app).get("/api/reference").json()["maps"]}
    assert dropped.id not in served
    assert served >= {m.id for m in live}


def test_a_too_thin_recent_window_falls_back_to_the_cumulative_cut(monkeypatch):
    # After a crawl outage longer than the window, the window restarts from empty for every map
    # at once. That means "no liveness signal", not "no rotation" — a mode whose window leader
    # is under the trust floor must fall back to the cumulative cut, not serve a thinned list.
    by_mode = {}
    for m in R.load_ranked_maps():
        by_mode.setdefault(m.mode, []).append(m)
    mode = next(k for k, v in by_mode.items() if len(v) >= 4)
    pool = by_mode[mode][:4]
    games = {m.id: 2000 for m in pool}
    recent = {m.id: 3 for m in pool}                      # far under the trust floor
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(games, recent))
    served = {m["id"] for m in TestClient(M.app).get("/api/reference").json()["maps"]}
    assert served >= {m.id for m in pool}


def test_a_partially_refilled_window_does_not_prune_live_maps(monkeypatch):
    # The transient right after an outage: the window is refilling and per-map counts sit in
    # the Poisson noise band, so one live map can clear a naive floor while its same-rate peers
    # straggle under it. The whole window is untrustworthy until its leader clears the trust
    # floor — the mode must serve the cumulative list, never a randomly thinned one.
    by_mode = {}
    for m in R.load_ranked_maps():
        by_mode.setdefault(m.mode, []).append(m)
    mode = next(k for k, v in by_mode.items() if len(v) >= 5)
    pool = by_mode[mode][:5]
    games = {m.id: 2000 for m in pool}                    # cumulative parity: all live
    recent = {pool[0].id: 30, pool[1].id: 13, pool[2].id: 11, pool[3].id: 10, pool[4].id: 8}
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(games, recent))
    served = {m["id"] for m in TestClient(M.app).get("/api/reference").json()["maps"]}
    assert served >= {m.id for m in pool}


def test_the_share_term_prunes_a_sub_share_map_in_a_trusted_window(monkeypatch):
    # Pins the share component of the recent cut: in a trusted window, a map above the absolute
    # floor but under the share-of-leader threshold is dropped, and its cumulative residue does
    # not rescue it (a trusted window overrides the cumulative signal for its whole mode).
    by_mode = {}
    for m in R.load_ranked_maps():
        by_mode.setdefault(m.mode, []).append(m)
    mode = next(k for k, v in by_mode.items() if len(v) >= 5)
    live, fading = by_mode[mode][:4], by_mode[mode][4]
    games = {m.id: 20000 for m in live}
    games[fading.id] = 20000
    recent = {m.id: 5000 for m in live}
    recent[fading.id] = 60      # above the 25 floor, under 0.02 * 5000 = 100
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(games, recent))
    served = {m["id"] for m in TestClient(M.app).get("/api/reference").json()["maps"]}
    assert fading.id not in served
    assert served >= {m.id for m in live}


def test_the_absolute_floor_holds_when_the_share_term_rounds_low(monkeypatch):
    # Pins the floor component: with a trusted-but-modest leader the share term drops below the
    # floor (0.02 * 600 = 12 < 25), and the floor is what separates a live map from stray noise.
    by_mode = {}
    for m in R.load_ranked_maps():
        by_mode.setdefault(m.mode, []).append(m)
    mode = next(k for k, v in by_mode.items() if len(v) >= 5)
    pool, stray = by_mode[mode][:4], by_mode[mode][4]
    games = {m.id: 2000 for m in pool}
    games[stray.id] = 2000
    recent = {m.id: 600 for m in pool}
    recent[stray.id] = 20       # over the share term (12), under the floor (25)
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(games, recent))
    served = {m["id"] for m in TestClient(M.app).get("/api/reference").json()["maps"]}
    assert stray.id not in served
    assert served >= {m.id for m in pool}


def test_recent_and_fallback_modes_coexist_in_one_response(monkeypatch):
    # A rotation flip only touched one mode (2026-08-25: Brawl Ball). That mode reads from its
    # trusted window — including the just-added map — while a mode with no window signal at all
    # keeps serving its cumulative list, in the same request.
    by_mode = {}
    for m in R.load_ranked_maps():
        by_mode.setdefault(m.mode, []).append(m)
    modes = [k for k, v in by_mode.items() if len(v) >= 6]
    flip_mode, quiet_mode = modes[0], modes[1]
    flip_old, flip_new = by_mode[flip_mode][:5], by_mode[flip_mode][5]
    quiet_pool = by_mode[quiet_mode][:4]
    games = {m.id: 24000 for m in flip_old}
    games[flip_new.id] = 40
    games.update({m.id: 2000 for m in quiet_pool})
    recent = {m.id: 5000 for m in flip_old}
    recent[flip_new.id] = 300
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(games, recent))
    served = {m["id"] for m in TestClient(M.app).get("/api/reference").json()["maps"]}
    assert flip_new.id in served
    assert served >= {m.id for m in flip_old}
    assert served >= {m.id for m in quiet_pool}


def test_reference_falls_back_to_the_full_catalog_before_stats_load(monkeypatch):
    # A cold start has no map_games yet. Showing too many maps beats showing none.
    monkeypatch.setattr(M, "_engine", _engine_with_map_games({}))
    served = TestClient(M.app).get("/api/reference").json()["maps"]
    assert len(served) == len(R.load_ranked_maps())


def test_reference_map_ids_stay_a_subset_of_the_model_vocabulary(monkeypatch):
    # The filter lives in the endpoint on purpose: load_ranked_maps() also builds the model's
    # pinned map vocabulary, so narrowing it there would shift embedding rows under the
    # trained checkpoint. Guard that the served list never grows past that vocabulary.
    all_ids = {m.id for m in R.load_ranked_maps()}
    monkeypatch.setattr(M, "_engine",
                        _engine_with_map_games({m.id: 10 for m in R.load_ranked_maps()[:3]}))
    served = TestClient(M.app).get("/api/reference").json()["maps"]
    assert {m["id"] for m in served} <= all_ids
