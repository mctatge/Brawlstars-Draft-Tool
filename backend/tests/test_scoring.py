"""Unit tests for the pick-scoring signal fusion (bsdraft.engine.scoring).

Guards the separation of the objective blend from personalization. The five ablation-tuned signals
produce a ``base_score``; personalization applies afterwards as signed win-rate-point adjustments
(:mod:`bsdraft.engine.readiness`). So a clearly stronger *meta* pick outranks a maxed-out *comfort*
pick, and — stronger than the 2026-08-17 de-weight this replaces — mastery cannot move a score at
all. Stub stats keep the assertions off the live dataset.

    PYTHONPATH=backend python -m pytest backend/tests/test_scoring.py
"""
from __future__ import annotations

from bsdraft.data import reference as R
from bsdraft.engine.engine import DraftEngine
from bsdraft.engine.mastery import Mastery
from bsdraft.engine.scoring import DEFAULT_WEIGHTS, _class_of, role_fit, score_candidate
from bsdraft.engine.state import DraftState
from bsdraft.engine.stats import DraftStats

# Shelly and Colt are both Damage Dealers, so role_fit is identical for the two candidates
# and cancels out of every head-to-head below — the comparisons turn purely on map/counter/mastery.
SHELLY, COLT, BULL = 16000000, 16000001, 16000002
JESSIE = 16000007          # Controller — high mode-archetype fit in Gem Grab (role tests below)
MODE, MAP = "Brawl Ball", 15000001


class _Rate:
    def __init__(self, winrate, confidence=1.0, games=100):
        self.winrate = winrate
        self.confidence = confidence
        self.games = games


class _StubStats:
    """Fixed per-brawler map win rate and counter rate; synergy neutral, no model completion."""

    def __init__(self, map_rates, counter_rates):
        self._map = map_rates
        self._counter = counter_rates

    def brawler_rate(self, bid, map_id):
        return _Rate(self._map.get(bid, 0.5))

    def synergy(self, a, b):
        return _Rate(0.5)

    def counter(self, a, b):
        return _Rate(self._counter.get(a, 0.5))

    def top_brawlers(self, map_id, n=40, min_games=3):
        return []


class _Mastery:
    def __init__(self, score):
        self.score = score

    def gaps(self):
        return []


def _score(cand, stats, roster, weights):
    # One enemy drafted so the `counter` signal is live (it's absent from an empty board).
    state = DraftState(map_id=MAP, mode=MODE, our_team=[], their_team=[BULL])
    return score_candidate(state, cand, stats, model=None, weights=weights, roster=roster).score


def test_strong_meta_beats_maxed_comfort_under_new_weights():
    # SHELLY: strong on the map and into the enemy, but the player has never built her (mastery 0).
    # COLT: weak on both, but maxed mastery. The meta pick must now win.
    stats = _StubStats(map_rates={SHELLY: 0.70, COLT: 0.45},
                       counter_rates={SHELLY: 0.70, COLT: 0.45})
    roster = {SHELLY: _Mastery(0.0), COLT: _Mastery(1.0)}
    shelly = _score(SHELLY, stats, roster, DEFAULT_WEIGHTS)
    colt = _score(COLT, stats, roster, DEFAULT_WEIGHTS)
    assert shelly > colt, f"meta pick should win now (shelly={shelly:.3f} colt={colt:.3f})"


def test_mastery_cannot_invert_a_ranking_at_any_value():
    # The de-weight became a removal: mastery is an investment index, not a win rate, so it no
    # longer multiplies into the score at all. Sweeping it across its whole range must not flip a
    # ranking the objective signals already decided. (Pre-2026-08-17 this inverted at mastery .25.)
    stats = _StubStats(map_rates={SHELLY: 0.70, COLT: 0.45},
                       counter_rates={SHELLY: 0.70, COLT: 0.45})
    for colt_mastery in (0.0, 0.25, 0.5, 0.75, 1.0):
        roster = {SHELLY: _Mastery(0.0), COLT: _Mastery(colt_mastery)}
        shelly = _score(SHELLY, stats, roster, DEFAULT_WEIGHTS)
        colt = _score(COLT, stats, roster, DEFAULT_WEIGHTS)
        assert shelly > colt, f"mastery {colt_mastery} moved the ranking"


def test_mastery_does_not_break_ties():
    # The inverse of the old guard, and deliberately strict: equal objective signals must produce
    # BIT-IDENTICAL scores however differently the two brawlers are built. A tie-break here would
    # mean mastery is still leaking into the number.
    stats = _StubStats(map_rates={SHELLY: 0.55, COLT: 0.55},
                       counter_rates={SHELLY: 0.55, COLT: 0.55})
    roster = {SHELLY: _Mastery(0.8), COLT: _Mastery(0.2)}
    assert _score(SHELLY, stats, roster, DEFAULT_WEIGHTS) == _score(COLT, stats, roster, DEFAULT_WEIGHTS)


def test_roster_presence_alone_does_not_move_the_score():
    # Loading a roster changes WHICH brawlers are candidates, never what a candidate scores. This is
    # what makes the meta and roster columns comparable — before this phase the personalized read
    # renormalized over a different denominator and printed a higher number for the same board.
    stats = _StubStats(map_rates={SHELLY: 0.62}, counter_rates={SHELLY: 0.58})
    without = _score(SHELLY, stats, None, DEFAULT_WEIGHTS)
    with_roster = _score(SHELLY, stats, {SHELLY: _Mastery(1.0)}, DEFAULT_WEIGHTS)
    assert without == with_roster


# --- B1: what the mastery metric actually is, and its default -----------------------------------
# The reported "Leon at ~100" is the mastery score (a 0..1 investment number) shown as a percentage
# — a maxed, comfortable brawler is 1.0 == "100%", not a bug. And a brawler the player does NOT own
# contributes *no* mastery signal (None, dropped from the fused average) — it is never defaulted to
# 0.5. These pin both facts so a future refactor can't silently reintroduce a 0.5 default.

def test_mastery_is_zero_to_one_and_maxes_at_100_percent():
    # 0.60*build + 0.40*comfort, clamped to [0,1]. build maxes when star power + gadget + gears are
    # owned; comfort maxes at 1000+ highest-trophies. A fully-built, comfortable brawler -> 1.0,
    # which a percentage display renders as "100" (the Leon-at-~100 sighting). An owned-but-unbuilt
    # brawler is 0.0 — the low end, not a 0.5 midpoint.
    maxed = Mastery(brawler_id=1, power=11, rank=30, trophies=1200, highest_trophies=1200,
                    has_starpower=True, has_gadget=True, has_gears=True, has_hypercharge=True)
    unbuilt = Mastery(brawler_id=2, power=11, rank=0, trophies=0, highest_trophies=0,
                      has_starpower=False, has_gadget=False, has_gears=False, has_hypercharge=False)
    assert maxed.score == 1.0 and round(maxed.score * 100) == 100
    assert unbuilt.score == 0.0
    partial = Mastery(brawler_id=3, power=9, rank=10, trophies=500, highest_trophies=500,
                      has_starpower=True, has_gadget=False, has_gears=False, has_hypercharge=False)
    assert 0.0 <= partial.score <= 1.0


def test_unowned_brawler_carries_no_mastery_signal_never_half():
    # A candidate absent from the roster: owned=False, mastery=None. The one on the roster carries
    # its real 0..1 score for display. Nothing is ever defaulted to 0.5.
    # `breakdown` now holds objective parts ONLY — every value in it is a win-rate-shaped [0,1]
    # quantity, which is precisely why mastery no longer belongs there.
    stats = _StubStats(map_rates={SHELLY: 0.6, COLT: 0.6}, counter_rates={})
    roster = {SHELLY: _Mastery(0.8)}                    # COLT deliberately unowned
    state = DraftState(map_id=MAP, mode=MODE)
    shelly = score_candidate(state, SHELLY, stats, model=None, weights=DEFAULT_WEIGHTS, roster=roster)
    colt = score_candidate(state, COLT, stats, model=None, weights=DEFAULT_WEIGHTS, roster=roster)
    assert shelly.owned is True and shelly.mastery == 0.8
    assert colt.owned is False and colt.mastery is None
    assert "mastery" not in shelly.breakdown and "mastery" not in colt.breakdown
    assert set(shelly.breakdown) <= set(DEFAULT_WEIGHTS)


# --- B2: teammate bans adjust the pool, symmetrically with enemy bans ----------------------------
# The backend keeps ONE flat ban list with no ally/enemy tag, so a teammate's ban and an enemy's
# ban both drop the brawler from the pick pool through the same `picked_or_banned()` union. Guards
# the reported "teammate bans don't adjust the pool" regression from coming back.

def test_bans_remove_candidates_symmetrically_regardless_of_side():
    engine = DraftEngine(stats=DraftStats(matches=[]))          # empty stats: candidates() needs none
    ids = [b.id for b in R.load_brawlers()][:6]
    a, b, c = ids[0], ids[1], ids[2]
    st = DraftState(map_id=MAP, mode=MODE, bans=[a, b])
    assert {a, b} <= st.picked_or_banned()                     # every ban is unioned, side-agnostic
    cands = set(engine.candidates(st))
    assert a not in cands and b not in cands
    # A third ban (e.g. the last teammate ban to land) is removed too — order- and side-free.
    st2 = DraftState(map_id=MAP, mode=MODE, bans=[c, a, b])
    assert set(engine.candidates(st2)).isdisjoint({a, b, c})


# --- role_fit confidence-scaling: role fades where real per-map data exists ----------------------
# role_eff = 0.5 + (1 - conf)*(role_fit - 0.5), conf = the candidate's per-(brawler, map) map_wr
# confidence. On a well-sampled map role_eff -> 0.5 for every candidate (a constant that can't
# re-order picks); on a zero-data map it stays the full archetype prior. Fixes the mode+class
# heuristic punching ~2.5x above its 0.10 weight and flat-topping mode-archetype brawlers.

class _ConfStats:
    """Stub stats with a per-brawler map win rate AND a controllable confidence, to drive the
    confidence-scaled role term. Synergy/counter neutral; no model completion."""

    def __init__(self, map_rates, confidence):
        self._map = map_rates
        self._conf = confidence

    def brawler_rate(self, bid, map_id):
        return _Rate(self._map.get(bid, 0.5), confidence=self._conf)

    def synergy(self, a, b):
        return _Rate(0.5)

    def counter(self, a, b):
        return _Rate(0.5)

    def top_brawlers(self, map_id, n=40, min_games=3):
        return []


def test_role_is_confidence_scaled_toward_neutral():
    state = DraftState(map_id=MAP, mode="Gem Grab")   # Jessie is a Controller -> high role_fit here
    rfit = role_fit(state, _class_of(JESSIE))
    assert rfit > 0.5, "premise: Jessie must carry an above-neutral archetype fit in Gem Grab"

    # Zero map data -> role_eff is the full archetype prior (a fresh/rotated map keeps it).
    fresh = score_candidate(state, JESSIE, _ConfStats({JESSIE: 0.5}, confidence=0.0), model=None)
    assert fresh.breakdown["role"] == round(rfit, 3)
    # Fully-sampled map -> role_eff collapses to neutral 0.5 (drops out of the ranking).
    sampled = score_candidate(state, JESSIE, _ConfStats({JESSIE: 0.5}, confidence=1.0), model=None)
    assert sampled.breakdown["role"] == 0.5
    # Half confidence -> linear interpolation between prior and neutral.
    half = score_candidate(state, JESSIE, _ConfStats({JESSIE: 0.5}, confidence=0.5), model=None)
    assert half.breakdown["role"] == round(0.5 + 0.5 * (rfit - 0.5), 3)
    # The raw role_fit field is preserved un-scaled for explainability, whatever the confidence.
    assert sampled.role_fit == rfit == fresh.role_fit


def test_role_stops_separating_picks_on_a_well_sampled_map():
    # Two brawlers, different mode-archetype fit, identical map win-rate. On a zero-data map the
    # higher-role brawler wins on the archetype prior; on a fully-sampled map role_eff is a constant
    # 0.5 for both, so the archetype no longer re-orders them — they tie on the (equal) map signal.
    state = DraftState(map_id=MAP, mode="Gem Grab")
    hi, lo = JESSIE, COLT                              # Controller (0.80) vs Damage Dealer (0.60)
    assert role_fit(state, _class_of(hi)) > role_fit(state, _class_of(lo))
    equal_map = {hi: 0.5, lo: 0.5}
    fresh = _ConfStats(equal_map, confidence=0.0)
    assert score_candidate(state, hi, fresh, None).score > score_candidate(state, lo, fresh, None).score
    sampled = _ConfStats(equal_map, confidence=1.0)
    assert score_candidate(state, hi, sampled, None).score == score_candidate(state, lo, sampled, None).score
