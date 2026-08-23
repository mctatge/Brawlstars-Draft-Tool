"""Readiness: pricing the distance between a player's copy of a brawler and the maxed one the
meta table describes.

Two families of assertion here. The first pins the deficit arithmetic — caps, provenance, the
confidence fade, and the gear-slot rule that stops power being charged twice. The second pins the
*separation* the whole phase exists for: the objective blend must be identical whether or not a
roster is loaded, so the two blind-pick columns print comparable numbers.

    PYTHONPATH=backend python -m pytest backend/tests/test_readiness.py
"""
from __future__ import annotations

import dataclasses

import pytest

from bsdraft.api import schemas as S
from bsdraft.engine import readiness as RD
from bsdraft.engine.readiness import Fielded, readiness
from bsdraft.engine.scoring import DEFAULT_WEIGHTS, PickScore, score_candidate
from bsdraft.engine.state import DraftState

SHELLY, COLT, BULL = 16000000, 16000001, 16000002
MODE, MAP = "Brawl Ball", 15000001


class _Rate:
    def __init__(self, winrate, confidence=1.0, games=100):
        self.winrate, self.confidence, self.games = winrate, confidence, games


class _StubStats:
    def __init__(self, map_rates, counter_rates=None):
        self._map, self._counter = map_rates, counter_rates or {}

    def brawler_rate(self, bid, map_id):
        return _Rate(self._map.get(bid, 0.5))

    def synergy(self, a, b):
        return _Rate(0.5)

    def counter(self, a, b):
        return _Rate(self._counter.get(a, 0.5))

    def top_brawlers(self, map_id, n=40, min_games=3):
        return []


class _Entry:
    """Roster stand-in carrying a readiness view, like the API's _ReqMastery."""

    def __init__(self, fielded: Fielded, score: float = 0.5):
        self.score = score
        self._f = fielded

    def gaps(self):
        return []

    def fielded(self):
        return self._f


def _score(cand, stats, roster):
    state = DraftState(map_id=MAP, mode=MODE, our_team=[], their_team=[BULL])
    return score_candidate(state, cand, stats, model=None, weights=DEFAULT_WEIGHTS, roster=roster)


# --------------------------------------------------------------------- the deficit arithmetic


def test_a_fully_fielded_copy_takes_no_deficit():
    deficit, reasons = readiness(Fielded.ready())
    assert deficit == 0.0 and reasons == []


def test_measured_power_deficit_is_charged_and_labelled_measured():
    d9, r9 = readiness(Fielded(power=9))
    assert d9 == pytest.approx(RD.power_deficit_table()[9])
    assert [x.source for x in r9] == [RD.MEASURED]
    assert r9[0].label == "P9"


def test_deficit_is_monotone_in_power():
    d = [readiness(Fielded(power=p))[0] for p in (11, 10, 9)]
    assert d[0] <= d[1] <= d[2]


def test_unknown_power_is_not_a_penalty():
    """power == 0 is the wire default for a client that predates the field. The floor gate keeps
    such entries deliberately; charging them a maximal deficit would turn a compatibility shim
    into a scoring penalty."""
    deficit, reasons = readiness(Fielded(power=0, n_gears=0))
    assert deficit == 0.0
    assert not any(x.label.startswith("P") for x in reasons)
    assert not any("gear" in x.label for x in reasons)


def test_hypercharge_is_surfaced_but_unpriced():
    """Battle logs carry no hypercharge field, so no estimator exists. It must be visible and
    worth exactly nothing — not silently dropped, and not guessed at."""
    deficit, reasons = readiness(Fielded(has_hypercharge=False))
    assert deficit == 0.0
    hc = [x for x in reasons if x.label == RD.GAP_NO_HYPERCHARGE]
    assert len(hc) == 1 and hc[0].points == 0.0 and hc[0].source == RD.UNPRICED
    # +0.0, not -0.0: the latter compares equal but serializes as "-0.0".
    assert str(hc[0].points) == "0.0"


def test_missing_loadout_is_estimated_not_measured():
    _, reasons = readiness(Fielded(has_starpower=False, has_gadget=False))
    assert {x.source for x in reasons} == {RD.ESTIMATED}


def test_no_declared_prior_can_outrank_a_measurement():
    """The ordering invariant: a hand-set guess must never move a score more than a measured
    effect, or the labels stop meaning anything."""
    assert RD._ordering_invariant_holds()


def test_gear_slots_that_power_has_not_unlocked_are_not_charged():
    """Ranked opens the second gear slot at Power 10. Charging a Power 9 copy for an empty second
    slot would bill the same shortfall twice — once as power, once as a missing gear."""
    p9_one_gear = readiness(Fielded(power=9, n_gears=1))[0]
    assert p9_one_gear == pytest.approx(RD.power_deficit_table()[9])
    # ...while a Power 11 copy with one gear IS missing a slot it could fill.
    p11_one_gear = readiness(Fielded(power=11, n_gears=1))[0]
    assert p11_one_gear == pytest.approx(RD.MISSING_GEAR_SLOT)


def test_reasons_sum_to_the_deficit():
    """The chips a user sees must add up to the movement in the number."""
    for f in (Fielded(power=9, has_starpower=False, has_hypercharge=False),
              Fielded(power=10, has_gadget=False, n_gears=0),
              Fielded(power=11, has_starpower=False, has_gadget=False, n_gears=0)):
        deficit, reasons = readiness(f)
        assert sum(x.points for x in reasons) == pytest.approx(-deficit)


def test_deficit_never_exceeds_the_cap():
    worst = Fielded(power=1, has_starpower=False, has_gadget=False, n_gears=0, has_hypercharge=False)
    assert readiness(worst)[0] <= RD.READY_CAP


def test_confidence_fades_the_deficit_out():
    """Your own record on this copy already contains the handicap, so charging it again
    double-counts. At full confidence the correction vanishes entirely."""
    f = Fielded(power=9)
    full = readiness(f, 0.0)[0]
    half = readiness(f, 0.5)[0]
    none = readiness(f, 1.0)[0]
    assert full > half > none == 0.0
    assert half == pytest.approx(full * 0.5)


def test_a_missing_readiness_view_is_zero_not_maximal():
    """Duck-typed roster entries without .fielded() (older stubs, other hosts) must degrade to
    'no information', never to 'maximally under-built'."""
    assert readiness(None) == (0.0, [])


# --------------------------------------------------------------------- the separation


def test_base_score_is_identical_with_and_without_a_roster():
    """The property that makes the meta and roster columns comparable. Before this phase the
    personalized read renormalized over a larger denominator and printed a higher number for the
    same board."""
    stats = _StubStats({SHELLY: 0.62}, {SHELLY: 0.58})
    meta = _score(SHELLY, stats, None)
    personal = _score(SHELLY, stats, {SHELLY: _Entry(Fielded.ready(), score=1.0)})
    assert meta.base_score == personal.base_score


def test_a_ready_brawler_scores_exactly_its_base():
    stats = _StubStats({SHELLY: 0.62}, {SHELLY: 0.58})
    p = _score(SHELLY, stats, {SHELLY: _Entry(Fielded.ready())})
    assert p.readiness == 0.0 and p.item_edge == 0.0 and p.history_edge == 0.0
    assert p.score == p.base_score


def test_under_leveled_copies_score_strictly_below_the_base():
    stats = _StubStats({SHELLY: 0.62}, {SHELLY: 0.58})
    base = _score(SHELLY, stats, None).base_score
    p9 = _score(SHELLY, stats, {SHELLY: _Entry(Fielded(power=9))})
    p10 = _score(SHELLY, stats, {SHELLY: _Entry(Fielded(power=10))})
    p11 = _score(SHELLY, stats, {SHELLY: _Entry(Fielded(power=11))})
    assert p9.score < p10.score < p11.score == base
    assert p9.score == pytest.approx(base - RD.power_deficit_table()[9])


def test_score_stays_inside_zero_and_one():
    """Adding signed adjustments to a weighted average makes the sum unbounded, and the frontend
    renders it as a percentage — an uncapped deficit could print a negative win rate."""
    stats = _StubStats({SHELLY: 0.02})
    worst = Fielded(power=1, has_starpower=False, has_gadget=False, n_gears=0, has_hypercharge=False)
    p = _score(SHELLY, stats, {SHELLY: _Entry(worst)})
    assert 0.0 <= p.score <= 1.0


def test_item_edge_is_inert_until_the_table_exists():
    """The seam ships wired but contributing exactly nothing — itemstats.json has never been
    built. Bit-identical, not approximately identical."""
    assert RD.item_edge() is None
    stats = _StubStats({SHELLY: 0.62})
    p = _score(SHELLY, stats, {SHELLY: _Entry(Fielded.ready())})
    assert p.item_edge == 0.0
    assert p.score == p.base_score


# --------------------------------------------------------------------- the player's record


class _StubPersonal:
    """Personal stats with a fixed per-brawler edge over the baseline and a chosen overall rate."""

    def __init__(self, edges, games=40.0, overall=0.5):
        self._edges, self._games, self._overall = edges, games, overall

    def brawler_rate(self, bid, map_id=None):
        return _Rate(0.5 + self._edges.get(bid, 0.0), confidence=0.8, games=self._games)

    def baseline_rate(self, bid, map_id=None):
        return 0.5

    def overall_rate(self):
        return _Rate(self._overall, confidence=0.8, games=200.0)


def _with_personal(cand, stats, personal):
    state = DraftState(map_id=MAP, mode=MODE, our_team=[], their_team=[BULL])
    return score_candidate(state, cand, stats, model=None, weights=DEFAULT_WEIGHTS,
                           roster=None, personal=personal)


def test_history_edge_is_capped():
    """A handful of lucky games must not float a brawler up the board. On a real account a 4-game
    record at 97% raw shrinks to +8.1 points of raw edge — the cap holds it to +2.0."""
    stats = _StubStats({SHELLY: 0.55})
    p = _with_personal(SHELLY, stats, _StubPersonal({SHELLY: 0.40}))
    assert p.history_edge == pytest.approx(RD.HISTORY_CAP)
    n = _with_personal(SHELLY, stats, _StubPersonal({SHELLY: -0.40}))
    assert n.history_edge == pytest.approx(-RD.HISTORY_CAP)


def test_history_edge_ignores_the_players_overall_rate():
    """Netting a per-player constant out of every candidate cannot reorder the list — it only moves
    the level. Folding it in silently added ~5.5 points to every row on a 40%-overall account, so
    the overall rate is a header fact, not a per-pick adjustment."""
    stats = _StubStats({SHELLY: 0.55})
    weak = _with_personal(SHELLY, stats, _StubPersonal({SHELLY: 0.01}, overall=0.40))
    strong = _with_personal(SHELLY, stats, _StubPersonal({SHELLY: 0.01}, overall=0.60))
    assert weak.history_edge == strong.history_edge
    assert weak.score == strong.score


def test_an_unplayed_brawler_gets_no_history_adjustment():
    stats = _StubStats({SHELLY: 0.55})
    p = _with_personal(SHELLY, stats, _StubPersonal({}, games=0.0))
    assert p.history_edge == 0.0 and p.score == p.base_score


def test_history_edge_stays_well_below_the_measured_power_effect():
    """An unvalidated product knob must not rival a measured one, or the provenance labels stop
    meaning anything to a reader comparing two chips."""
    assert RD.HISTORY_CAP < RD.power_deficit_table()[9]


# --------------------------------------------------------------------- the wire


def test_every_pickscore_field_survives_onto_the_wire():
    """/api/recommend builds its response with ``PickRec(**vars(p))`` and pydantic ignores unknown
    keys, so a PickScore field with no PickRec counterpart is dropped with no error, no warning and
    no test failure. This is that test."""
    engine_fields = {f.name for f in dataclasses.fields(PickScore)}
    wire_fields = set(S.PickRec.model_fields)
    missing = engine_fields - wire_fields
    assert not missing, f"PickScore fields silently dropped from the API response: {sorted(missing)}"


def test_readiness_reasons_serialize_through_pydantic():
    """The reasons are frozen dataclasses, not dicts — the wire model needs from_attributes or the
    splat raises at request time rather than at import."""
    _, reasons = readiness(Fielded(power=9, has_hypercharge=False))
    rec = S.PickRec(brawler_id=SHELLY, name="Shelly", cls="Damage Dealer", score=0.5,
                    map_winrate=0.5, role_fit=0.5, confidence=1.0, breakdown={},
                    readiness_reasons=reasons)
    assert [r.label for r in rec.readiness_reasons] == ["P9", RD.GAP_NO_HYPERCHARGE]
    assert [r.source for r in rec.readiness_reasons] == [RD.MEASURED, RD.UNPRICED]


def test_api_roster_stand_ins_all_expose_a_readiness_view():
    """The roster dict holds three different mastery-likes depending on host and source. A missing
    .fielded() on any of them silently drops readiness for that whole path."""
    import bsdraft.api.main as M
    from bsdraft.engine.mastery import Mastery

    req = M._ReqMastery(0.5, [RD.GAP_NO_STAR_POWER], power=9, n_gears=1)
    assert req.fielded() == Fielded(power=9, has_starpower=False, has_gadget=True,
                                    n_gears=1, has_hypercharge=True)

    # A boosted brawler arrives fully maxed, so it must take no deficit at all.
    assert M._BoostedMastery().fielded() == Fielded.ready()
    assert readiness(M._BoostedMastery().fielded())[0] == 0.0

    m = Mastery(brawler_id=SHELLY, power=9, rank=10, trophies=500, highest_trophies=500,
                has_starpower=True, has_gadget=True, has_gears=True, has_hypercharge=False)
    assert m.fielded().power == 9 and m.fielded().has_hypercharge is False


def test_roster_schema_is_declared_and_monotonic():
    assert isinstance(S.ROSTER_SCHEMA, int) and S.ROSTER_SCHEMA >= 2
    assert S.RosterResponse(loaded=True, tag="X", name="Y").roster_schema == S.ROSTER_SCHEMA


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
