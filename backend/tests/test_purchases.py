"""Purchase-advisor tests: given an ownership snapshot and a Ranked bracket, the advisor must
recommend the *unowned* purchases that matter, price each one as the full package it needs
(power climb to the bracket's power floor, a core build, the unlock), value the package
additively, and rank by win-rate value per coin-equivalent — so a star power on a Power-3
brawler you can't even field never outranks a 1,000-coin gadget on a maxed meta brawler.

Runs on a synthetic catalog + hand-checkable stats so every expected direction is known in
advance, with the economy table inlined so cost math doesn't depend on the shipped JSON.

    PYTHONPATH=backend python -m pytest backend/tests/test_purchases.py   # or run directly
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Tuple

from bsdraft.engine import purchases as P
from bsdraft.engine.purchases import OwnedState


# --- synthetic catalog -----------------------------------------------------------------------

@dataclass(frozen=True)
class _Acc:
    id: int
    name: str
    kind: str


@dataclass(frozen=True)
class _Brawler:
    id: int
    name: str
    cls: str
    rarity: str
    gadgets: Tuple[_Acc, ...] = ()
    star_powers: Tuple[_Acc, ...] = ()


def _two(i, kind):
    k = "gadget" if kind == "g" else "star_power"
    return (_Acc(i * 100 + (1 if kind == "g" else 3), f"{kind}{i}a", k),
            _Acc(i * 100 + (2 if kind == "g" else 4), f"{kind}{i}b", k))


CATALOG = [
    _Brawler(1, "Strong", "Assassin", "Legendary", gadgets=_two(1, "g"), star_powers=_two(1, "s")),
    _Brawler(2, "Mid", "Support", "Epic", gadgets=_two(2, "g"), star_powers=_two(2, "s")),
    _Brawler(3, "Fresh", "Marksman", "Mythic",
             gadgets=(_Acc(301, "g3a", "gadget"),), star_powers=(_Acc(303, "s3a", "star_power"),)),
]
# ids of brawler 1's items, for readability
G1A, G1B, S1A, S1B = 101, 102, 103, 104
G2A, G2B, S2A, S2B = 201, 202, 203, 204


class _Ref:
    def load_brawlers(self):
        return CATALOG

    def load_ranked_boosted(self):
        return ()


@dataclass
class _Rate:
    winrate: float
    games: float = 200.0


class _Stats:
    """Per-brawler win rate (same for every map, so meta_strength == the brawler's rate); ``games``
    overrides the per-brawler effective sample (default 200)."""
    def __init__(self, rates, games=None):
        self.rates = rates
        self.games = games or {}

    def brawler_rate(self, brawler_id, map_id=None):
        return _Rate(self.rates.get(brawler_id, 0.5), self.games.get(brawler_id, 200.0))


@dataclass
class _Map:
    id: int
    mode: str = "Gem Grab"


RANKED_MAPS = [_Map(10, "Gem Grab"), _Map(11, "Gem Grab"), _Map(12, "Heist")]

# Economy inlined — mirrors data/reference/economy.json so cost assertions are deterministic.
CUM_PP = [0, 0, 20, 50, 100, 180, 310, 520, 860, 1410, 2300, 3740]
CUM_COINS = [0, 0, 20, 55, 130, 270, 560, 1040, 1840, 3090, 4965, 7765]
PRIORS = {
    "gadget_first": 3.0, "gadget_second": 0.6, "star_power_first": 3.5, "star_power_second": 0.6,
    "gear_core": 0.7, "gear_extra": 0.25, "hypercharge": 6.0, "power_level": 1.5,
    "access": 3.0, "readiness": 0.5,
}
SCORING = {
    "meta_center": 0.50, "meta_scale": 0.125,
    "resource_weights": {"coins": 1.0, "power_points": 0.5, "credits": 1.0},
    "depth_free": 8, "depth_k": 12, "mode_shrink_games": 300, "delta_scale": 0.10,
    "boosted_discount": 0.5, "unknown_floor": 11,
}
ECON = {
    "power_cost_cumulative": {"power_points": CUM_PP, "coins": CUM_COINS},
    "item_costs": {
        "gadget": {"coins": 1000}, "star_power": {"coins": 2000}, "gear": {"coins": 1000},
        "hypercharge": {"coins": 5000},
    },
    "new_brawler_credits": {"Epic": 925, "Mythic": 1900, "Legendary": 3800, "Ultra Legendary": 5500},
    "power_gates": {"gadget": 7, "gear": 8, "star_power": 9, "hypercharge": 11},
    "impact_priors": PRIORS,
    "scoring": SCORING,
    "hypercharge_availability": {"mode": "all_except", "list": []},
}
W_PP = 0.5

GEARS = [("Shield", "shield", 0.60, {}), ("Speed", "speed", 0.47, {}), ("Health", "health", 0.50, {})]


def mf(s):
    return math.exp((s - 0.5) / 0.125)


def climb(cur, target):
    return {"coins": CUM_COINS[target] - CUM_COINS[cur], "power_points": CUM_PP[target] - CUM_PP[cur]}


def equiv(cost):
    return cost.get("coins", 0) + W_PP * cost.get("power_points", 0) + cost.get("credits", 0)


@contextmanager
def _synthetic(gears=GEARS, catalog=None):
    saved = (P.R, P._gear_names, CATALOG[:])
    P.R = _Ref()
    P._gear_names = lambda economy: list(gears)
    if catalog is not None:
        CATALOG[:] = catalog
    try:
        yield
    finally:
        P.R, P._gear_names = saved[0], saved[1]
        CATALOG[:] = saved[2]


def _run(owned, rates, itemstats=None, econ=ECON, top=50, bracket="Mythic", power_floor=None,
         min_per_kind=0, boosted=(), games=None, catalog=None):
    with _synthetic(catalog=catalog):
        recs = P.recommend_purchases(owned, _Stats(rates, games), itemstats=itemstats, top=top,
                                     ranked_maps=RANKED_MAPS, economy=econ, rank_bracket=bracket,
                                     power_floor=power_floor, boosted=boosted,
                                     min_per_kind=min_per_kind)
    return recs


def _by(recs, brawler_id=None, kind=None):
    return [r for r in recs
            if (brawler_id is None or r["brawler_id"] == brawler_id)
            and (kind is None or r["kind"] == kind)]


def _maxed(i, hc=True, gears=("shield", "speed", "health")):
    return dict(gadgets=frozenset({i * 100 + 1, i * 100 + 2}),
                star_powers=frozenset({i * 100 + 3, i * 100 + 4}),
                gears=frozenset(gears), has_hypercharge=hc)


def _deep_catalog(n, start=100):
    """n fully-specified filler brawlers (ids start..start+n-1) to build a deep roster with."""
    return [_Brawler(i, f"Deep{i}", "Tank", "Epic", gadgets=_two(i, "g"), star_powers=_two(i, "s"))
            for i in range(start, start + n)]


# --- the bug report: a star power on an unbuilt, unfieldable brawler ---------------------------

def test_sub_floor_brawler_gets_only_the_ranked_ready_package():
    """The report: 'buy Damien's star power' topped the list while Damien sat far below Power 11
    with no items. Below the bracket's floor a brawler gets exactly ONE rec — the climb to the
    floor plus a first gadget and star power — and no item recs at all."""
    owned = {1: OwnedState(power=3)}                             # Mythic+ ⇒ Power 11 floor
    recs = _by(_run(owned, {1: 0.58}, bracket="Mythic"), brawler_id=1)
    assert len(recs) == 1 and recs[0]["kind"] == "power_upgrade", recs
    pkg = recs[0]
    assert [s["label"] for s in pkg["steps"]] == ["Power 3→11", "First gadget", "First star power"], pkg["steps"]
    c = climb(3, 11)
    assert pkg["cost"] == {"coins": c["coins"] + 3000, "power_points": c["power_points"]}, pkg["cost"]
    assert pkg["target_power"] == 11 and pkg["gate"] == "requires Power 11"
    assert "can't be fielded" in pkg["rationale"]
    # additive package value: access + first gadget + first star power, × meta factor
    assert math.isclose(pkg["value_lift"], (3.0 + 3.0 + 3.5) * mf(0.58), rel_tol=1e-3), pkg


def test_unbuilt_project_ranks_far_below_a_cheap_item_on_a_fielded_brawler():
    """A 1,000-coin first gadget on a maxed, equally strong brawler must beat the whole Power-3
    project — by a wide margin — because the ranking is value per coin, prerequisites included."""
    owned = {1: OwnedState(power=3), 2: OwnedState(power=11, star_powers=frozenset({S2A, S2B}))}
    recs = _run(owned, {1: 0.58, 2: 0.58})
    gadget = _by(recs, brawler_id=2, kind="gadget")[0]
    project = _by(recs, brawler_id=1, kind="power_upgrade")[0]
    assert recs[0] == gadget, recs[0]
    assert gadget["value_score"] > 2 * project["value_score"], (gadget, project)


def test_value_score_is_lift_per_thousand_coin_equivalents():
    owned = {2: OwnedState(power=11)}
    for r in _run(owned, {2: 0.55}):
        ce = r["cost_equiv"]
        assert ce == equiv(r["cost"]), (r["cost"], ce)
        assert math.isclose(r["value_score"], r["value_lift"] / ce * 1000, rel_tol=1e-3), r


def test_same_prior_cheaper_item_wins_per_coin():
    """gadget_first 3.0 @1,000 vs star_power_first 3.5 @2,000 ⇒ the gadget scores 1.71× per coin."""
    owned = {2: OwnedState(power=11)}
    recs = _run(owned, {2: 0.55})
    g = _by(recs, brawler_id=2, kind="gadget")[0]
    sp = _by(recs, brawler_id=2, kind="star_power")[0]
    assert math.isclose(g["value_score"] / sp["value_score"], (3.0 / 1000) / (3.5 / 2000), rel_tol=1e-3)


def test_meta_factor_is_exponential_in_win_rate():
    """exp((s − .5)/.125): a 55% brawler's first gadget is worth 1.49× a 50% one's, 45% ⇒ 0.67×."""
    owned = {1: OwnedState(power=11), 2: OwnedState(power=11), 3: OwnedState(power=11)}
    recs = _run(owned, {1: 0.55, 2: 0.50, 3: 0.45})
    a, b, c = (_by(recs, brawler_id=i, kind="gadget")[0]["value_lift"] for i in (1, 2, 3))
    assert math.isclose(a / b, math.exp(0.4), rel_tol=1e-3) and math.isclose(c / b, math.exp(-0.4), rel_tol=1e-3)


# --- the Ranked power floor --------------------------------------------------------------------

def test_floor_follows_the_bracket_and_unknown_assumes_11():
    """Power 9 is fieldable through Diamond (item recs + a climb to 11) but not from Mythic up
    (package only). Unknown bracket ⇒ the stricter Power-11 floor; an explicit floor pins it."""
    owned = {1: OwnedState(power=9, gadgets=frozenset({G1A}), star_powers=frozenset({S1A}))}
    diamond = _by(_run(owned, {1: 0.55}, bracket="Diamond"), brawler_id=1)
    assert {r["kind"] for r in diamond} == {"gadget", "star_power", "gear", "power_upgrade"}, diamond
    for bracket in ("Legendary", None, "Nonsense"):
        recs = _by(_run(owned, {1: 0.55}, bracket=bracket), brawler_id=1)
        assert len(recs) == 1 and recs[0]["kind"] == "power_upgrade" and recs[0]["gate"] == "requires Power 11", (bracket, recs)
        assert recs[0]["steps"][0]["label"] == "Power 9→11" and len(recs[0]["steps"]) == 1   # build owned
    pinned = _by(_run(owned, {1: 0.55}, bracket=None, power_floor=9), brawler_id=1)
    assert {r["kind"] for r in pinned} == {"gadget", "star_power", "gear", "power_upgrade"}
    assert P.resolve_floor("Gold", None, ECON) == 9 and P.resolve_floor("Masters", None, ECON) == 11
    assert P.resolve_floor(None, None, ECON) == 11 and P.resolve_floor("Pro", 9, ECON) == 9
    assert P.resolve_floor(None, 10, ECON) == 11, "an invalid explicit floor is ignored"


def test_climb_to_11_under_a_power9_floor_is_stats_plus_readiness():
    """A fieldable P9 brawler gets one power rec (no hypercharge yet): real stats for the two
    levels plus half its ranked-ready value (Mythic will need Power 11), no gate."""
    owned = {1: OwnedState(power=9, gadgets=frozenset({G1A, G1B}), star_powers=frozenset({S1A}),
                           gears=frozenset({"shield"}))}
    recs = _by(_run(owned, {1: 0.55}, bracket="Diamond"), brawler_id=1)
    assert not _by(recs, kind="hypercharge"), "HC is the next purchase only once at Power 11"
    pw = _by(recs, kind="power_upgrade")
    assert len(pw) == 1 and pw[0]["cost"] == climb(9, 11) and pw[0]["gate"] is None and pw[0]["target_power"] == 11
    owned_lift = 3.0 + 0.6 + 3.5 + 0.7            # gadget×2, first SP, one core gear
    expect = (1.5 * 2 + 0.5 * (3.0 + owned_lift)) * mf(0.55)
    assert math.isclose(pw[0]["value_lift"], expect, rel_tol=1e-3), (pw[0]["value_lift"], expect)
    assert "+11% HP and damage" in pw[0]["rationale"] and "second gear slot" in pw[0]["rationale"]


def test_built_main_climb_beats_fringe_first_gadget_below_mythic():
    """Diamond: maxing a built 54% main (your best brawler) from P9 outranks a first gadget on a
    47% brawler that sits behind twenty-one stronger fieldable picks — stats + Mythic readiness on
    the main, depth-discounted flexibility on the fringe pick, the way a top player would order it."""
    deep = _deep_catalog(20)
    owned = {1: OwnedState(power=9, **_maxed(1, hc=False)), 2: OwnedState(power=11)}
    owned.update({b.id: OwnedState(power=11, **_maxed(b.id)) for b in deep})
    rates = {1: 0.54, 2: 0.47, **{b.id: 0.53 for b in deep}}
    recs = _run(owned, rates, bracket="Diamond", catalog=CATALOG + deep)
    main = _by(recs, brawler_id=1, kind="power_upgrade")[0]
    fringe = _by(recs, brawler_id=2, kind="gadget")[0]
    assert main["value_score"] > 1.5 * fringe["value_score"], (main, fringe)
    assert "field 21 stronger" in fringe["rationale"] or fringe["value_lift"] < 3.0 * mf(0.47) * 0.6


def test_power_zero_means_unknown_and_is_treated_as_maxed():
    owned = {1: OwnedState(power=0, gadgets=frozenset({G1A}), star_powers=frozenset({S1A, S1B}))}
    recs = _by(_run(owned, {1: 0.55}), brawler_id=1)
    assert not _by(recs, kind="power_upgrade")
    assert _by(recs, kind="hypercharge"), "unknown power ⇒ assumed 11 ⇒ the HC is buyable"
    g = _by(recs, kind="gadget")[0]
    assert g["cost"] == {"coins": 1000} and g["gate"] is None


# --- depth: how many stronger brawlers you already field ---------------------------------------

def test_depth_discounts_past_the_free_tier_and_applies_to_items_too():
    """8 stronger fieldable brawlers are free; with 20 the factor is 1/(1+12/12) = 0.5 — on the
    unfieldable brawler's package AND on a fielded brawler's item recs alike."""
    deep = _deep_catalog(20)
    shallow = {1: OwnedState(power=3), 2: OwnedState(power=11, gadgets=frozenset({G2A}))}
    rates = {1: 0.55, 2: 0.55, **{b.id: 0.60 for b in deep}}
    a = _run(shallow, rates, catalog=CATALOG + deep)
    owned = {**shallow, **{b.id: OwnedState(power=11, **_maxed(b.id)) for b in deep}}
    b = _run(owned, rates, catalog=CATALOG + deep)
    pkg_a, pkg_b = (_by(r, brawler_id=1, kind="power_upgrade")[0] for r in (a, b))
    gad_a, gad_b = (_by(r, brawler_id=2, kind="gadget")[0] for r in (a, b))
    assert math.isclose(pkg_a["value_lift"] / pkg_b["value_lift"], 2.0, rel_tol=1e-3)
    assert math.isclose(gad_a["value_lift"] / gad_b["value_lift"], 2.0, rel_tol=1e-3)
    assert "field 20 stronger" in pkg_b["rationale"], pkg_b["rationale"]
    # eight stronger ⇒ still free
    eight = {**shallow, **{b_.id: OwnedState(power=11, **_maxed(b_.id)) for b_ in deep[:8]}}
    c = _run(eight, rates, catalog=CATALOG + deep)
    assert math.isclose(_by(c, brawler_id=1)[0]["value_lift"], pkg_a["value_lift"], rel_tol=1e-3)


def test_depth_counts_build_weight_and_sub_floor_brawlers_not_at_all():
    """A bare P11 counts half an option; a stronger brawler you can't field doesn't count."""
    deep = _deep_catalog(20)
    base = {1: OwnedState(power=3)}
    rates = {1: 0.55, **{b.id: 0.60 for b in deep}}
    bare = {**base, **{b.id: OwnedState(power=11) for b in deep}}              # 20 × 0.5 = 10 ⇒ n−free = 2
    unfieldable = {**base, **{b.id: OwnedState(power=9, **_maxed(b.id)) for b in deep}}
    ref = _by(_run(base, rates, catalog=CATALOG + deep), brawler_id=1)[0]["value_lift"]
    v_bare = _by(_run(bare, rates, catalog=CATALOG + deep, top=500), brawler_id=1)[0]["value_lift"]
    v_unf = _by(_run(unfieldable, rates, catalog=CATALOG + deep, top=500), brawler_id=1)[0]["value_lift"]
    assert math.isclose(ref / v_bare, 1 + 2 / 12, rel_tol=1e-3), (ref, v_bare)
    assert math.isclose(v_unf, ref, rel_tol=1e-3)


def test_depth_uses_the_brawlers_best_ranked_mode():
    """A mode specialist — 20th overall but the best Heist pick — keeps full value: depth is the
    smallest count over the overall pool and each ranked mode with real data."""
    deep = _deep_catalog(20)
    owned = {1: OwnedState(power=11, gadgets=frozenset({G1A})), **{b.id: OwnedState(power=11, **_maxed(b.id)) for b in deep}}
    rates = {1: 0.52, **{b.id: 0.65 for b in deep}}

    class _ModeStats(_Stats):
        def brawler_rate(self, brawler_id, map_id=None):
            if map_id == 12 and brawler_id == 1:          # the Heist map: brawler 1 shines
                return _Rate(0.70, 600.0)                  # enough games to survive the shrinkage
            return super().brawler_rate(brawler_id, map_id)

    with _synthetic(catalog=CATALOG + deep):
        flat = P.recommend_purchases(owned, _Stats(rates), ranked_maps=RANKED_MAPS, economy=ECON, rank_bracket="Mythic", boosted=(), top=500)
        heist = P.recommend_purchases(owned, _ModeStats(rates), ranked_maps=RANKED_MAPS, economy=ECON, rank_bracket="Mythic", boosted=(), top=500)
    sp_flat = _by(flat, brawler_id=1, kind="star_power")[0]
    sp_heist = _by(heist, brawler_id=1, kind="star_power")[0]
    # the overall (games-weighted) strength drives the meta factor; in the overall view all 20 are
    # stronger (depth 0.5), in the Heist view none is (depth 1) — the mode view is what lifts it
    assert sp_flat["value_lift"] < sp_heist["value_lift"]
    overall = (0.52 * 2 * 200 + 0.70 * 600) / (400 + 600)
    assert overall < 0.65, "the deep brawlers must be stronger overall for the mode view to matter"
    assert math.isclose(sp_heist["value_lift"], 3.5 * mf(overall), rel_tol=1e-3), sp_heist
    assert math.isclose(sp_flat["value_lift"], 3.5 * mf(0.52) * 0.5, rel_tol=1e-3), sp_flat
    # a thin mode blip doesn't: 120 Heist games shrink back toward the overall rate (k = 300)
    class _Blip(_Stats):
        def brawler_rate(self, brawler_id, map_id=None):
            if map_id == 12 and brawler_id == 1:
                return _Rate(0.70, 120.0)
            return super().brawler_rate(brawler_id, map_id)
    with _synthetic(catalog=CATALOG + deep):
        blip = P.recommend_purchases(owned, _Blip(rates), ranked_maps=RANKED_MAPS, economy=ECON, rank_bracket="Mythic", boosted=(), top=500)
    sp_blip = _by(blip, brawler_id=1, kind="star_power")[0]
    assert sp_blip["value_lift"] < sp_heist["value_lift"] * 0.7, (sp_blip, sp_heist)


# --- item recs on fieldable brawlers ------------------------------------------------------------

def test_recommends_the_missing_item_not_the_owned_one():
    """Owning gadget A of {A,B} ⇒ recommend B; owning both star powers ⇒ no star-power rec."""
    owned = {1: OwnedState(power=11, gadgets=frozenset({G1A}), star_powers=frozenset({S1A, S1B}))}
    recs = _run(owned, {1: 0.55})
    gadget = _by(recs, brawler_id=1, kind="gadget")
    assert len(gadget) == 1 and gadget[0]["item_id"] == G1B, gadget
    assert not _by(recs, brawler_id=1, kind="star_power"), "owns both star powers → nothing to buy"


def test_first_item_outranks_second_at_equal_strength():
    owned = {1: OwnedState(power=11, gadgets=frozenset({G1A})),      # owns one → second-gadget buy
             2: OwnedState(power=11)}                                 # owns none → first-gadget buy
    recs = _run(owned, {1: 0.55, 2: 0.55})
    first = _by(recs, brawler_id=2, kind="gadget")[0]
    second = _by(recs, brawler_id=1, kind="gadget")[0]
    assert first["value_score"] > second["value_score"], (first, second)


def test_gear_slots_follow_power_and_a_spare_gear_needs_a_measured_reason():
    """Gear slots open at Power 8 and 10. A gear for an empty slot is a rec; a *spare* gear (both
    slots filled) is only a rec when the data says you'd actually swap to it (significant positive
    delta) — unmeasured spares are filler, not advice."""
    none = {1: OwnedState(power=11, **dict(_maxed(1), gears=frozenset()))}
    two = {1: OwnedState(power=11, **dict(_maxed(1), gears=frozenset({"shield", "speed"})))}
    one_p9 = {1: OwnedState(power=9, **dict(_maxed(1, hc=False), gears=frozenset({"shield"})))}
    core = _by(_run(none, {1: 0.55}), brawler_id=1, kind="gear")[0]
    assert not _by(_run(two, {1: 0.55}), brawler_id=1, kind="gear"), "unmeasured spare gear ⇒ no rec"
    its = {"meta": {"gear_ids_by_name": {"health": 901}}, "cells": {"1:901": {"significant": True, "delta": 0.04}}}
    extra = _by(_run(two, {1: 0.55}, itemstats=its), brawler_id=1, kind="gear")[0]
    assert extra["item_name"] == "Health" and extra["confidence"] == "measured", extra
    assert core["value_lift"] > 2 * extra["value_lift"], (core, extra)      # 0.7 vs 0.25×1.4
    p9 = _by(_run(one_p9, {1: 0.55}, bracket="Gold"), brawler_id=1, kind="gear")
    assert not p9, "P9 has one slot, already filled ⇒ a second gear is a spare ⇒ no unmeasured rec"
    p10 = {1: OwnedState(power=10, **dict(_maxed(1, hc=False), gears=frozenset({"shield"})))}
    g10 = _by(_run(p10, {1: 0.55}, bracket="Gold"), brawler_id=1, kind="gear")[0]
    assert "fills an empty gear slot" in g10["rationale"] and math.isclose(g10["value_lift"], 0.7 * mf(0.55), rel_tol=1e-3)


def test_measured_negative_delta_never_beats_an_unmeasured_item():
    """A significant NEGATIVE delta says that item is the worse buy: it must not win selection over
    an unmeasured alternative — for gadgets/star powers and for gears alike."""
    first = {1: OwnedState(power=11)}
    its = {"cells": {f"1:{G1A}": {"significant": True, "delta": -0.04}}}
    g = _by(_run(first, {1: 0.55}, itemstats=its), brawler_id=1, kind="gadget")[0]
    assert g["item_id"] == G1B and g["item_delta"] is None, g          # the unmeasured one wins
    gears = {1: OwnedState(power=11, **dict(_maxed(1), gears=frozenset()))}
    its_g = {"meta": {"gear_ids_by_name": {"shield": 902}}, "cells": {"1:902": {"significant": True, "delta": -0.03}}}
    gr = _by(_run(gears, {1: 0.55}, itemstats=its_g), brawler_id=1, kind="gear")[0]
    assert gr["item_name"] != "Shield" and gr["item_delta"] is None, gr
    # when every candidate is measured, the best (least bad) one still wins
    its_both = {"cells": {f"1:{G1A}": {"significant": True, "delta": -0.04},
                          f"1:{G1B}": {"significant": True, "delta": -0.01}}}
    g2 = _by(_run(first, {1: 0.55}, itemstats=its_both), brawler_id=1, kind="gadget")[0]
    assert g2["item_id"] == G1B and g2["item_delta"] == -0.01, g2


def test_hypercharge_is_a_straight_buy_only_at_power_11():
    owned = {1: OwnedState(power=11, **_maxed(1, hc=False)),
             2: OwnedState(power=10, **_maxed(2, hc=False))}
    recs = _run(owned, {1: 0.55, 2: 0.55}, bracket="Gold")
    hc = _by(recs, brawler_id=1, kind="hypercharge")
    assert len(hc) == 1 and hc[0]["cost"] == {"coins": 5000} and hc[0]["gate"] is None, hc
    assert math.isclose(hc[0]["value_lift"], 6.0 * mf(0.55), rel_tol=1e-3)
    assert not _by(recs, brawler_id=1, kind="power_upgrade")
    assert not _by(recs, brawler_id=2, kind="hypercharge") and len(_by(recs, brawler_id=2, kind="power_upgrade")) == 1
    assert "Hypercharge slot" in _by(recs, brawler_id=2, kind="power_upgrade")[0]["rationale"]


def test_hypercharge_policy_and_missing_section():
    owned = {1: OwnedState(power=11, **_maxed(1, hc=False))}
    only = dict(ECON, hypercharge_availability={"mode": "only", "list": []})
    assert not _by(_run(owned, {1: 0.55}, econ=only), kind="hypercharge")
    missing = {k: v for k, v in ECON.items() if k != "hypercharge_availability"}
    assert not _by(_run(owned, {1: 0.55}, econ=missing), kind="hypercharge")
    assert _by(_run(owned, {1: 0.55}), kind="hypercharge")


def test_no_buffie_recommendations():
    """Buffies are unadvised: the roster reports which buffies you own but not how many exist, so
    'slot open' can't be told from 'no buffie released' (R-T has none)."""
    recs = _run({1: OwnedState(power=11, **_maxed(1))}, {1: 0.55})
    assert not _by(recs, brawler_id=1, kind="buffie")


# --- new-brawler unlocks ---------------------------------------------------------------------

def test_unowned_brawler_is_a_priced_unlock_package_on_the_current_starr_road_tier():
    """An unowned brawler is a Starr Road unlock + the climb to the floor + a core build, priced
    by rarity — never the bare credit price, since a Power-1 brawler is unfieldable — and only
    for the lowest purchasable tier that still has an unowned brawler."""
    owned = {1: OwnedState(power=11, **_maxed(1)), 2: OwnedState(power=11, **_maxed(2))}
    recs = _run(owned, {1: 0.55, 2: 0.55, 3: 0.60}, bracket="Diamond")
    nb = _by(recs, brawler_id=3, kind="new_brawler")
    assert len(nb) == 1 and len(_by(recs, brawler_id=3)) == 1, recs
    pkg = nb[0]
    assert [s["label"] for s in pkg["steps"]] == ["Unlock (Mythic)", "Power 1→9", "First gadget", "First star power"]
    c = climb(1, 9)
    assert pkg["cost"] == {"credits": 1900, "coins": c["coins"] + 3000, "power_points": c["power_points"]}, pkg["cost"]
    assert pkg["target_power"] == 9 and not pkg["cost_estimated"]
    assert math.isclose(pkg["value_lift"], (3.0 + 3.0 + 3.5) * mf(0.60), rel_tol=1e-3)
    # an unowned Epic (brawler 2) puts the road on the Epic tier ⇒ the Mythic isn't offered yet
    recs2 = _run({1: OwnedState(power=11, **_maxed(1))}, {1: 0.55, 2: 0.55, 3: 0.60})
    assert _by(recs2, brawler_id=2, kind="new_brawler") and not _by(recs2, brawler_id=3)
    assert "Starr Road" in _by(recs2, brawler_id=2)[0]["rationale"]


def test_trophy_road_rarities_and_zero_data_brawlers_are_not_unlock_recs():
    rare = [_Brawler(9, "RareGuy", "Tank", "Rare", gadgets=_two(9, "g"), star_powers=_two(9, "s"))]
    owned = {1: OwnedState(power=11, **_maxed(1)), 2: OwnedState(power=11, **_maxed(2))}
    recs = _run(owned, {1: 0.55, 2: 0.55, 3: 0.60, 9: 0.60}, catalog=CATALOG + rare)
    assert not _by(recs, brawler_id=9), "Rare brawlers are Trophy Road rewards, not purchasable"
    assert _by(recs, brawler_id=3, kind="new_brawler")
    assert not _by(_run(owned, {1: 0.55, 2: 0.55, 3: 0.60}, games={3: 5.0}), brawler_id=3), \
        "a brawler the data has never seen (unreleased / retired) is not vouched for"


def test_existence_gate_uses_the_global_table_not_the_thin_bracket():
    """A bracket table's per-brawler games only count that bracket's matches; "has the data ever
    seen this brawler?" must be asked of the global table behind it, or thin brackets lose their
    unlock recs (Masters: Pam at 17 games) while the global sample is rich."""
    class _Bracket(_Stats):
        def __init__(self, rates, games, fallback):
            super().__init__(rates, games)
            self.fallback = fallback
    owned = {1: OwnedState(power=11, **_maxed(1)), 2: OwnedState(power=11, **_maxed(2))}
    rates = {1: 0.55, 2: 0.55, 3: 0.60}
    thin = _Bracket(rates, {3: 12.0}, fallback=_Stats(rates))        # 12 games here, 200 globally
    with _synthetic():
        recs = P.recommend_purchases(owned, thin, ranked_maps=RANKED_MAPS, economy=ECON,
                                     rank_bracket="Masters", boosted=(), top=50)
    assert _by(recs, brawler_id=3, kind="new_brawler"), recs
    # ...but a brawler the GLOBAL table never saw is still not vouched for
    never = _Bracket(rates, {3: 200.0}, fallback=_Stats(rates, {3: 0.0}))
    with _synthetic():
        recs2 = P.recommend_purchases(owned, never, ranked_maps=RANKED_MAPS, economy=ECON,
                                      rank_bracket="Masters", boosted=(), top=50)
    assert not _by(recs2, brawler_id=3)


def test_zero_data_catalog_entry_does_not_hold_the_starr_road_tier_open():
    """Owning every obtainable Legendary but not a retired collab (0 games, same rarity) must not
    pin the road on 'Legendary' forever — the Ultra Legendary tier's unlocks are offered."""
    extra = [_Brawler(7, "Retired", "Tank", "Legendary", gadgets=_two(7, "g"), star_powers=_two(7, "s")),
             _Brawler(8, "Ultra", "Tank", "Ultra Legendary", gadgets=_two(8, "g"), star_powers=_two(8, "s"))]
    owned = {1: OwnedState(power=11, **_maxed(1)), 2: OwnedState(power=11, **_maxed(2)), 3: OwnedState(power=11, **_maxed(3))}
    rates = {1: 0.55, 2: 0.55, 3: 0.55, 7: 0.55, 8: 0.56}
    recs = _run(owned, rates, games={7: 0.0}, catalog=CATALOG + extra)
    assert not _by(recs, brawler_id=7) and _by(recs, brawler_id=8, kind="new_brawler"), recs


def test_depth_note_wording():
    """A lone bare P11 (build weight 0.5) never prints 'you already field 0 stronger brawlers';
    a specialist mode view names the mode and the brawler's rate there."""
    owned = {1: OwnedState(power=3), 2: OwnedState(power=11)}                 # 2 is bare and stronger
    note = _by(_run(owned, {1: 0.55, 2: 0.60}), brawler_id=1)[0]["rationale"]
    assert "nothing you field is stronger" in note and "0 stronger" not in note, note
    deep = _deep_catalog(20)
    owned = {1: OwnedState(power=3), **{b.id: OwnedState(power=11, **_maxed(b.id)) for b in deep}}
    rates = {1: 0.52, **{b.id: 0.65 for b in deep}}

    class _ModeStats(_Stats):
        def brawler_rate(self, brawler_id, map_id=None):
            if map_id == 12 and brawler_id == 1:
                return _Rate(0.70, 600.0)
            return super().brawler_rate(brawler_id, map_id)
    with _synthetic(catalog=CATALOG + deep):
        recs = P.recommend_purchases(owned, _ModeStats(rates), ranked_maps=RANKED_MAPS, economy=ECON,
                                     rank_bracket="Mythic", boosted=(), top=500)
    why = _by(recs, brawler_id=1)[0]["rationale"]
    assert "nothing you field is stronger in Heist (" in why and "% there)" in why, why


def test_boosted_sub_floor_package_leads_with_the_boost():
    owned = {1: OwnedState(power=9, **_maxed(1, hc=False))}
    why = _by(_run(owned, {1: 0.55}, boosted=(1,)), brawler_id=1)[0]["rationale"]
    assert why.startswith("Strong is free at Power 11") and "can't be fielded" not in why, why


def test_unknown_rarity_is_priced_at_the_dearest_known_tier_and_flagged():
    """A rarity the file prices as null (unknown) gets the dearest known price as a nominal and
    is flagged — it must never rank as a free unlock; a key merely absent from the file takes the
    code default (Ultra Legendary 5,500)."""
    odd = [_Brawler(8, "Oddity", "Tank", "Ultra Legendary", gadgets=_two(8, "g"), star_powers=_two(8, "s"))]
    owned = {1: OwnedState(power=11, **_maxed(1)), 2: OwnedState(power=11, **_maxed(2)), 3: OwnedState(power=11, **_maxed(3))}
    rates = {1: 0.55, 2: 0.55, 3: 0.55, 8: 0.55}
    unknown = dict(ECON, new_brawler_credits={"Epic": 925, "Mythic": 1900, "Legendary": 3800, "Ultra Legendary": None})
    pkg = _by(_run(owned, rates, econ=unknown, catalog=CATALOG + odd), brawler_id=8)[0]
    assert pkg["cost"]["credits"] == 3800 and pkg["cost_estimated"] is True, pkg
    absent = dict(ECON, new_brawler_credits={"Epic": 925, "Mythic": 1900, "Legendary": 3800})
    dflt = _by(_run(owned, rates, econ=absent, catalog=CATALOG + odd), brawler_id=8)[0]
    assert dflt["cost"]["credits"] == 5500 and dflt["cost_estimated"] is False, dflt
    full = _by(_run(owned, rates, catalog=CATALOG + odd), brawler_id=8)[0]
    assert full["cost"]["credits"] == 5500 and full["cost_estimated"] is False


def test_boosted_brawler_is_discounted_and_counts_as_fieldable():
    """Every rec on a season-free brawler is halved and says so; an unowned boosted brawler still
    counts toward depth (it's fieldable at full build this season)."""
    owned = {1: OwnedState(power=3), 2: OwnedState(power=11, gadgets=frozenset({G2A}))}
    plain = _run(owned, {1: 0.55, 2: 0.55})
    boosted = _run(owned, {1: 0.55, 2: 0.55}, boosted=(1, 2))
    for kind, bid in (("power_upgrade", 1), ("gadget", 2)):
        a = _by(plain, brawler_id=bid, kind=kind)[0]
        b = _by(boosted, brawler_id=bid, kind=kind)[0]
        assert math.isclose(b["value_lift"], 0.5 * a["value_lift"], rel_tol=1e-3), (a, b)
        assert "this season" in b["rationale"] and "this season" not in a["rationale"]
    deep = _deep_catalog(20)
    rates = {1: 0.55, **{b.id: 0.60 for b in deep}}
    ref = _by(_run({1: OwnedState(power=3)}, rates, catalog=CATALOG + deep), brawler_id=1)[0]["value_lift"]
    via_boost = _by(_run({1: OwnedState(power=3)}, rates, catalog=CATALOG + deep, boosted=tuple(b.id for b in deep)), brawler_id=1)[0]["value_lift"]
    assert math.isclose(ref / via_boost, 2.0, rel_tol=1e-3), (ref, via_boost)


def test_engine_wrapper_unions_data_derived_free_brawlers():
    """The engine's ``recommend_purchases`` wrapper must feed the advisor the SAME free/"boosted"
    union every other surface uses: the hand-maintained ``load_ranked_boosted`` list ∪ the
    data-derived ``stats.free_brawler_ids`` carried in the stats artifact. Left to its default the
    advisor sees only the hand list, so an unannounced mid-season free grant (the Nori case) gets no
    boosted discount and is over-recommended as a purchase — a brawler you can already field free."""
    import types
    from bsdraft.engine import engine as E

    captured = {}
    saved = (E.purchases_mod.recommend_purchases, E.R.load_ranked_boosted,
             E.itemstats_mod.get_itemstats)
    E.purchases_mod.recommend_purchases = lambda owned, stats, **kw: captured.update(kw) or []
    E.R.load_ranked_boosted = lambda: (111,)                        # hand-maintained list
    E.itemstats_mod.get_itemstats = lambda: None
    try:
        eng = types.SimpleNamespace(
            stats=types.SimpleNamespace(free_brawler_ids=frozenset({222})),  # data-derived, unannounced
            bracket_stats={})
        E.DraftEngine.recommend_purchases(eng, owned={}, rank_bracket="Mythic")
        assert captured["boosted"] == frozenset({111, 222})
    finally:
        (E.purchases_mod.recommend_purchases, E.R.load_ranked_boosted,
         E.itemstats_mod.get_itemstats) = saved


# --- measurement, degradation, list mechanics -----------------------------------------------------

def test_measured_delta_picks_a_first_item_but_scales_only_a_second():
    """A significant delta is item-vs-the-other-item: on a first item it decides WHICH to buy
    (confidence 'measured', lift unchanged); on a second item it scales the value — signed and
    capped at halving/doubling (±10pp ⇒ ×2 / ×0.5)."""
    first = {1: OwnedState(power=11)}
    its = {"cells": {f"1:{G1B}": {"significant": True, "delta": 0.06}}}
    m = _by(_run(first, {1: 0.55}, itemstats=its), brawler_id=1, kind="gadget")[0]
    p = _by(_run(first, {1: 0.55}), brawler_id=1, kind="gadget")[0]
    assert m["item_id"] == G1B and m["confidence"] == "measured" and m["item_delta"] == 0.06
    assert p["item_id"] == G1A and p["confidence"] == "heuristic" and p["item_delta"] is None
    assert math.isclose(m["value_lift"], p["value_lift"], rel_tol=1e-3), "a first item's delta doesn't scale"

    second = {1: OwnedState(power=11, gadgets=frozenset({G1A}))}          # missing G1B
    base = _by(_run(second, {1: 0.55}), brawler_id=1, kind="gadget")[0]["value_lift"]
    for delta, factor in ((0.05, 1.5), (0.30, 2.0), (-0.03, 0.7), (-0.30, 0.5)):
        its = {"cells": {f"1:{G1B}": {"significant": True, "delta": delta}}}
        r = _by(_run(second, {1: 0.55}, itemstats=its), brawler_id=1, kind="gadget")[0]
        assert math.isclose(r["value_lift"], factor * base, rel_tol=1e-3), (delta, r["value_lift"], base)
        assert r["confidence"] == "measured"


def test_degrades_without_itemstats():
    owned = {2: OwnedState(power=11)}
    recs = _run(owned, {2: 0.55}, itemstats=None)
    assert recs, "should still produce recommendations from priors alone"
    assert all(r["confidence"] != "measured" for r in recs if r["kind"] in ("gadget", "star_power", "gear"))


def test_partial_or_missing_economy_falls_back_to_defaults_per_section():
    """Cost is the ranking denominator now, so a missing section must NOT make a package free:
    dropping any one section (or the whole file) reproduces the fully-priced ranking, and a
    legacy 0..1-unit impact_priors table is ignored rather than mixed into lift units."""
    owned = {1: OwnedState(power=11, gadgets=frozenset({G1A})), 2: OwnedState(power=3)}
    rates = {1: 0.55, 2: 0.55, 3: 0.56}
    full = [(r["brawler_id"], r["kind"], r["cost"]) for r in _run(owned, rates)]
    for drop in ("power_cost_cumulative", "item_costs", "new_brawler_credits", "impact_priors", "scoring"):
        econ = {k: v for k, v in ECON.items() if k != drop}
        got = [(r["brawler_id"], r["kind"], r["cost"]) for r in _run(owned, rates, econ=econ)]
        assert got == full, (drop, got, full)
    legacy = dict(ECON, impact_priors={"gadget_first": .85, "gadget_second": .45, "star_power_first": .90,
                                       "star_power_second": .40, "gear": .55, "hypercharge": .80, "new_brawler": .70})
    assert [(r["brawler_id"], r["kind"], r["value_score"]) for r in _run(owned, rates, econ=legacy)] == \
           [(r["brawler_id"], r["kind"], r["value_score"]) for r in _run(owned, rates)]
    # the whole file gone: every price falls back to the defaults; only the hypercharge rec drops
    # out, because the availability *policy* is deliberately fail-safe to "don't vouch for one"
    bare = _run(owned, rates, econ={})
    assert [(r["brawler_id"], r["kind"], r["cost"]) for r in bare] == [x for x in full if x[1] != "hypercharge"]
    assert all(math.isfinite(r["value_score"]) and r["value_score"] > 0 for r in bare)


def test_meta_strength_averages_across_ranked_maps():
    owned = {3: OwnedState(power=11, gadgets=frozenset({301}), star_powers=frozenset({303}))}
    recs = _by(_run(owned, {3: 0.573}), brawler_id=3)
    assert recs and all(abs(r["meta_winrate"] - 0.573) < 1e-9 for r in recs), recs


def test_results_are_sorted_by_value_score_desc_with_a_stable_tiebreak():
    owned = {1: OwnedState(power=11, gadgets=frozenset({G1A})), 2: OwnedState(power=11, gadgets=frozenset({G2A}))}
    recs = _run(owned, {1: 0.55, 2: 0.55})
    scores = [r["value_score"] for r in recs]
    assert scores == sorted(scores, reverse=True), scores
    ties = [r["brawler_name"] for r in recs if r["kind"] == "gadget"]
    assert ties == ["Mid", "Strong"], ties                    # identical recs ⇒ name order


def test_top_limit_is_respected():
    recs = _run({1: OwnedState(power=11)}, {1: 0.55}, top=2)
    assert len(recs) == 2


def test_min_per_kind_keeps_low_efficiency_kinds_discoverable():
    owned = {1: OwnedState(power=11, gadgets=frozenset({G1A}), star_powers=frozenset({S1A}))}
    rates = {1: 0.55, 2: 0.52, 3: 0.52}
    assert not _by(_run(owned, rates, top=2), kind="new_brawler")
    reserved = _run(owned, rates, top=2, min_per_kind=1)
    assert _by(reserved, kind="new_brawler") and len(reserved) >= 2
    scores = [r["value_score"] for r in reserved]
    assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
