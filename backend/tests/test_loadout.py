"""Tests for the loadout advice engine and the owned-item roster plumbing."""
from __future__ import annotations

import pytest

from bsdraft.data import reference as R
from bsdraft.engine import loadout as L
from bsdraft.engine import mastery


def _ids_by_class(cls_name: str, n: int) -> list:
    """First n brawler ids of a class from the live reference — refresh-proof, no hardcoded ids."""
    ids = [b.id for b in R.load_brawlers() if b.cls == cls_name]
    assert len(ids) >= n, f"reference needs at least {n} {cls_name}s"
    return ids[:n]


def test_classify_effect_buckets():
    assert L._classify("Shelly dashes to an aimed direction, fully reloading her ammo") == "mobility"
    assert L._classify("she instantly heals for x health") == "sustain"
    assert L._classify("Super shells slow down enemies") == "control"
    assert L._classify("deals +x extra damage at max range") == "damage"
    assert L._classify("increases her attack range by one tile") == "range"
    assert L._classify("reveals enemies hidden in a bush") == "vision"
    # No matched keyword falls back to utility rather than erroring.
    assert L._classify("some brand new mechanic with no keywords") == "utility"


def test_classify_catalog_text_is_word_aware_and_handles_projectile_defense():
    assert L._classify("Poco removes adverse effects and grants immunity") == "sustain"
    assert L._classify("Pam removes ammo from opposing Brawlers") == "reload"
    assert L._classify("El Primo flexes and destroys incoming projectiles for 2 sec") == "projectile_defense"
    assert L._classify("Sprout destroys its Hedge, but instantly has its Super fully charged") == "super_charge"
    assert L._classify("Darryl spins around, dealing damage and recharging his Super") == "damage"


def test_clean_strips_value_tokens():
    assert L._clean("slows for <!card.value1.ticks> sec") == "slows for sec"
    assert L._clean("  extra   spaces ") == "extra spaces"


def test_mode_fit_is_mode_sensitive():
    # Mobility matters more in Brawl Ball; sustain more in Knockout.
    assert L._mode_fit("mobility", "Brawl Ball") > L._mode_fit("mobility", "Heist")
    assert L._mode_fit("sustain", "Knockout") > L._mode_fit("sustain", "Heist")
    assert L._mode_fit("damage", "Heist") > L._mode_fit("damage", "Gem Grab")


def test_super_charge_is_classified_and_r_t_gadget_varies_by_mode():
    rt = R.brawler_by_name("R-T")
    assert rt is not None
    assert L._classify(rt.gadgets[0].description) == "super_charge"

    picks = {
        mode: next(g["name"] for g in L.loadout_advice(rt.id, mode)["gadgets"]
                   if g["recommended"])
        for mode in ("Gem Grab", "Brawl Ball", "Knockout", "Heist", "Hot Zone", "Bounty")
    }
    assert picks["Gem Grab"] == "Out Of Line"
    assert picks["Heist"] == "Hacks"
    assert len(set(picks.values())) == 2


def test_el_primo_gadgets_match_current_rework():
    primo = R.brawler_by_name("El Primo")
    assert primo is not None
    assert primo.gadgets[0].description == (
        "El Primo Dashes, grabbing any hit enemies and suplexing them behind him."
    )
    assert primo.gadgets[1].description.startswith(
        "El primo flexes his pecs and destroys any incoming projectiles"
    )
    assert L._classify(primo.gadgets[0].description) == "mobility"
    assert L._classify(primo.gadgets[1].description) == "projectile_defense"


def test_loadout_advice_shape_and_single_recommendation():
    b = R.brawler_by_name("Shelly")
    adv = L.loadout_advice(b.id, "Knockout")
    assert adv is not None
    assert adv["brawler_id"] == b.id and adv["brawler_name"] == "Shelly"
    assert adv["gadgets"] and adv["star_powers"] and adv["gears"]
    # Exactly one gadget and one star power flagged as the best-fit pick.
    assert sum(g["recommended"] for g in adv["gadgets"]) == 1
    assert sum(s["recommended"] for s in adv["star_powers"]) == 1
    for item in adv["gadgets"] + adv["star_powers"]:
        assert 0.0 <= item["fit"] <= 1.0
        assert item["effect"] and item["why"]
        assert item["source"] == "heuristic"


def test_loadout_recommendation_changes_with_mode():
    b = R.brawler_by_name("Shelly")
    ko = {g["name"]: g for g in L.loadout_advice(b.id, "Knockout")["star_powers"]}
    heist = {g["name"]: g for g in L.loadout_advice(b.id, "Heist")["star_powers"]}
    # Band-Aid (sustain) is the Knockout pick; it should not also be the Heist pick.
    assert ko["Band-Aid"]["recommended"] is True
    assert heist["Band-Aid"]["recommended"] is False


def test_gears_are_mode_and_role_ranked():
    # Only the six stable Super Rare gears, two flagged as the core pick.
    gears = L.loadout_advice(R.brawler_by_name("Shelly").id, "Knockout")["gears"]
    assert len(gears) == 6
    assert sum(g["recommended"] for g in gears) == 2
    # For a Damage Dealer, Shield tops a survival mode (Knockout) while the Damage gear tops a burst
    # mode (Heist) — the mode weighting flips the ranking.
    ko = {g["name"]: g["fit"] for g in L.loadout_advice(R.brawler_by_name("Shelly").id, "Knockout")["gears"]}
    heist = {g["name"]: g["fit"] for g in L.loadout_advice(R.brawler_by_name("Shelly").id, "Heist")["gears"]}
    assert ko["Shield"] > ko["Damage"]
    assert heist["Damage"] > heist["Shield"]


def test_loadout_advice_unknown_brawler_is_none():
    assert L.loadout_advice(999_999_999, "Bounty") is None


# ---- enemy-comp overlay ----------------------------------------------------------------------

def test_comp_identity_without_enemies():
    # The deploy-safety guarantee: no enemies (or only unknown ids) must be byte-identical to the
    # comp-blind call — the overlay is absent, not defaulted.
    b = R.brawler_by_name("Shelly")
    base = L.loadout_advice(b.id, "Knockout")
    for enemies in (None, [], [999_999_999]):
        assert L.loadout_advice(b.id, "Knockout", enemies=enemies) == base
    assert base["comp_reads"] == []
    assert all(it["comp_delta"] == 0.0 and it["comp_why"] == [] and it["comp_flipped"] is False
               for it in base["gadgets"] + base["star_powers"])


def test_single_enemy_fires_no_read():
    # All reads threshold at >=2, so one enemy pick changes nothing (graceful partial-draft start).
    adv = L.loadout_advice(R.brawler_by_name("Shelly").id, "Knockout", enemies=_ids_by_class("Tank", 1))
    assert adv["comp_reads"] == []
    assert all(it["comp_delta"] == 0.0 for it in adv["gadgets"] + adv["star_powers"])


def test_comp_reads_fire_and_fold_into_fit():
    tanks = _ids_by_class("Tank", 2)
    b = R.brawler_by_name("Shelly")
    blind = L.loadout_advice(b.id, "Gem Grab")
    comp = L.loadout_advice(b.id, "Gem Grab", enemies=tanks)
    assert comp["comp_reads"]                                  # tanky (and aggro) fired
    assert "enemy comp" in comp["note"]
    boosted = 0
    for a, c in zip(blind["gadgets"] + blind["star_powers"], comp["gadgets"] + comp["star_powers"]):
        # fit - comp_delta reconstructs the comp-blind fit exactly
        assert c["fit"] - c["comp_delta"] == pytest.approx(a["fit"], abs=1e-9)
        # chips only above the display threshold, and signed to match the delta
        if abs(c["comp_delta"]) >= L._COMP_CHIP_MIN:
            assert c["comp_why"]
        else:
            assert c["comp_why"] == []
        boosted += c["comp_delta"] != 0.0
    assert boosted > 0                                          # the overlay actually moved something


def test_comp_bonus_clamped(monkeypatch):
    # A synthetic oversized table value must clamp at ±_COMP_CLAMP (real table values sum below it).
    monkeypatch.setitem(L._COMP_EFFECT, "tanky", {"control": 0.5, "range": -0.5})
    overlay = L._comp_overlay(_ids_by_class("Tank", 2))
    assert overlay["bonus"]["control"] == L._COMP_CLAMP
    assert overlay["bonus"]["range"] == -L._COMP_CLAMP


def test_comp_preserves_measured_overlay(monkeypatch):
    # Comp folds into fit AFTER the measured overlay: source/why keep the measured claim verbatim,
    # and comp_delta carries the adjustment separately.
    shelly = R.brawler_by_name("Shelly")
    gid = shelly.gadgets[0].id
    table = {"version": 1, "meta": {"gear_ids_by_name": {}}, "brawler_baseline": {},
             "cells": {f"{shelly.id}:{gid}": {"item_type": "gadget", "delta": 0.06,
                                              "item_winrate": 0.55, "significant": 1, "n_eff": 240,
                                              "n_players": 61, "n_eff_rest": 300}}}
    monkeypatch.setattr(L.IS, "get_itemstats", lambda *a, **k: table)
    adv = L.loadout_advice(shelly.id, "Knockout", enemies=_ids_by_class("Tank", 2))
    measured = next(g for g in adv["gadgets"] if g["id"] == gid)
    assert measured["source"] == "winrate"
    assert "win rate" in measured["why"]                        # measured prose not rewritten
    assert measured["fit"] - measured["comp_delta"] == pytest.approx(0.5 + 3.0 * 0.06)


def test_comp_cannot_launder_measured_negative(monkeypatch):
    # A measured-WORSE item (base fit < 0.5) must not become the pick via a comp bump: the
    # measured-better test runs on the comp-blind base fit.
    shelly = R.brawler_by_name("Shelly")
    gid = shelly.gadgets[0].id
    table = {"version": 1, "meta": {"gear_ids_by_name": {}}, "brawler_baseline": {},
             "cells": {f"{shelly.id}:{gid}": {"item_type": "gadget", "delta": -0.05,
                                              "item_winrate": 0.46, "significant": 1, "n_eff": 240,
                                              "n_players": 61, "n_eff_rest": 300}}}
    monkeypatch.setattr(L.IS, "get_itemstats", lambda *a, **k: table)
    adv = L.loadout_advice(shelly.id, "Knockout", enemies=_ids_by_class("Tank", 2))
    measured = next(g for g in adv["gadgets"] if g["id"] == gid)
    other = next(g for g in adv["gadgets"] if g["id"] != gid)
    assert measured["recommended"] is False
    assert other["recommended"] is True


def test_mark_best_flags_comp_flip():
    # The winner differs from the comp-blind winner only via its comp bump -> comp_flipped.
    a = {"name": "A", "kind": "gadget", "fit": 0.60, "comp_delta": 0.0, "comp_flipped": False,
         "recommended": False, "source": "heuristic", "why": "Mobility: reposition."}
    b = {"name": "B", "kind": "gadget", "fit": 0.64, "comp_delta": 0.10, "comp_flipped": False,
         "recommended": False, "source": "heuristic", "why": "Control: lock down."}
    L._mark_best([a, b], "Knockout")
    assert b["recommended"] is True and b["comp_flipped"] is True   # base 0.54 loses, final 0.64 wins
    assert a["recommended"] is False


def test_mark_best_flip_flag_exact_on_tied_bases():
    # fit and comp_delta are each 3-dp rounded, so a raw `fit - comp_delta` reconstruction carries
    # ~1e-16 float noise (0.59 - 0.04 != 0.55) that used to mis-resolve TRUE base-fit ties in the
    # exact max comparisons. A won the comp-blind tie (first in list) AND wins comp-adjusted — the
    # comp changed nothing, so no flip flag.
    a = {"name": "A", "kind": "gadget", "fit": 0.59, "comp_delta": 0.04, "comp_flipped": False,
         "recommended": False, "source": "heuristic", "why": "Mobility: reposition."}
    b = {"name": "B", "kind": "gadget", "fit": 0.55, "comp_delta": 0.0, "comp_flipped": False,
         "recommended": False, "source": "heuristic", "why": "Reload: uptime."}
    L._mark_best([a, b], "Knockout")
    assert a["recommended"] is True and a["comp_flipped"] is False


def test_mark_best_flip_flag_fires_on_genuine_tie_break():
    # Comp-blind the tie goes to A (list order); the comp bump flips the pick to B — flagged, even
    # though B's raw base reconstruction (0.61 - 0.06) is float-noisy.
    a = {"name": "A", "kind": "gadget", "fit": 0.55, "comp_delta": 0.0, "comp_flipped": False,
         "recommended": False, "source": "heuristic", "why": "Reload: uptime."}
    b = {"name": "B", "kind": "gadget", "fit": 0.61, "comp_delta": 0.06, "comp_flipped": False,
         "recommended": False, "source": "heuristic", "why": "Sustain: survive."}
    L._mark_best([a, b], "Knockout")
    assert b["recommended"] is True and b["comp_flipped"] is True


def test_mark_best_rank_caps_measured_negative():
    # A measured-WORSE item (base 0.44) with a big comp bump (final 0.58) must not out-rank a
    # comp-blind-better heuristic sibling (0.55): it competes at its measured base fit, so the comp
    # bump can never promote an item whose own measurement says it loses.
    neg = {"name": "N", "kind": "gadget", "fit": 0.58, "comp_delta": 0.14, "comp_flipped": False,
           "recommended": False, "source": "winrate", "why": "Measured −2.0% win rate vs siblings."}
    heur = {"name": "H", "kind": "gadget", "fit": 0.55, "comp_delta": 0.0, "comp_flipped": False,
            "recommended": False, "source": "heuristic", "why": "Reload: uptime."}
    L._mark_best([neg, heur], "Knockout")
    assert heur["recommended"] is True and neg["recommended"] is False


def test_comp_overlay_dedupes_and_ignores_self():
    # One opponent sent twice must not fire the >=2-threshold reads (public never-4xx endpoint),
    # and the queried brawler listed among its own enemies is filtered out.
    tank1, tank2 = _ids_by_class("Tank", 2)
    assert L._comp_overlay([tank1, tank1])["reads"] == []
    adv = L.loadout_advice(tank1, "Gem Grab", enemies=[tank1, tank2])
    assert adv["comp_reads"] == []          # self filtered -> one real enemy -> nothing fires


def test_cc_kit_overrides_correct_the_scan():
    # Audit corrections (2026-08-18): Carl/Janet's control keywords describe SELF-movement — denied;
    # Frank's pull and Sandy's sleep are real CC the first-match-wins scan bucketed elsewhere — added.
    cc = L._cc_kit_ids()
    assert R.brawler_by_name("Carl").id not in cc
    assert R.brawler_by_name("Janet").id not in cc
    assert R.brawler_by_name("Frank").id in cc
    assert R.brawler_by_name("Sandy").id in cc
    assert R.brawler_by_name("Darryl").id in cc          # scan-flagged, audit-confirmed (Tar Barrel)


def test_cc_heavy_fires_only_on_full_cc_team():
    # cc_heavy thresholds at 3 (the corrected CC-kit base rate is ~54% of the roster, so 2-of-3
    # would fire in most drafts): all three enemies must carry CC kits.
    cc3 = sorted(L._cc_kit_ids())[:3]
    non_cc = next(b.id for b in R.load_brawlers() if b.id not in L._cc_kit_ids())
    assert "cc_heavy" in L._comp_overlay(cc3)["fired"]
    assert "cc_heavy" not in L._comp_overlay(cc3[:2] + [non_cc])["fired"]


def test_cc_heavy_requires_the_whole_enemies_list():
    # The read means "the FULL enemy team carries CC" — a hand-crafted longer list (the public
    # endpoint accepts up to 5 ids) must not fire it at 3-of-4+, or the full-team threshold's
    # base-rate rationale collapses.
    cc3 = sorted(L._cc_kit_ids())[:3]
    non_cc = next(b.id for b in R.load_brawlers() if b.id not in L._cc_kit_ids())
    assert "cc_heavy" in L._comp_overlay(cc3)["fired"]
    assert "cc_heavy" not in L._comp_overlay(cc3 + [non_cc])["fired"]


def _synthetic_gear_guide():
    return {"note": "", "gears": [
        {"name": "Alpha", "effect": "a", "description": "", "base": 0.60},
        {"name": "Mid", "effect": "m", "description": "", "base": 0.58},
        {"name": "Bumped", "effect": "b", "description": "", "base": 0.55, "vs": {"tanky": 0.10}},
    ]}


def test_gear_comp_flip_is_flagged(monkeypatch):
    # A gear that cracks the top-2 only because of its comp offset carries comp_flipped — the same
    # contract as the accessory ★ PICK · COMP badge.
    monkeypatch.setattr(L, "_gear_guide", _synthetic_gear_guide)
    comp = L._comp_overlay(_ids_by_class("Tank", 2))          # tanky (and aggro) fire
    gears = {g["name"]: g for g in L._gear_items("Tank", "Gem Grab", 16000000, None, comp)}
    assert gears["Bumped"]["recommended"] is True and gears["Bumped"]["comp_flipped"] is True
    assert gears["Alpha"]["recommended"] is True and gears["Alpha"]["comp_flipped"] is False
    assert gears["Mid"]["recommended"] is False               # displaced by the comp pick
    # And without a comp: the blind top-2, nothing flagged.
    blind = {g["name"]: g for g in L._gear_items("Tank", "Gem Grab", 16000000, None, None)}
    assert blind["Bumped"]["recommended"] is False
    assert all(g["comp_flipped"] is False for g in blind.values())


def test_gear_comp_cannot_launder_measured_negative(monkeypatch):
    # A gear measured WORSE than its siblings (base fit <= 0.5) competes at its base fit: the comp
    # bump must not push it into the recommended top-2 — mirror of the accessory rank cap.
    monkeypatch.setattr(L, "_gear_guide", lambda: {"note": "", "gears": [
        {"name": "Good", "effect": "g", "description": "", "base": 0.55},
        {"name": "Mid", "effect": "m", "description": "", "base": 0.50},
        {"name": "Bad", "effect": "b", "description": "", "base": 0.40, "vs": {"tanky": 0.15}},
    ]})
    bid = 16000000
    table = {"version": 1, "meta": {"gear_ids_by_name": {"bad": 999}}, "brawler_baseline": {},
             "cells": {f"{bid}:999": {"item_type": "gear", "delta": -0.03, "significant": 1,
                                      "item_winrate": 0.47, "n_eff": 240, "n_players": 61,
                                      "n_eff_rest": 300}}}
    comp = L._comp_overlay(_ids_by_class("Tank", 2))
    gears = {g["name"]: g for g in L._gear_items("Tank", "Gem Grab", bid, table, comp)}
    assert gears["Bad"]["source"] == "winrate" and gears["Bad"]["comp_delta"] > 0
    assert gears["Bad"]["recommended"] is False               # rank-capped at its measured base fit
    assert gears["Good"]["recommended"] is True and gears["Mid"]["recommended"] is True


def test_malformed_gear_vs_degrades_not_errors(monkeypatch):
    # The gear guide's fail-safe contract extends to junk `vs` values: null, strings, and non-dict
    # shapes contribute nothing rather than 500ing the comp-aware endpoint.
    monkeypatch.setattr(L, "_gear_guide", lambda: {"note": "", "gears": [
        {"name": "NullVs", "effect": "x", "description": "", "base": 0.50, "vs": {"tanky": None}},
        {"name": "StrVs", "effect": "x", "description": "", "base": 0.50, "vs": {"tanky": "lots"}},
        {"name": "NonDict", "effect": "x", "description": "", "base": 0.50, "vs": "tanky"},
        {"name": "BadBase", "effect": "x", "description": "", "base": "junk", "modes": 7},
    ]})
    comp = L._comp_overlay(_ids_by_class("Tank", 2))
    gears = {g["name"]: g for g in L._gear_items("Tank", "Gem Grab", 16000000, None, comp)}
    assert all(g["comp_delta"] == 0.0 and g["comp_why"] == [] for g in gears.values())
    assert gears["BadBase"]["fit"] == pytest.approx(0.4)      # junk base -> default prior, no crash


def test_gear_comp_offsets_from_vs_dict():
    # Gears extend the curated additive idiom with a sparse per-gear "vs" dict: fired reads sum
    # into the fit (clamped), recorded on comp_delta with signed chips — same contract as
    # accessories. Gears without a matching "vs" key stay untouched.
    marks = _ids_by_class("Marksman", 2)
    b = R.brawler_by_name("Shelly")
    blind = {g["name"]: g for g in L.loadout_advice(b.id, "Knockout")["gears"]}
    comp = {g["name"]: g for g in L.loadout_advice(b.id, "Knockout", enemies=marks)["gears"]}
    assert comp["Health"]["comp_delta"] > 0 and comp["Health"]["comp_why"]      # vs: poke
    assert comp["Health"]["fit"] - comp["Health"]["comp_delta"] == pytest.approx(blind["Health"]["fit"])
    assert comp["Gadget Charge"]["comp_delta"] == 0.0                            # no "vs" entry
    assert comp["Gadget Charge"]["fit"] == blind["Gadget Charge"]["fit"]


def test_mark_best_no_flip_when_winner_unchanged():
    a = {"name": "A", "kind": "gadget", "fit": 0.80, "comp_delta": 0.05, "comp_flipped": False,
         "recommended": False, "source": "heuristic", "why": "Sustain: survive."}
    b = {"name": "B", "kind": "gadget", "fit": 0.60, "comp_delta": 0.0, "comp_flipped": False,
         "recommended": False, "source": "heuristic", "why": "Vision: reveal."}
    L._mark_best([a, b], "Knockout")
    assert a["recommended"] is True and a["comp_flipped"] is False  # wins with and without comp


def test_loadout_endpoint_enemies_param():
    # First API-level test for this endpoint: CSV parsing is defensive (junk tokens skipped, never
    # a 4xx) and the unknown-brawler empty-body contract survives the new param.
    from fastapi.testclient import TestClient
    from bsdraft.api.main import app, _parse_id_csv
    assert _parse_id_csv("16000010, junk,,16000024") == [16000010, 16000024]
    assert _parse_id_csv(None) == []
    assert _parse_id_csv(",".join(str(i) for i in range(10))) == list(range(5))  # capped
    client = TestClient(app)
    b = R.brawler_by_name("Shelly")
    tanks = ",".join(str(i) for i in _ids_by_class("Tank", 2))
    r = client.get("/api/loadout", params={"brawler": b.id, "mode": "Knockout", "enemies": tanks})
    assert r.status_code == 200 and r.json()["comp_reads"]
    r = client.get("/api/loadout", params={"brawler": b.id, "mode": "Knockout", "enemies": "x,y"})
    assert r.status_code == 200 and r.json()["comp_reads"] == []   # junk degrades to comp-blind
    r = client.get("/api/loadout", params={"brawler": 999_999_999, "mode": "Knockout", "enemies": tanks})
    assert r.status_code == 200 and r.json()["brawler_name"] == ""  # unknown brawler still empty body


def test_parse_roster_retains_owned_item_ids():
    player = {"brawlers": [{
        "id": 16000000, "power": 11,
        "starPowers": [{"id": 1, "name": "A"}],
        "gadgets": [{"id": 2, "name": "G"}, {"id": 3, "name": "H"}],
        "gears": [{"id": 5, "name": "Speed", "level": 3}],
    }]}
    m = mastery.parse_roster(player)[16000000]
    assert m.owned_star_powers == (1,)
    assert m.owned_gadgets == (2, 3)
    assert m.owned_gears == ({"id": 5, "name": "Speed", "level": 3},)
    # Backwards-compatible booleans still derived.
    assert m.has_starpower and m.has_gadget and m.has_gears


def test_all_false_buffies_do_not_penalize_a_brawler_without_released_buffies():
    # R-T has no Buffies in the game — its object is all-False even on maxed top-100 rosters. The
    # curated availability policy, not the fixed three-key object, must decide whether false means
    # "missing". This is the false-positive that caused Buffy scoring to be removed originally.
    rt = R.brawler_by_name("R-T")
    assert rt is not None
    player = {"brawlers": [{
        "id": rt.id, "power": 11,
        "starPowers": [{"id": 1}], "gadgets": [{"id": 2}],
        "gears": [{"id": 5, "name": "Speed", "level": 3}],
        "hyperCharges": [{"id": 9}],
        "buffies": {"gadget": False, "starPower": False, "hyperCharge": False},
    }]}
    m = mastery.parse_roster(player)[rt.id]
    assert m.build == 1.0
    assert m.gaps() == []


def test_eligible_buffies_are_normalized_into_build_and_exact_gaps():
    gus = R.brawler_by_name("Gus")
    assert gus is not None and R.has_buffies(gus.id)
    player = {"brawlers": [{
        "id": gus.id, "power": 11,
        "starPowers": [{"id": 1}], "gadgets": [{"id": 2}],
        "gears": [{"id": 5, "name": "Speed", "level": 3}],
        "hyperCharges": [{"id": 9}],
        "buffies": {"gadget": False, "starPower": True, "hyperCharge": False},
    }]}
    m = mastery.parse_roster(player)[gus.id]
    assert m.buffies == {"gadget": False, "star_power": True, "hypercharge": False}
    assert m.build == 0.8       # 7 base shares + 1 owned Buffy, over 10 known shares
    assert m.gaps() == ["no gadget buffie", "no hyper buffie"]


def test_absent_buffie_object_is_unknown_and_neutral():
    gus = R.brawler_by_name("Gus")
    player = {"brawlers": [{
        "id": gus.id, "power": 11,
        "starPowers": [{"id": 1}], "gadgets": [{"id": 2}],
        "gears": [{"id": 5, "name": "Speed", "level": 3}],
        "hyperCharges": [{"id": 9}],
    }]}
    m = mastery.parse_roster(player)[gus.id]
    assert m.buffies is None
    assert m.build == 1.0 and m.gaps() == []
