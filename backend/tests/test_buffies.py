"""Curated Buffy availability and the ownership/availability join.

The official roster exposes the same three all-false ownership flags both when a player owns no
Buffies and when that brawler has no Buffies to own. The checked-in cumulative policy is therefore
load-bearing: Gus must be priced, R-T must remain neutral, and malformed policy data must fail
closed.

    PYTHONPATH=backend python -m pytest backend/tests/test_buffies.py -q
"""
from __future__ import annotations

import json

from bsdraft.collect.profiles import owned_summary
from bsdraft.data import reference as R


def test_checked_in_policy_has_the_three_functional_slots_and_27_brawlers():
    ids = R.load_buffie_brawlers()
    assert R.BUFFIE_SLOTS == ("gadget", "star_power", "hypercharge")
    assert len(ids) == 27
    assert R.brawler_by_name("Gus").id in ids
    assert R.brawler_by_name("R-T").id not in ids


def test_policy_loader_fails_closed_on_bad_slots(tmp_path):
    path = tmp_path / "buffies.json"
    path.write_text(json.dumps({
        "schema": 1,
        "slots": ["gadget", "star_power", "hypercharge", "cosmetic"],
        "brawlers": {"Gus": R.brawler_by_name("Gus").id},
    }), encoding="utf-8")
    old = R.BUFFIES_PATH
    R.BUFFIES_PATH = path
    R.load_buffie_brawlers.cache_clear()
    try:
        assert R.load_buffie_brawlers() == frozenset()
    finally:
        R.BUFFIES_PATH = old
        R.load_buffie_brawlers.cache_clear()


def test_policy_skips_a_name_id_mismatch_instead_of_overcovering(tmp_path):
    path = tmp_path / "buffies.json"
    path.write_text(json.dumps({
        "schema": 1,
        "slots": list(R.BUFFIE_SLOTS),
        "brawlers": {
            "Gus": R.brawler_by_name("R-T").id,
            "Shelly": R.brawler_by_name("Shelly").id,
        },
    }), encoding="utf-8")
    old = R.BUFFIES_PATH
    R.BUFFIES_PATH = path
    R.load_buffie_brawlers.cache_clear()
    try:
        assert R.load_buffie_brawlers() == frozenset({R.brawler_by_name("Shelly").id})
    finally:
        R.BUFFIES_PATH = old
        R.load_buffie_brawlers.cache_clear()


def test_profile_snapshot_preserves_unknown_vs_explicit_none_owned():
    """Future measurement needs the same tri-state distinction the live scorer uses."""
    gus = R.brawler_by_name("Gus")
    rt = R.brawler_by_name("R-T")
    summary = owned_summary({"brawlers": [
        {
            "id": gus.id,
            "power": 11,
            "buffies": {"gadget": False, "starPower": False, "hyperCharge": False},
        },
        {"id": rt.id, "power": 11},
    ]})
    assert summary[str(gus.id)]["bf"] == {
        "gadget": False, "star_power": False, "hypercharge": False,
    }
    assert summary[str(rt.id)]["bf"] is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
