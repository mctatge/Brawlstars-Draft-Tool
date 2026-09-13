"""Unit tests for the catalog watcher (bsdraft.data.catalog).

Offline and synthetic — the diff is what gates an *auto-merging* pull request, so the safety
rules (what counts as destructive, what may land unattended) are pinned here rather than
discovered in production.

    PYTHONPATH=backend python -m pytest backend/tests/test_catalog.py    # or run directly
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bsdraft.data import catalog as C
from bsdraft.data import reference as R


def brawler(bid, name, cls="Tank", rarity="Rare", sp=(), gadgets=()):
    return {
        "id": bid, "name": name,
        "class": {"name": cls}, "rarity": {"name": rarity},
        "starPowers": [{"id": i, "name": n} for i, n in sp],
        "gadgets": [{"id": i, "name": n} for i, n in gadgets],
    }


BASE = [
    brawler(1, "Shelly", sp=[(101, "Shell Shock")], gadgets=[(201, "Fast Forward")]),
    brawler(2, "Colt", cls="Marksman", sp=[(102, "Slick Boots")]),
]


# --- brawler-level changes ------------------------------------------------------

def test_no_change():
    d = C.diff_catalogs(BASE, BASE)
    assert not d.changed and not d.destructive
    assert not d.safe_to_automerge          # nothing to merge
    assert "unchanged" in d.summary()


def test_added_brawler_is_additive_and_automergeable():
    after = BASE + [brawler(3, "Nori", cls="Assassin", rarity="Legendary")]
    d = C.diff_catalogs(BASE, after)
    assert [c.name for c in d.new_brawlers] == ["Nori"]
    assert d.new_brawlers[0].detail == "Legendary / Assassin"
    assert d.changed and d.safe_to_automerge and not d.destructive
    assert d.n_before == 2 and d.n_after == 3


def test_removed_brawler_is_destructive_and_blocks_automerge():
    d = C.diff_catalogs(BASE, BASE[:1])
    assert [c.name for c in d.removed_brawlers] == ["Colt"]
    assert d.changed and not d.safe_to_automerge
    # Both the removal and the shrink are reported.
    assert any("removed" in r for r in d.destructive)
    assert any("shrank" in r for r in d.destructive)


def test_rename_class_and_rarity_changes():
    after = [brawler(1, "Shelly", sp=[(101, "Shell Shock")], gadgets=[(201, "Fast Forward")]),
             brawler(2, "Colt Jr", cls="Assassin", rarity="Epic", sp=[(102, "Slick Boots")])]
    d = C.diff_catalogs(BASE, after)
    kinds = {c.change: c.detail for c in d.brawler_changes}
    assert kinds["renamed"] == "Colt -> Colt Jr"
    assert kinds["class"] == "Marksman -> Assassin"
    assert kinds["rarity"] == "Rare -> Epic"
    # Reclassification is not destructive: nothing disappears.
    assert d.safe_to_automerge


# --- accessory-level changes ----------------------------------------------------

def test_accessory_added_and_renamed():
    after = [
        brawler(1, "Shelly", sp=[(101, "Shell Shock"), (103, "Band-Aid")],
                gadgets=[(201, "Fast Forward")]),
        brawler(2, "Colt", cls="Marksman", sp=[(102, "Slick Boots II")]),
    ]
    d = C.diff_catalogs(BASE, after)
    by = {(c.change, c.name): c for c in d.accessory_changes}
    assert ("added", "Band-Aid") in by
    assert by[("added", "Band-Aid")].kind == "star power"
    ren = [c for c in d.accessory_changes if c.change == "renamed"][0]
    assert (ren.old_name, ren.name) == ("Slick Boots", "Slick Boots II")
    assert d.safe_to_automerge          # additive + rename only


def test_accessory_description_change_is_reported():
    after = [brawler(1, "Shelly", sp=[(101, "Shell Shock")],
                     gadgets=[(201, "Fast Forward")]), BASE[1]]
    after[0]["gadgets"][0]["description"] = "Shelly dashes forward."
    d = C.diff_catalogs(BASE, after)
    changes = [c for c in d.accessory_changes if c.change == "description"]
    assert len(changes) == 1
    assert changes[0].old_description == ""
    assert changes[0].new_description == "Shelly dashes forward."
    assert d.safe_to_automerge
    assert "description updated" in d.summary()


def test_refresh_accessory_details_preserves_catalog_membership():
    current = {"list": [brawler(1, "Shelly", gadgets=[(201, "Fast Forward")])]}
    current["list"][0]["gadgets"][0]["description"] = "Old description."
    current["list"].append(brawler(99, "LocalOnly", gadgets=[(999, "Local Gadget")]))
    live = {"list": [brawler(1, "Shelly", gadgets=[(201, "New Name")]),
                      brawler(2, "New Brawler", gadgets=[(202, "New Gadget")])]}
    live["list"][0]["gadgets"][0].update({
        "path": "New-Name", "description": "Current description.",
        "descriptionHtml": "Current description.", "imageUrl": "current.png",
    })
    notes = C.refresh_accessory_details(current, live)
    item = current["list"][0]["gadgets"][0]
    assert item["name"] == "New Name"
    assert item["description"] == "Current description."
    assert current["list"][1]["name"] == "LocalOnly"
    assert current["list"][1]["gadgets"][0]["name"] == "Local Gadget"
    assert len(notes) == 1 and "description" in notes[0]


def test_accessory_removal_blocks_automerge():
    after = [brawler(1, "Shelly", sp=[(101, "Shell Shock")], gadgets=[]),  # gadget gone
             BASE[1]]
    d = C.diff_catalogs(BASE, after)
    assert d.changed and not d.safe_to_automerge
    assert any("gadget removed" in r for r in d.destructive)


# --- duplicated-accessory guard ---------------------------------------------------

def test_dedupe_strips_duplicate_from_the_brawler_the_description_disowns():
    # The real 2026-08 case: the live API served Brock's gadgets under Bolt too.
    brock = brawler(1, "Brock", gadgets=[(201, "Rocket Laces")])
    bolt = brawler(2, "Bolt", gadgets=[(201, "Rocket Laces"), (202, "Oil Change")])
    for b in (brock, bolt):
        b["gadgets"][0]["description"] = "Brock jumps to targeted area."
    bolt["gadgets"][1]["description"] = "Bolt gains a shield."
    notes = C.dedupe_accessories({"list": [brock, bolt]})
    assert [g["id"] for g in brock["gadgets"]] == [201]     # rightful owner keeps it
    assert [g["id"] for g in bolt["gadgets"]] == [202]      # impostor copy stripped
    assert len(notes) == 1 and "Bolt" in notes[0] and "Rocket Laces" in notes[0]


def test_dedupe_refuses_when_descriptions_cannot_pick_an_owner():
    a = brawler(1, "Shelly", gadgets=[(201, "Mystery Box")])
    b = brawler(2, "Colt", gadgets=[(201, "Mystery Box")])
    # Names neither claimant …
    a["gadgets"][0]["description"] = b["gadgets"][0]["description"] = "Does something."
    with pytest.raises(ValueError, match="Mystery Box"):
        C.dedupe_accessories({"list": [a, b]})
    # … and naming both is just as ambiguous.
    a["gadgets"][0]["description"] = b["gadgets"][0]["description"] = "Shelly and Colt swap."
    with pytest.raises(ValueError, match="single out"):
        C.dedupe_accessories({"list": [a, b]})


def test_dedupe_leaves_a_clean_catalog_untouched():
    payload = {"list": [brawler(1, "Shelly", gadgets=[(201, "Fast Forward")]),
                        brawler(2, "Colt", cls="Marksman", gadgets=[(202, "Speedloader")])]}
    before = json.dumps(payload, sort_keys=True)
    assert C.dedupe_accessories(payload) == []
    assert json.dumps(payload, sort_keys=True) == before


# --- degraded-payload safety rules ----------------------------------------------

def test_losing_a_class_is_destructive():
    # A payload that drops class data would leave the brawler UNCLASSIFIED and quietly degrade
    # composition reasoning — nothing is "removed", so the counts alone wouldn't catch it.
    after = [brawler(1, "Shelly", cls="Unknown", sp=[(101, "Shell Shock")],
                     gadgets=[(201, "Fast Forward")]), BASE[1]]
    d = C.diff_catalogs(BASE, after)
    assert not d.safe_to_automerge
    assert any("class lost" in r for r in d.destructive)


def test_gaining_a_class_is_not_destructive():
    # The common, benign direction: Unknown -> a real class (what the notes bridge does).
    before = [brawler(1, "Nori", cls="Unknown")]
    after = [brawler(1, "Nori", cls="Assassin")]
    d = C.diff_catalogs(before, after)
    assert d.changed and d.safe_to_automerge


def test_mass_edit_burst_requires_review():
    after = [brawler(i, f"B{i}", cls="Assassin", rarity="Epic") for i in range(1, 9)]
    before = [brawler(i, f"B{i}", cls="Tank", rarity="Rare") for i in range(1, 9)]
    d = C.diff_catalogs(before, after)
    assert not d.safe_to_automerge
    assert any("edits to existing brawlers" in r for r in d.destructive)


# --- apply_catalog side effects --------------------------------------------------

def _isolated(tmp: Path, overrides: dict):
    """Point catalog.py's write targets at a temp dir; returns a restore callable."""
    ov_path = tmp / "class_overrides.json"
    ov_path.write_text(json.dumps({"_updated": "2020-01-01", "overrides": overrides}),
                       encoding="utf-8")
    prev_dir, prev_ov = C.REFERENCE_DIR, R.CLASS_OVERRIDES_PATH
    C.REFERENCE_DIR = tmp
    R.CLASS_OVERRIDES_PATH = ov_path

    def restore():
        C.REFERENCE_DIR, R.CLASS_OVERRIDES_PATH = prev_dir, prev_ov
    return ov_path, restore


def test_apply_never_writes_maps():
    # Ranked-map indices are positional, so an auto-merged maps.json would shift trained rows.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _ov, restore = _isolated(tmp, {})
        try:
            written = C.apply_catalog({"list": [brawler(1, "Shelly")]})
        finally:
            restore()
        assert [p.name for p in written] == ["brawlers.json"]
        assert not (tmp / "maps.json").exists()


def test_apply_migrates_override_across_a_rename():
    # class_overrides.json is keyed by NAME; without migration a rename orphans the entry and
    # the brawler silently drops to UNCLASSIFIED.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ov_path, restore = _isolated(tmp, {"Colt": "Marksman"})
        try:
            C.apply_catalog({"list": [brawler(1, "Colt Jr")]}, None, [("Colt", "Colt Jr")])
        finally:
            restore()
        ov = json.loads(ov_path.read_text(encoding="utf-8"))["overrides"]
        assert ov == {"Colt Jr": "Marksman"}          # moved, not duplicated or dropped


def test_apply_adds_new_overrides_without_clobbering():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ov_path, restore = _isolated(tmp, {"Kaze": "Assassin"})
        try:
            C.apply_catalog({"list": [brawler(1, "Shelly")]}, {"Nori": "Assassin", "Kaze": "Tank"})
        finally:
            restore()
        ov = json.loads(ov_path.read_text(encoding="utf-8"))["overrides"]
        assert ov["Nori"] == "Assassin"
        assert ov["Kaze"] == "Assassin"               # pre-existing entry wins


# --- validation (a bad payload must never overwrite a good snapshot) -------------

def test_validate_rejects_junk():
    good = {"list": [{"id": 1, "name": "Shelly"}]}
    assert C.validate(good, "brawlers") == good["list"]
    for bad, why in (
        ({}, "no list"),
        ({"list": []}, "empty"),
        ({"list": [{"name": "no id"}]}, "missing id"),
        ({"list": [{"id": "x", "name": "str id"}]}, "non-int id"),
        ({"list": [{"id": 1}]}, "missing name"),
        ("<html>blocked</html>", "html body"),
    ):
        try:
            C.validate(bad, "brawlers")
            raise AssertionError(f"expected ValueError for {why}")
        except ValueError:
            pass


# --- PR rendering ---------------------------------------------------------------

def test_render_pr_includes_new_brawlers_accessories_and_overrides():
    after = BASE + [brawler(3, "Nori", cls="Unknown", rarity="Legendary",
                            sp=[(103, "Big Haul")], gadgets=[(203, "Sushi Snack")])]
    d = C.diff_catalogs(BASE, after)
    title, body = C.render_pr(d, "https://api.brawlapi.com/v1/brawlers", {"Nori": "Assassin"})
    assert title == "Catalog: add Nori"
    assert "Nori" in body and "`16000107`" not in body      # id comes from the diff, not hardcoded
    assert "Big Haul" in body and "Sushi Snack" in body
    assert "**Nori** → `Assassin`" in body                   # class override surfaced
    assert "Needs review" not in body                        # additive -> no review banner


def test_render_pr_flags_destructive():
    d = C.diff_catalogs(BASE, BASE[:1])
    _title, body = C.render_pr(d, "https://api.brawlapi.com/v1/brawlers")
    assert "Needs review — not auto-merged" in body
    assert "Colt" in body


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
