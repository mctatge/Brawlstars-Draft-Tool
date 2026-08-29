"""Unreleased catalog entries stay in the vocabulary but out of every pickable pool.

A brawler flagged ``released: false`` upstream (a datamined / leaked-early entry, e.g. the
"Buzz Lightyear" collab that shipped in the initial snapshot) has no match data, so its signals
collapse to a neutral prior and it floats to the top of an otherwise-empty pick board while not
being selectable in-game. The fix filters it out of the candidate pool, the ban pool, and the
reference grid — but NOT out of ``load_brawlers`` / ``brawler_index``, which pin the model's
embedding vocabulary (dropping an entry there would shift every higher-id brawler onto a
neighbour's trained row, desyncing from ``winprob.npz``).

These tests key off the live ``released`` flag rather than a hard-coded id, so they keep asserting
the mechanism after the leaked brawler eventually releases (``released`` flips true, the entry
re-enters the pools on its own) — and degrade to a consistency check if the snapshot ever carries
no unreleased entries at all.
"""
from __future__ import annotations

import json

from bsdraft.data import reference as R
from bsdraft.engine.engine import DraftEngine
from bsdraft.engine.state import DraftState
from bsdraft.engine.stats import DraftStats


def _unreleased_ids() -> set:
    """Ids the raw snapshot flags ``released: false`` — computed from the file, not hard-coded."""
    raw = json.load(open(R.REFERENCE_DIR / "brawlers.json", encoding="utf-8"))["list"]
    return {b["id"] for b in raw if b.get("released", True) is False}


def test_pickable_is_released_only_and_a_subset_of_the_vocab():
    load_ids = {b.id for b in R.load_brawlers()}
    pick_ids = {b.id for b in R.pickable_brawlers()}
    unreleased = _unreleased_ids()

    # Every pickable brawler is released; the pool is exactly the vocab minus the unreleased set.
    assert all(b.released for b in R.pickable_brawlers())
    assert pick_ids <= load_ids
    assert pick_ids == load_ids - unreleased


def test_vocab_keeps_unreleased_entries():
    """load_brawlers()/brawler_index() must stay complete: the pinned model vocab includes the
    unreleased rows, so filtering them out of the *loader* would misalign every higher-id row."""
    load_ids = {b.id for b in R.load_brawlers()}
    index_ids = set(R.brawler_index())
    unreleased = _unreleased_ids()

    assert unreleased <= load_ids           # the loader still carries them
    assert unreleased <= index_ids          # …and so does the positional index (the vocab)
    # brawler_index() is 1:1 with load_brawlers() — the field addition changed neither.
    assert len(index_ids) == len(load_ids)


def test_unreleased_never_enters_the_candidate_pool():
    """The reported bug: an unreleased brawler surfaced as the top pick on a first-pick board."""
    unreleased = _unreleased_ids()
    if not unreleased:
        return  # nothing flagged unreleased in the current snapshot — vacuously safe

    engine = DraftEngine(stats=DraftStats(matches=[]))  # candidates() needs no stats/model
    a_map = R.load_ranked_maps()[0]
    first_pick = DraftState(map_id=a_map.id, mode=a_map.mode, we_pick_first=True)

    cands = set(engine.candidates(first_pick))
    assert cands.isdisjoint(unreleased)
    # And it is a real filter, not an empty pool.
    assert cands == {b.id for b in R.pickable_brawlers()}


def test_released_field_defaults_true_and_round_trips():
    """A snapshot missing the key must not hide a brawler; an explicit false must be carried."""
    kept = R.Brawler(id=1, name="X", cls="Damage Dealer", rarity="", star_powers=(),
                     gadgets=(), image_url="")
    hidden = R.Brawler(id=2, name="Y", cls="Damage Dealer", rarity="", star_powers=(),
                       gadgets=(), image_url="", released=False)
    assert kept.released is True            # defaulted → visible
    assert hidden.released is False
