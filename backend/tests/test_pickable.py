"""The released/pickable split: a datamined-but-unshipped brawler must never reach a draft
surface, yet must stay in the model's pinned embedding vocabulary.

The catalog carries brawlers Supercell has added to the game files but not shipped, flagged
``released:false`` (Buzz Lightyear, id 16000088, is the live example). Two invariants are in
tension and both are pinned here:

  * ``load_brawlers()`` / ``brawler_index()`` keep EVERY entry, released or not — the trained
    model's ``winprob.npz`` vocabulary was exported from that full, id-sorted list, so dropping
    an unreleased row would slide every later brawler onto a neighbour's trained embedding.
  * the draft surfaces — pick candidates, ban targets, the ``/api/reference`` pool the frontend
    renders — read ``pickable_brawlers()`` instead, so an unshipped brawler is never offered.

    PYTHONPATH=backend python -m pytest backend/tests/test_pickable.py   # or run directly
"""
from __future__ import annotations

from bsdraft.data import reference as R
from bsdraft.engine import bans as B
from bsdraft.engine.engine import DraftEngine
from bsdraft.engine.state import DraftState
from bsdraft.engine.stats import DraftStats

# The unreleased set as the committed snapshot currently reads it.
UNRELEASED = tuple(b for b in R.load_brawlers() if not b.released)
BUZZ = 16000088


def _state() -> DraftState:
    mp = next(iter(R.load_ranked_maps()))
    return DraftState(map_id=mp.id, mode=mp.mode)


def test_snapshot_still_has_an_unreleased_brawler_to_exercise():
    # These tests only prove anything while the catalog carries a datamined entry. Buzz Lightyear
    # is that entry today; if this fails the catalog shipped everything (or Buzz released and was
    # dropped) — refresh the expectation rather than deleting the coverage.
    assert UNRELEASED, "no released:false brawler in the snapshot — update this fixture"
    assert any(b.id == BUZZ for b in UNRELEASED), "Buzz Lightyear no longer the unreleased example"


def test_load_brawlers_and_index_keep_unreleased_entries():
    # The pinned-vocab invariant: the full list and its contiguous index must span every entry,
    # or the model's embedding rows shift out from under the trained checkpoint.
    load_ids = {b.id for b in R.load_brawlers()}
    idx = R.brawler_index()
    for b in UNRELEASED:
        assert b.id in load_ids
        assert b.id in idx
    assert len(idx) == len(R.load_brawlers())
    # Contiguous 0..N-1, so serve.py's pinned rows line up.
    assert sorted(idx.values()) == list(range(len(load_ids)))


def test_pickable_is_load_brawlers_minus_the_unreleased():
    pick = R.pickable_brawlers()
    assert all(b.released for b in pick)
    pick_ids = {b.id for b in pick}
    load_ids = {b.id for b in R.load_brawlers()}
    unreleased_ids = {b.id for b in UNRELEASED}
    assert pick_ids == load_ids - unreleased_ids
    assert BUZZ not in pick_ids
    assert len(pick) == len(R.load_brawlers()) - len(UNRELEASED)


def test_candidates_never_offer_an_unreleased_brawler():
    cands = set(DraftEngine().candidates(_state()))
    assert cands.isdisjoint(b.id for b in UNRELEASED)
    # …and it's exactly the pickable pool minus what's off the board (nothing here).
    assert cands == {b.id for b in R.pickable_brawlers()}


def test_ban_pool_never_offers_an_unreleased_brawler():
    # No model -> threat-only ordering, but the pool it ranks is still pickable-only.
    rows = B.recommend(_state(), DraftStats(), None, top=9999)
    ids = {r.brawler_id for r in rows}
    assert ids.isdisjoint(b.id for b in UNRELEASED)
    assert ids == {b.id for b in R.pickable_brawlers()}


def test_api_reference_serves_only_pickable_brawlers():
    from bsdraft.api import main as M
    ids = {b.id for b in M.reference().brawlers}
    assert ids.isdisjoint(b.id for b in UNRELEASED)
    assert ids == {b.id for b in R.pickable_brawlers()}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
