"""`/api/roster` must not persist per-request state onto the shared engine.

`bsdraft.api.main._engine` is a single process-global :class:`DraftEngine` shared by every
request. The roster endpoint once wrote the fetched roster onto it (``_engine.roster = ...``),
so whichever tag last hit ``/api/roster`` became the engine's roster — and ``_roster_for`` (the
``/api/recommend`` personalization fallback) reads exactly that. Two visitors in a row could
then cross-contaminate the moment the home API serves recommendations.

These tests pin the invariant: resolving one tag's roster, then another's, leaves no roster on
the shared engine for a third request to observe. Everything is mocked in-memory — no disk, no
network, no live Supercell key.

    PYTHONPATH=backend python -m pytest backend/tests/test_roster_state.py
"""
from __future__ import annotations

import asyncio

import bsdraft.api.main as main
from bsdraft.api import schemas as S
from bsdraft.collect.client import normalize_tag
from bsdraft.engine.engine import DraftEngine
from bsdraft.engine.mastery import Mastery
from bsdraft.engine.stats import DraftStats


class _FakeClient:
    """Async-context stand-in for BrawlStarsClient — never touches the network."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _mastery(bid: int) -> Mastery:
    return Mastery(
        brawler_id=bid, power=11, rank=0, trophies=0, highest_trophies=0,
        has_starpower=True, has_gadget=True, has_gears=True, has_hypercharge=True,
    )


def _install(monkeypatch, by_tag):
    """Isolate the endpoint: fresh engine + empty cache, and resolve rosters from ``by_tag``
    (keyed by normalized tag) with no network."""
    # Empty stats table: a bare DraftEngine() builds DraftStats from the whole local dataset
    # (minutes on a machine that has the 1.5 GB matches.jsonl). Roster still starts None.
    monkeypatch.setattr(main, "_engine", DraftEngine(stats=DraftStats(matches=[])))
    monkeypatch.setattr(main, "_roster_cache", {})
    monkeypatch.setattr(main, "BrawlStarsClient", _FakeClient)

    async def fake_fetch_roster(client, tag):
        key = normalize_tag(tag)
        return dict(by_tag[key]), f"name-{key}"

    monkeypatch.setattr(main.mastery, "fetch_roster", fake_fetch_roster)


def test_roster_does_not_mutate_shared_engine(monkeypatch):
    tag_a, tag_b = normalize_tag("#AAA"), normalize_tag("#BBB")
    _install(monkeypatch, {tag_a: {1: _mastery(1)}, tag_b: {2: _mastery(2)}})

    resp_a = asyncio.run(main.roster(tag="#AAA"))
    assert resp_a.loaded and [o.id for o in resp_a.owned] == [1]
    # The endpoint must not have written the fetched roster onto the shared engine.
    assert main._engine.roster is None
    assert main._engine.roster_name == ""

    resp_b = asyncio.run(main.roster(tag="#BBB"))
    assert resp_b.loaded and [o.id for o in resp_b.owned] == [2]
    # Still clean after a second, different tag — no last-writer-wins residue.
    assert main._engine.roster is None
    assert main._engine.roster_name == ""


def test_third_request_cannot_observe_a_prior_tags_roster(monkeypatch):
    """The observable consequence: after two roster lookups, a personalized /api/recommend that
    sends no client roster must resolve to *no* roster — not the last visitor's."""
    tag_a, tag_b = normalize_tag("#AAA"), normalize_tag("#BBB")
    _install(monkeypatch, {tag_a: {1: _mastery(1)}, tag_b: {2: _mastery(2)}})

    asyncio.run(main.roster(tag="#AAA"))
    asyncio.run(main.roster(tag="#BBB"))

    req = S.RecommendRequest(map_id=0, mode="Brawl Ball", personalize=True)  # roster=None
    assert main._roster_for(req) is None
