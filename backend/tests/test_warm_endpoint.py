"""/api/warm: the client-triggered personal-stats pre-warm on the scoring host.

Exists because rank resolution moved to the keyed roster tunnel: the warm that /api/rank fires
lands on the tunnel host, so the host that actually scores /api/recommend stayed cold and the
day's first personalized pick blocked on the full dataset scan. The client now pings /api/warm
(on API_BASE) whenever a tag resolves; these tests pin the endpoint's contract.

The headline test gates the stubbed build on an Event the test only opens AFTER the response
arrives — so a regression to a synchronous in-request build (the exact stall this endpoint
exists to prevent) fails loudly instead of passing on a fast stub.
"""
from __future__ import annotations

import threading
import time
import types

from fastapi.testclient import TestClient

from bsdraft.api import main as M

TAG = "#2PP00YCV"   # plausible: 8 chars from the Supercell tag alphabet (see _plausible_tag)


def _reset(monkeypatch, build):
    """Point the app at a stub engine + personal-stats builder and clear the warm caches."""
    monkeypatch.setattr(M, "_engine", types.SimpleNamespace(stats="fallback-stats"))
    monkeypatch.setattr(M, "build_personal_stats", build)
    M._personal_cache.clear()
    M._personal_locks.clear()
    return TestClient(M.app)


def _wait_for(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_warm_returns_while_the_build_is_still_running(monkeypatch):
    # The endpoint's whole contract: respond immediately, scan in the background. The stub build
    # BLOCKS until the test opens the gate — which only happens after the response is in hand —
    # so an implementation that ran the scan in-request would trip the in-build assert and 500.
    gate = threading.Event()
    started = threading.Event()

    def build(tag, fallback=None, **kw):
        started.set()
        assert gate.wait(5), "build ran in-request: the gate only opens after the response"
        return ("personal", tag)

    c = _reset(monkeypatch, build)
    r = c.get("/api/warm", params={"tag": TAG})
    assert r.status_code == 200 and r.json() == {"ok": True}
    key = M.normalize_tag(TAG)
    assert key not in M._personal_cache          # response beat the build — it's truly background
    assert _wait_for(started.is_set), "background build never started"
    gate.set()
    assert _wait_for(lambda: key in M._personal_cache), "cache was not populated"
    assert M._personal_cache[key][1] == ("personal", key)


def test_a_duplicate_warm_for_an_in_flight_tag_does_not_burn_a_slot(monkeypatch):
    # A build's result is only cached when the scan finishes, so a naive duplicate warm would
    # pass the cache check, take the second (and last) pool slot, and park on the per-tag lock
    # doing nothing — one visitor pinning the whole pool. The in-flight check must skip instead.
    gate = threading.Event()
    calls = []

    def build(tag, fallback=None, **kw):
        calls.append(tag)
        gate.wait(5)
        return "built"

    c = _reset(monkeypatch, build)
    key = M.normalize_tag(TAG)
    try:
        assert c.get("/api/warm", params={"tag": TAG}).status_code == 200
        assert _wait_for(lambda: M._personal_locks.get(key) is not None
                         and M._personal_locks[key].locked()), "first scan never started"
        assert c.get("/api/warm", params={"tag": TAG}).status_code == 200
        time.sleep(0.05)   # give a (wrongly) spawned lock-waiter a beat to grab the slot
        assert M._warm_slots.acquire(blocking=False), "duplicate warm consumed the second slot"
        M._warm_slots.release()
    finally:
        gate.set()         # unblock the scan thread even on assertion failure
    assert _wait_for(lambda: key in M._personal_cache)
    assert calls == [key]  # the duplicate triggered no second scan


def test_blank_tag_is_a_no_op_not_an_error(monkeypatch):
    calls = []

    def build(tag, fallback=None, **kw):
        calls.append(tag)
        return "never"

    c = _reset(monkeypatch, build)
    r = c.get("/api/warm", params={"tag": "   "})
    assert r.status_code == 200 and r.json() == {"ok": True}
    time.sleep(0.05)  # give a (wrongly) spawned scan thread a beat to show up in `calls`
    assert calls == [] and not M._personal_cache


def test_implausible_tags_never_reach_the_scan(monkeypatch):
    # The endpoint is unauthenticated; garbage must be dropped before spending a scan. Short
    # garbage is the worst case (it defeats the dataset scan's substring prefilter), so the
    # gate is the whole defense — pin it for wrong-length and wrong-alphabet shapes.
    calls = []

    def build(tag, fallback=None, **kw):
        calls.append(tag)
        return "never"

    c = _reset(monkeypatch, build)
    for bad in ["A", "0V", "hello!", "ABC123", "P" * 20]:
        assert c.get("/api/warm", params={"tag": bad}).status_code == 200
    time.sleep(0.05)
    assert calls == [] and not M._personal_cache


def test_an_already_warm_tag_is_not_rebuilt(monkeypatch):
    calls = []

    def build(tag, fallback=None, **kw):
        calls.append(tag)
        return "rebuilt"

    c = _reset(monkeypatch, build)
    key = M.normalize_tag(TAG)
    M._personal_cache[key] = (M._last_change, "already-warm")   # warm for the current data version
    r = c.get("/api/warm", params={"tag": TAG})
    assert r.status_code == 200
    time.sleep(0.05)
    assert calls == [] and M._personal_cache[key][1] == "already-warm"


def test_missing_tag_param_is_rejected(monkeypatch):
    c = _reset(monkeypatch, lambda *a, **kw: None)
    assert c.get("/api/warm").status_code == 422


# ---- serve-stale-while-revalidate (_personal_for) --------------------------------------------
# The refresh loop bumps _last_change whenever a dataset sync lands — including mid-draft, when
# no client warm ping is coming (those only fire on tag/map changes). Rebuilding inline there
# stalled the next pick request on the full dataset scan (~26 s live, 2026-08-25). The contract:
# a stale entry is served as-is, and the rebuild happens in the background.


def test_a_stale_entry_is_served_immediately_and_revalidated_in_background(monkeypatch):
    # The stub build BLOCKS until the gate opens, and the gate only opens after _personal_for
    # has returned — so a regression back to an inline rebuild deadlocks the call and fails
    # on the 5s gate timeout instead of passing on a fast stub.
    gate = threading.Event()
    started = threading.Event()

    def build(tag, fallback=None, **kw):
        started.set()
        assert gate.wait(5), "build ran in-request: the gate only opens after _personal_for returns"
        return "fresh-stats"

    _reset(monkeypatch, build)
    key = M.normalize_tag(TAG)
    M._personal_cache[key] = (M._last_change - 1, "stale-stats")   # entry from a prior data epoch
    try:
        assert M._personal_for(TAG) == "stale-stats"   # served as-is, no waiting on the scan
        assert M._personal_cache[key][1] == "stale-stats"   # rebuild hasn't landed yet
        assert _wait_for(started.is_set), "no background revalidation was kicked off"
    finally:
        gate.set()   # unblock the scan thread even on assertion failure
    assert _wait_for(lambda: M._personal_cache[key] == (M._last_change, "fresh-stats")), \
        "revalidation never refreshed the cache entry"


def test_repeated_stale_reads_share_one_revalidation_scan(monkeypatch):
    # Every pick request during the ~seconds-long rebuild re-enters the stale branch. The
    # in-flight check in _warm_personal must make those no-ops: one scan total, and each
    # read still gets the stale entry immediately.
    gate = threading.Event()
    calls = []

    def build(tag, fallback=None, **kw):
        calls.append(tag)
        gate.wait(5)
        return "fresh-stats"

    _reset(monkeypatch, build)
    key = M.normalize_tag(TAG)
    M._personal_cache[key] = (M._last_change - 1, "stale-stats")
    try:
        for _ in range(3):
            assert M._personal_for(TAG) == "stale-stats"
        assert _wait_for(lambda: M._personal_locks.get(key) is not None
                         and M._personal_locks[key].locked()), "revalidation scan never started"
        assert M._personal_for(TAG) == "stale-stats"   # scan in flight — still served stale
    finally:
        gate.set()
    assert _wait_for(lambda: M._personal_cache[key][1] == "fresh-stats")
    assert calls == [key], "stale reads stacked redundant scans"


def test_a_never_seen_tag_still_builds_inline(monkeypatch):
    # Serve-stale needs something to serve: with no cache entry at all there's no better
    # answer than the scan, so the miss path stays synchronous (the /api/warm ping on LOAD
    # exists to make this case rare).
    calls = []

    def build(tag, fallback=None, **kw):
        calls.append(tag)
        return "built-inline"

    _reset(monkeypatch, build)
    key = M.normalize_tag(TAG)
    assert M._personal_for(TAG) == "built-inline"      # returned synchronously
    assert M._personal_cache[key] == (M._last_change, "built-inline")
    assert calls == [key]
