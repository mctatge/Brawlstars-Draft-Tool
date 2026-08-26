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
    """Point the app at a stub engine + personal-stats builder and clear the warm caches.

    Also drains stragglers from the PREVIOUS test before touching shared module state: a
    warm worker that outlives its test resolves ``build_personal_stats``/``_engine`` as
    module globals at call time, so it would run THIS test's stub and corrupt its ``calls``
    bookkeeping — a one-in-thousands flake with a baffling signature. Joining the threads
    and then round-tripping both semaphore slots (which doubles as a leak assert) closes
    that door and fails loudly at the START of the offending test instead."""
    for th in threading.enumerate():
        if th.name == "warm-personal":
            th.join(timeout=5)
            assert not th.is_alive(), "a warm worker from a previous test would not finish"
    for _ in range(2):
        assert M._warm_slots.acquire(timeout=1), "a previous test leaked a warm-pool slot"
    M._warm_slots.release(); M._warm_slots.release()
    monkeypatch.setattr(M, "_engine", types.SimpleNamespace(stats="fallback-stats"))
    monkeypatch.setattr(M, "build_personal_stats", build)
    M._personal_cache.clear()
    M._personal_locks.clear()
    M._warm_inflight.clear()
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


def test_a_stale_none_entry_is_served_stale_too(monkeypatch):
    # The cache maps tag -> (version, PersonalStats|None): a visitor with no labeled games
    # is cached as (version, None) — likely the most common entry on the public host. The
    # stale branch must key off the TUPLE's presence, not the stats' truthiness; a guard
    # "tidied" to `hit[1] is not None` would send every such visitor back to the inline
    # ~26 s stall after each data refresh, invisibly (their answer is None either way).
    gate = threading.Event()
    started = threading.Event()

    def build(tag, fallback=None, **kw):
        started.set()
        assert gate.wait(5), "build ran in-request for a stale None entry"
        return None    # still no labeled games

    _reset(monkeypatch, build)
    key = M.normalize_tag(TAG)
    M._personal_cache[key] = (M._last_change - 1, None)
    try:
        assert M._personal_for(TAG) is None            # served immediately, no scan wait
        assert _wait_for(started.is_set), "stale None entry was not revalidated"
    finally:
        gate.set()
    assert _wait_for(lambda: M._personal_cache[key] == (M._last_change, None))


def test_a_full_warm_pool_skips_the_stale_revalidation_and_never_queues(monkeypatch):
    # "A full pool skips, never queues" on the stale path: with both slots held, a stale
    # read must return the stale entry with NO scan scheduled and NO wait on the semaphore.
    # Run the read in a worker thread so a skip-to-queue regression (blocking acquire)
    # fails this test's join timeout instead of hanging the whole suite.
    calls = []

    def build(tag, fallback=None, **kw):
        calls.append(tag)
        return "never"

    _reset(monkeypatch, build)
    key = M.normalize_tag(TAG)
    M._personal_cache[key] = (M._last_change - 1, "stale-stats")
    assert M._warm_slots.acquire(blocking=False) and M._warm_slots.acquire(blocking=False)
    try:
        result = []
        th = threading.Thread(target=lambda: result.append(M._personal_for(TAG)), daemon=True)
        th.start()
        th.join(timeout=2)
        assert not th.is_alive(), "stale read blocked on the full warm pool instead of skipping"
        assert result == ["stale-stats"]
        time.sleep(0.05)   # give a (wrongly) scheduled scan a beat to show up
        assert calls == [] and M._personal_cache[key][1] == "stale-stats"
    finally:
        M._warm_slots.release(); M._warm_slots.release()


def test_stale_revalidations_across_many_tags_respect_the_warm_pool_bound(monkeypatch):
    # A data sync stales EVERY cached tag at once, so a burst of picks across distinct tags
    # is the scan-pile-up case _WARM_MAX_CONCURRENCY exists for. The stale path must route
    # through the bounded pool: with 4 stale tags read back-to-back, exactly 2 scans run and
    # the other 2 tags stay stale (skipped, lazily recovered) — a stale branch that spawned
    # raw threads would run all 4 concurrent full scans on the 512 MB box.
    tags = ["2PP00YCV", "8QQ22LJR", "9RR88GUC", "2VV00PYL"]
    gate = threading.Event()
    calls = []

    def build(tag, fallback=None, **kw):
        calls.append(tag)
        gate.wait(5)
        return "fresh-stats"

    _reset(monkeypatch, build)
    keys = [M.normalize_tag(t) for t in tags]
    old_version = M._last_change - 1
    for k in keys:
        M._personal_cache[k] = (old_version, "stale-stats")
    try:
        for t in tags:
            assert M._personal_for(t) == "stale-stats"   # every read serves stale instantly
        # Slot acquisition happens synchronously in the calling thread, so after 4 reads the
        # pool is deterministically drained by the first two tags and the rest were skipped.
        assert not M._warm_slots.acquire(blocking=False), "stale reads did not consume the pool"
        assert _wait_for(lambda: len(calls) == 2), "expected exactly the pool-bound scans"
    finally:
        gate.set()
    assert _wait_for(lambda: all(M._personal_cache[k][1] == "fresh-stats" for k in keys[:2]))
    assert sorted(calls) == sorted(keys[:2])
    for k in keys[2:]:
        assert M._personal_cache[k] == (old_version, "stale-stats")   # skipped, still stale


def test_duplicate_warm_before_the_worker_grabs_the_lock_takes_no_slot(monkeypatch):
    # The in-flight registry must be atomic with slot acquisition. lock.locked() alone has a
    # window — after Thread.start(), before the worker reaches the per-tag lock — where a
    # duplicate warm for the SAME tag sees the lock free, takes the last slot, and parks on
    # the lock for a whole scan. Freeze worker 1 BEFORE it can acquire anything and assert
    # the duplicate is skipped without touching the pool.
    gate = threading.Event()
    entered = threading.Event()
    real_rebuild = M._rebuild_personal
    calls = []

    def gated_rebuild(t):
        entered.set()
        assert gate.wait(5), "gate never opened for the frozen worker"
        calls.append(t)
        return real_rebuild(t)

    _reset(monkeypatch, lambda tag, fallback=None, **kw: "fresh-stats")
    monkeypatch.setattr(M, "_rebuild_personal", gated_rebuild)
    key = M.normalize_tag(TAG)
    M._personal_cache[key] = (M._last_change - 1, "stale-stats")
    try:
        M._warm_personal(key)
        assert _wait_for(entered.is_set), "first warm worker never started"
        # Worker 1 is frozen before the per-tag lock: lock.locked() is False, so only the
        # registry can catch this duplicate.
        M._warm_personal(key)
        assert M._warm_slots.acquire(blocking=False), "duplicate warm consumed the second slot"
        M._warm_slots.release()
    finally:
        gate.set()
    assert _wait_for(lambda: M._personal_cache[key][1] == "fresh-stats")
    assert calls == [key]   # one worker ran one rebuild


def test_ancient_lag_escalates_to_an_inline_rebuild_even_with_a_full_pool(monkeypatch):
    # The staleness bound: revalidation is best-effort, so an entry whose data version lags
    # beyond _STALE_LAG_MAX_SECONDS means the warm kept getting skipped — serving it stale
    # yet again would let the lag grow without limit. Past the bound the read pays the
    # blocking rebuild even when the pool is full (the pool caps background scans, not the
    # request's own right to rebuild).
    calls = []

    def build(tag, fallback=None, **kw):
        calls.append(tag)
        return "fresh-stats"

    _reset(monkeypatch, build)
    key = M.normalize_tag(TAG)
    M._personal_cache[key] = (M._last_change - M._STALE_LAG_MAX_SECONDS - 1, "ancient-stats")
    assert M._warm_slots.acquire(blocking=False) and M._warm_slots.acquire(blocking=False)
    try:
        assert M._personal_for(TAG) == "fresh-stats"   # rebuilt inline, not served ancient
        assert calls == [key]
        assert M._personal_cache[key] == (M._last_change, "fresh-stats")
    finally:
        M._warm_slots.release(); M._warm_slots.release()


def test_a_mid_scan_data_refresh_leaves_the_entry_stale_born(monkeypatch):
    # _rebuild_personal stamps the version the scan STARTED under. If a sync lands mid-scan,
    # the scan read the pre-refresh file to the end, so stamping the post-scan version would
    # serve those stats as current for a whole data epoch. The entry must come out stale-born
    # and revalidate on its next use.
    monkeypatch.setattr(M, "_last_change", M._last_change)   # register for teardown restore
    v0 = M._last_change

    def build(tag, fallback=None, **kw):
        M._last_change = v0 + 100    # a dataset sync lands while the scan is running
        return "scanned-under-v0"

    _reset(monkeypatch, build)
    key = M.normalize_tag(TAG)
    assert M._rebuild_personal(key) == "scanned-under-v0"
    assert M._personal_cache[key] == (v0, "scanned-under-v0"), \
        "entry must carry the scan-START version, not the post-scan one"
    started = threading.Event()

    def build2(tag, fallback=None, **kw):
        started.set()
        return "rebuilt-under-v1"

    monkeypatch.setattr(M, "build_personal_stats", build2)
    assert M._personal_for(TAG) == "scanned-under-v0"   # stale-born: served + revalidated
    assert _wait_for(started.is_set), "stale-born entry was never revalidated"
    assert _wait_for(lambda: M._personal_cache[key] == (v0 + 100, "rebuilt-under-v1"))
