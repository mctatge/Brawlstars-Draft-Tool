"""Offline regressions for publication failures, overload, and home-profile outages."""
import asyncio
import gzip
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from bsdraft.api import main
from bsdraft.collect.client import AuthError, LiveProfiles, LiveProfileUnavailable
from bsdraft.data import sync
from bsdraft.data.balance_eras import current_balance_era
from bsdraft.engine.stats import DraftStats
from bsdraft.engine.stats_store import stats_payload


def _transport(monkeypatch, body, status=200):
    client = httpx.Client
    requests = []
    def response(request):
        requests.append(request)
        return httpx.Response(status, content=body, headers={"ETag": "new-etag"})
    monkeypatch.setattr(sync.httpx, "Client", lambda **kwargs: client(
        transport=httpx.MockTransport(response), **kwargs))
    return requests


def _paths(tmp_path):
    dest, etag, sha = (tmp_path / name for name in ("artifact", "etag", "sha"))
    dest.write_bytes(b"last-good")
    etag.write_text("old-etag")
    sha.write_text("old-sha")
    return dest, etag, sha


@pytest.mark.parametrize("label,body", [
    ("model", b"not-an-npz"),
    ("stats", b'{"global":{}}'),
    ("rank index", b"not-an-index"),
    ("meta report", b'{"shifted": false}'),
    ("itemstats", b'{"cells": []}'),
    ("matches", b'{"error":"server failed"}\n'),
])
def test_http200_invalid_publication_preserves_file_and_get_metadata(monkeypatch, tmp_path, label, body):
    paths = _paths(tmp_path)
    _transport(monkeypatch, body)
    assert not sync._sync_file("https://offline.invalid/data", *paths, 1, label)
    assert [path.read_bytes() for path in paths] == [b"last-good", b"old-etag", b"old-sha"]
    assert sync.sync_status()[label]["error"]
    assert not (tmp_path / "artifact.tmp").exists()


def test_truncated_gzip_is_rejected_even_when_json_body_is_complete(monkeypatch, tmp_path):
    paths = _paths(tmp_path)
    _transport(monkeypatch, gzip.compress(b'{"ok":true}')[:-8])
    called = []
    assert not sync._sync_file("https://offline.invalid/data", *paths, 1, "test",
                               validator=lambda path: called.append(path))
    assert called == []
    assert paths[0].read_bytes() == b"last-good"
    assert paths[1].read_text() == "old-etag"


def test_validated_update_promotes_file_and_metadata_together(monkeypatch, tmp_path):
    paths = _paths(tmp_path)
    requests = _transport(monkeypatch, gzip.compress(b'{"ok":true}'))
    seen = []
    assert sync._sync_file("https://offline.invalid/data", *paths, 1, "test",
                          validator=lambda path: seen.append(json.loads(path.read_text())))
    assert seen == [{"ok": True}]
    assert paths[0].read_bytes() == b'{"ok":true}'
    assert paths[1].read_text() == "new-etag"
    assert requests[0].headers["If-None-Match"] == "old-etag"
    assert sync.sync_status()["test"]["last_success"]
    assert sync.sync_status()["test"]["error"] is None


def test_stats_validation_rejects_wrong_balance_era(monkeypatch, tmp_path):
    paths = _paths(tmp_path)
    table = DraftStats(matches=[])
    body = json.dumps(stats_payload(table, {})).encode()
    _transport(monkeypatch, body)
    assert current_balance_era() is not None
    assert not sync._sync_file("https://offline.invalid/stats", *paths, 1, "stats")
    assert paths[0].read_bytes() == b"last-good"


def test_valid_current_era_stats_are_accepted(monkeypatch, tmp_path):
    paths = _paths(tmp_path)
    era = current_balance_era()
    table = DraftStats(matches=[], analysis_era_id=era.id, analysis_start_ts=era.start_ts)
    body = json.dumps(stats_payload(table, {})).encode()
    _transport(monkeypatch, gzip.compress(body))
    assert sync._sync_file("https://offline.invalid/stats", *paths, 1, "stats")


def test_every_personal_build_respects_cap_and_never_waits(monkeypatch):
    started, release = threading.Barrier(3), threading.Event()
    monkeypatch.setattr(main, "_personal_build_slots", threading.BoundedSemaphore(2))
    monkeypatch.setattr(main, "_personal_cache", {})
    monkeypatch.setattr(main, "_personal_locks", {})
    monkeypatch.setattr(main, "_engine", SimpleNamespace(stats=SimpleNamespace(analysis_start_ts=0)))
    def build(*args, **kwargs):
        started.wait(timeout=3)
        assert release.wait(3)
        return "history"
    monkeypatch.setattr(main, "build_personal_stats", build)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(main._rebuild_personal, tag) for tag in ("PYY", "PQQ")]
        try:
            started.wait(timeout=3)
            assert main._rebuild_personal("PLL") is None
            assert main._personal_cache == {}
        finally:
            release.set()
        assert [future.result(timeout=3) for future in futures] == ["history", "history"]


def test_roster_and_rank_share_profile_singleflight_and_limiter(monkeypatch):
    profiles = LiveProfiles()
    monkeypatch.setattr(main, "_profiles", profiles)
    monkeypatch.setattr(main, "_roster_cache", {})
    monkeypatch.setattr(main, "_rank_cache", {})
    calls, limiters = [], []
    class Client:
        def __init__(self, rate_limiter): limiters.append(rate_limiter)
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get_player(self, tag):
            calls.append(tag)
            await asyncio.sleep(0)
            return {"name": "tester", "rankedRank": 4, "brawlers": []}
    monkeypatch.setattr(main, "BrawlStarsClient", Client)
    async def scenario():
        roster, rank = await asyncio.gather(main.roster("#PYY"), main._live_rank("PYY"))
        assert roster.loaded and rank[0] == "ok"
        await profiles.get("PQQ", client_factory=Client)
    asyncio.run(scenario())
    assert calls == ["PYY", "PQQ"]
    assert limiters[0] is limiters[1]


def test_live_profile_failure_is_briefly_cached_and_sanitized():
    profiles, calls = LiveProfiles(), []
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get_player(self, tag):
            calls.append(tag)
            raise RuntimeError("secret-token in upstream response")
    async def scenario():
        for _ in range(2):
            with pytest.raises(LiveProfileUnavailable) as error:
                await profiles.get("PYY", client_factory=Client)
            assert "secret-token" not in str(error.value)
    asyncio.run(scenario())
    assert calls == ["PYY"]


def test_auth_failure_opens_short_shared_circuit():
    profiles, calls = LiveProfiles(), []
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get_player(self, tag):
            calls.append(tag)
            raise AuthError(403, "private detail")
    async def scenario():
        for tag in ("PYY", "PQQ"):
            with pytest.raises(LiveProfileUnavailable):
                await profiles.get(tag, client_factory=Client)
    asyncio.run(scenario())
    assert calls == ["PYY"]


def test_live_profile_deadline_covers_whole_request():
    profiles = LiveProfiles(deadline=0.01)
    cancelled = []
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get_player(self, tag):
            try: await asyncio.Event().wait()
            finally: cancelled.append(tag)
    async def scenario():
        with pytest.raises(LiveProfileUnavailable):
            await asyncio.wait_for(profiles.get("PYY", client_factory=Client), timeout=1)
    asyncio.run(scenario())
    assert cancelled == ["PYY"]


def test_live_profile_pending_queue_is_bounded():
    profiles, started, release = LiveProfiles(max_inflight=1), asyncio.Event(), asyncio.Event()
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get_player(self, tag):
            started.set()
            await release.wait()
            return {"name": "test"}
    async def scenario():
        first = asyncio.create_task(profiles.get("PYY", client_factory=Client))
        await started.wait()
        try:
            with pytest.raises(LiveProfileUnavailable, match="busy"):
                await profiles.get("PQQ", client_factory=Client)
        finally:
            release.set()
        assert await first == {"name": "test"}
    asyncio.run(scenario())


def test_configured_missing_meta_does_not_scan_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, "meta_report_url", "https://offline.invalid/meta")
    monkeypatch.setattr(sync, "META_REPORT_PATH", tmp_path / "absent")
    monkeypatch.setattr(main, "detect_drift", lambda: pytest.fail("public meta replayed matches"))
    response = main.meta()
    assert response.n_recent == 0 and "unavailable" in response.note


def test_fallback_stats_refresh_when_matches_change_but_artifact_does_not(monkeypatch):
    monkeypatch.setattr(main, "_engine", SimpleNamespace(stats=SimpleNamespace(n=123), bracket_stats={}))
    monkeypatch.setattr(main, "_stats_source", "rebuild")
    for key in ("model_url", "model_manifest_url", "meta_report_url", "rank_index_url", "itemstats_url"):
        monkeypatch.setattr(main.settings, key, "")
    monkeypatch.setattr(main.settings, "data_url", "https://offline.invalid/data")
    monkeypatch.setattr(main.settings, "stats_url", "https://offline.invalid/stats")
    monkeypatch.setattr(sync, "sync_matches", lambda _: True)
    monkeypatch.setattr(sync, "sync_stats", lambda _: False)
    monkeypatch.setattr(main, "count_matches", lambda: 999)
    new_stats = SimpleNamespace(n=456)
    monkeypatch.setattr(main, "_build_stats", lambda: (new_stats, {}))
    ticks = []
    async def sleep(_):
        if ticks: raise asyncio.CancelledError()
        ticks.append(True)
    monkeypatch.setattr(main.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError): asyncio.run(main._refresh_loop())
    assert main._engine.stats is new_stats


def test_health_does_not_report_empty_service_ready(monkeypatch):
    monkeypatch.setattr(main, "_engine", None)
    response = TestClient(main.app).get("/api/health")
    assert response.status_code == 503
    assert response.json()["ready"] is False
