"""Unit tests for the crawler's auth-failure handling (bsdraft.collect.crawler).

Pins the fix for the July 2026 silent stall: the home IP rotated off the Supercell key's
allow-list, every request 403'd (accessDenied.invalidIp), and the crawler swallowed each
one with ``continue`` — stamping every dequeued player "scanned" with zero data for a
month. An :class:`AuthError` must now abort the run (so collect.py can alert) *without*
consuming the players it never actually fetched. Fake client, temp dirs — no network.

    PYTHONPATH=backend python -m pytest backend/tests/test_collect_auth.py    # or run directly
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import bsdraft.collect.crawler as crawler_mod
from bsdraft.collect.client import AuthError, BrawlStarsError
from bsdraft.collect.crawler import Crawler


class FakeClient:
    """Raises ``exc`` from every battlelog fetch (None = return no battles)."""

    def __init__(self, exc: Exception | None = None):
        self.exc = exc
        self.calls = 0

    async def get_battlelog(self, tag: str) -> list:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return []

    async def get_top_players(self, country: str) -> list:
        if self.exc is not None:
            raise self.exc
        return []


def _mk(client: FakeClient, tmp: Path, revisit_after: float = 0.0) -> Crawler:
    """Point the module's state paths into a temp dir and build a crawler."""
    crawler_mod.RAW_DIR = tmp
    crawler_mod.MATCHES_PATH = tmp / "matches.jsonl"
    crawler_mod.VISITED_PATH = tmp / "visited_tags.txt"
    return Crawler(client, revisit_after=revisit_after)


def _run_expecting_auth_error(crawler: Crawler) -> None:
    try:
        asyncio.run(crawler.run(target_matches=10))
    except AuthError:
        return
    raise AssertionError("crawler.run swallowed the AuthError")


def test_auth_error_aborts_without_burning_the_queue():
    with tempfile.TemporaryDirectory() as d:
        c = _mk(FakeClient(AuthError(403, "accessDenied.invalidIp")), Path(d))
        c._enqueue("AAA")
        _run_expecting_auth_error(c)
        # The player was never actually fetched: not stamped scanned (in memory or in the
        # compacted file), and back in the frontier for the next run.
        assert "AAA" not in c.visited
        assert "AAA" not in crawler_mod.VISITED_PATH.read_text(encoding="utf-8")
        assert "AAA" in c.frontier


def test_auth_error_restores_prior_scan_timestamp():
    with tempfile.TemporaryDirectory() as d:
        crawler = _mk(FakeClient(AuthError(403, "accessDenied.invalidIp")), Path(d),
                      revisit_after=1.0)
        crawler.visited["AAA"] = 123.0  # scanned long ago -> due for a re-scan
        crawler._enqueue("AAA")
        _run_expecting_auth_error(crawler)
        assert crawler.visited["AAA"] == 123.0  # not overwritten with "now"


def test_auth_error_stops_the_run_immediately():
    with tempfile.TemporaryDirectory() as d:
        client = FakeClient(AuthError(403, "accessDenied.invalidIp"))
        c = _mk(client, Path(d))
        for tag in ("AAA", "BBB", "CCC"):
            c._enqueue(tag)
        _run_expecting_auth_error(c)
        assert client.calls == 1  # no per-player grind through a dead key


def test_not_found_still_skips_and_stamps_the_player():
    with tempfile.TemporaryDirectory() as d:
        c = _mk(FakeClient(BrawlStarsError(404, "not found")), Path(d))
        c._enqueue("AAA")
        assert asyncio.run(c.run(target_matches=10)) == 0  # completes, no raise
        assert "AAA" in c.visited  # per-player failures still consume the slot


def test_seed_propagates_auth_error():
    with tempfile.TemporaryDirectory() as d:
        c = _mk(FakeClient(AuthError(403, "accessDenied.invalidIp")), Path(d))
        try:
            asyncio.run(c.seed(["global"]))
        except AuthError:
            return
        raise AssertionError("seed swallowed the AuthError")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
