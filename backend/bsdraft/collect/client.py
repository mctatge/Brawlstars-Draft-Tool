"""Async client for the official Brawl Stars API (https://developer.brawlstars.com).

Handles bearer auth, player-tag normalization, rate limiting, and retry/backoff. The
API is player-centric: you fetch a known player's profile or recent battle log, plus
country/global leaderboards that the crawler uses to seed player tags.
"""
from __future__ import annotations

import asyncio
import logging
import time
import weakref
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from bsdraft.config import settings

logger = logging.getLogger(__name__)

API_BASE = "https://api.brawlstars.com/v1"


def normalize_tag(tag: str) -> str:
    """'#2yulp2' -> '2YULP2' (path form, no '#')."""
    return tag.strip().lstrip("#").upper()


def encode_tag(tag: str) -> str:
    """URL-encode a tag for a path segment ('#' -> '%23')."""
    return "%23" + normalize_tag(tag)


class BrawlStarsError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


class AuthError(BrawlStarsError):
    """401/403 — bad token, or this machine's public IP fell off the key's allow-list
    (residential IPs rotate). Unlike a 404, this is fatal for the whole run, not one
    request: every subsequent call fails the same way, so callers must stop and alert
    rather than skip — swallowing it burns the scan queue collecting nothing."""


class RateLimited(BrawlStarsError):
    """429 — retried with backoff."""


class ServerError(BrawlStarsError):
    """5xx — retried with backoff."""


class RateLimiter:
    """At most ``rate_per_sec`` requests/second (enforces a minimum gap)."""

    def __init__(self, rate_per_sec: float):
        self._min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            delay = self._last + self._min_interval - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = loop.time()


class BrawlStarsClient:
    """Async API client. Use as an async context manager."""

    def __init__(self, token: Optional[str] = None, rate_per_sec: Optional[float] = None,
                 rate_limiter: Optional[RateLimiter] = None):
        self._token = token or settings.brawlstars_api_token
        if not self._token:
            raise RuntimeError(
                "No API token. Set BRAWLSTARS_API_TOKEN in .env "
                "(create a key at https://developer.brawlstars.com)."
            )
        self._limiter = rate_limiter or RateLimiter(rate_per_sec or settings.crawl_rate_limit_per_sec)
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(20.0),
        )

    async def __aenter__(self) -> "BrawlStarsClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, RateLimited, ServerError)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _get(self, path: str) -> Any:
        await self._limiter.wait()
        resp = await self._client.get(path)
        code = resp.status_code
        if code == 200:
            return resp.json()
        if code == 429:
            raise RateLimited(code, "request throttling limits exceeded")
        if 500 <= code < 600:
            raise ServerError(code, "server error")
        if code == 404:
            raise BrawlStarsError(code, f"not found: {path}")
        if code in (401, 403):
            try:
                reason = resp.json().get("reason", "")
            except ValueError:
                reason = ""
            raise AuthError(
                code,
                f"{reason or 'auth/IP error'} — check the token and that this machine's "
                "current public IP is on the key's allow-list "
                "(https://developer.brawlstars.com)",
            )
        raise BrawlStarsError(code, resp.text[:200])

    # --- Endpoints ---
    async def get_player(self, tag: str) -> dict:
        return await self._get(f"/players/{encode_tag(tag)}")

    async def get_battlelog(self, tag: str) -> list:
        data = await self._get(f"/players/{encode_tag(tag)}/battlelog")
        return data.get("items", [])

    async def get_top_players(self, country: str = "global", limit: int = 200) -> list:
        data = await self._get(f"/rankings/{country}/players?limit={limit}")
        return data.get("items", [])

    async def get_top_players_for_brawler(
        self, brawler_id: int, country: str = "global", limit: int = 200
    ) -> list:
        data = await self._get(f"/rankings/{country}/brawlers/{brawler_id}?limit={limit}")
        return data.get("items", [])

    async def get_brawlers(self) -> list:
        data = await self._get("/brawlers")
        return data.get("items", [])


class LiveProfileUnavailable(RuntimeError):
    """Safe public error: never includes credentials, upstream body, or request URLs."""


class LiveProfiles:
    """Bound live roster/rank lookups together, with cache and per-tag single-flight.

    One event-loop limiter is shared by all cache misses. At most ``max_inflight`` profile
    fetches can be pending, including time spent waiting for the rate limiter. The overall
    deadline covers retry delays too. A brief auth circuit avoids retrying every tag during
    the home host's recurring IP-lock outages. Each production worker has its own instance.
    """
    def __init__(self, *, deadline: float = 8.0, failure_ttl: float = 10.0,
                 max_inflight: int = 8, max_cache: int = 256):
        self.deadline = deadline
        self.failure_ttl = failure_ttl
        self.max_inflight = max_inflight
        self.max_cache = max_cache
        self._cache: dict = {}
        self._states = weakref.WeakKeyDictionary()
        self._auth_retry_at = 0.0

    def clear(self) -> None:
        self._cache.clear()
        self._auth_retry_at = 0.0

    async def get(self, tag: str, *, client_factory=None) -> dict:
        key = normalize_tag(tag)
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit is not None and hit[0] > now:
            if hit[1] is not None:
                return hit[1]
            raise LiveProfileUnavailable("Live player lookup is temporarily unavailable. Try again shortly.")
        if now < self._auth_retry_at:
            raise LiveProfileUnavailable("Live player lookup is temporarily unavailable. Try again shortly.")
        loop = asyncio.get_running_loop()
        state = self._states.setdefault(loop, {
            "limiter": RateLimiter(settings.crawl_rate_limit_per_sec), "inflight": {}})
        task = state["inflight"].get(key)
        if task is None:
            if len(state["inflight"]) >= self.max_inflight:
                raise LiveProfileUnavailable("Live player lookup is busy. Try again shortly.")
            task = loop.create_task(self._fetch(key, state, client_factory or BrawlStarsClient))
            state["inflight"][key] = task
        player = await asyncio.shield(task)
        if player is None:
            raise LiveProfileUnavailable("Live player lookup is temporarily unavailable. Try again shortly.")
        return player

    async def _fetch(self, key, state, client_factory):
        async def request():
            async with client_factory(rate_limiter=state["limiter"]) as client:
                player = await client.get_player(key)
                if not isinstance(player, dict):
                    raise ValueError("invalid player profile")
                return player
        try:
            player = await asyncio.wait_for(request(), timeout=self.deadline)
            ttl = settings.roster_ttl_seconds
        except Exception as exc:
            player, ttl = None, self.failure_ttl
            if isinstance(exc, AuthError):
                self._auth_retry_at = time.monotonic() + self.failure_ttl
            logger.warning("live profile lookup unavailable (%s)", type(exc).__name__)
        finally:
            state["inflight"].pop(key, None)
        now = time.monotonic()
        if len(self._cache) >= self.max_cache:
            for old in [tag for tag, value in self._cache.items() if value[0] <= now]:
                self._cache.pop(old, None)
            if len(self._cache) >= self.max_cache:
                self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (now + ttl, player)
        return player
