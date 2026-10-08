"""FastAPI app exposing the draft engine.

    PYTHONPATH=backend uvicorn bsdraft.api.main:app --reload --port 8000

Loads the engine (empirical stats + trained model) at startup. When DATA_URL / MODEL_URL are
set, it also syncs the published dataset and model every REFRESH_SECONDS — rebuilding stats
and hot-swapping the model — so the live site stays current with no restart. Loads the
player's roster (mastery personalization) if PLAYER_TAG is set — a local-only feature (needs
the IP-locked key).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bsdraft.api import schemas as S
from bsdraft.collect.client import BrawlStarsClient, LiveProfiles, normalize_tag
from bsdraft.config import settings
from bsdraft.constants import RANKED_MODES
from bsdraft.data.balance_eras import current_balance_era
from bsdraft.data import reference as R
from bsdraft.data.ranked_maps import select_current_ranked_maps
from bsdraft.data import sync
from bsdraft.data.dataset import count_matches
from bsdraft.engine import mastery
from bsdraft.engine.drift import detect_drift, load_report
from bsdraft.engine.engine import DraftEngine
from bsdraft.engine.loadout import loadout_advice
from bsdraft.engine import purchases as purchases_mod
from bsdraft.engine.personal import build_personal_stats, matches_from_battlelog
from bsdraft.engine.readiness import Fielded
from bsdraft.engine.state import DraftState
from bsdraft.engine.playerrank import build_rank_index, current_ranked_tier
from bsdraft.engine.rank_store import RankIndex, load_rank_index
from bsdraft.engine.stats import DraftStats, build_bracketed
from bsdraft.engine.stats_store import load_stats
from bsdraft.engine.tiers import BRACKETS, bracket_of_tier, is_mythic_plus, min_power_for_bracket, tier_label
from bsdraft.models.serve import WinProbModel
from bsdraft.models import bundles

logger = logging.getLogger("bsdraft.api")

_engine: Optional[DraftEngine] = None
_dataset_count: int = 0    # total matches in the synced dataset (headline count; recomputed on sync change)
_stats_source: str = ""    # "artifact" (current-era stats loaded) or "rebuild" (capped fallback)
_last_check: float = 0.0   # epoch of the last sync attempt (liveness)
_last_change: float = 0.0  # epoch of the last actual data change
_meta_cache = None         # (data_version, MetaReport); recomputed lazily when data changes
_rank_idx_cache = None     # (token, RankIndex); recomputed lazily when the data/artifact changes
_rank_version = 0          # bumps when the synced rank-index artifact changes (invalidates the cache)
_personal_cache: dict = {} # tag -> (data_version, PersonalStats|None); rebuilt when data changes
_personal_locks: dict = {} # tag -> Lock; single-flights the per-tag dataset scan (no stampede)
_personal_locks_guard = threading.Lock()
_roster_cache: dict = {}   # normalized tag -> (fetched_at, RosterResponse); short TTL spares the live API
_profiles = LiveProfiles()  # one cache/limiter for both live roster and rank
_ROSTER_CACHE_MAX = 256    # hard bound so distinct tags can't grow the cache without limit

_rank_cache: dict = {}     # normalized tag -> (fetched_at, RankResponse); short TTL on live rank lookups


def _load_model():
    bundled = bundles.load_current() if settings.model_manifest_url else None
    if bundled is not None:
        return bundled
    try:
        return WinProbModel(validate_current_era=True)
    except Exception as exc:
        logger.warning("local model rejected (%s); using empirical stats", type(exc).__name__)
        # An absent path deliberately gives the neutral unavailable model.
        return WinProbModel(sync.MODEL_PATH.parent / ".unavailable-model")


def _sync_model():
    if settings.model_manifest_url:
        changed = bundles.refresh(settings.model_manifest_url)
        if not bundles.status().get("missing") or bundles.load_current() is not None:
            return changed
    return sync.sync_model(settings.model_url) if settings.model_url else False


def _build_stats():
    """Produce ``(global_stats, {bracket: stats})``. When STATS_URL is set the API **loads** the
    precomputed current-era stats artifact (built off-box, ~tens of MB, no OOM); otherwise it
    **rebuilds** them from the synced matches, capped to STATS_MAX_MATCHES within the current era.
    Which path won is surfaced on /api/health as ``stats_source`` — the fallback is a graceful
    degradation (a current-era window instead of the full archive)."""
    global _stats_source
    era = current_balance_era()
    if settings.stats_url and sync.STATS_PATH.exists():
        try:
            result = load_stats(sync.STATS_PATH)
            if era and (
                getattr(result[0], "analysis_start_ts", 0) != era.start_ts
                or getattr(result[0], "analysis_era_id", "") != era.id
            ):
                raise ValueError(
                    f"stats artifact is for era {getattr(result[0], 'analysis_era_id', '') or 'legacy'}"
                    f" / {getattr(result[0], 'analysis_start_ts', 0)}, expected {era.id} / {era.start_ts}"
                )
            _stats_source = "artifact"
            return result
        except Exception as e:  # noqa: BLE001 — a corrupt/old artifact must not break startup
            logger.warning("stats load failed (%s); rebuilding from matches", e)
    _stats_source = "rebuild"
    return build_bracketed(
        halflife_days=settings.stats_halflife_days,
        max_matches=settings.stats_max_matches,
        analysis_start_ts=era.start_ts if era else 0,
        analysis_era_id=era.id if era else "",
    )


def _rank_index() -> RankIndex:
    """Cached ``tag -> tier`` lookup.

    **When RANK_INDEX_URL is set (the deployed API) this only ever LOADS the published artifact,
    and serves an EMPTY index if that fails.** It deliberately does not fall back to building from
    the matches there: that build is a ~200 MB transient at 3M tags, which OOM-kills a 512 MB box
    and takes the whole site down for everyone — strictly worse than ranks reading as unknown until
    the next sync. The artifact exists precisely to avoid that build, so "artifact unavailable"
    must never route back into it. Only a host with no artifact URL configured (home machine /
    local dev, which has the RAM) builds in memory.

    Cached under ``(use_artifact, _rank_version)``, so an empty index sticks until
    ``sync_rank_index`` reports a change rather than retrying a failing load on every request —
    a persistently broken artifact stays quietly degraded, which is intended."""
    global _rank_idx_cache
    use_artifact = bool(settings.rank_index_url)
    token = (use_artifact, _rank_version if use_artifact else _last_change)
    if _rank_idx_cache is None or _rank_idx_cache[0] != token:
        if use_artifact:
            try:
                idx = load_rank_index(sync.RANK_INDEX_PATH)
            except FileNotFoundError:
                logger.error("rank index artifact missing at %s — serving an EMPTY index "
                             "(ranks unknown) until the next sync", sync.RANK_INDEX_PATH)
                idx = RankIndex.empty()
            except Exception as e:  # noqa: BLE001 — a corrupt/old artifact degrades, never 500s
                logger.error("rank index load failed (%s) — serving an EMPTY index (ranks unknown) "
                             "until the next sync", e)
                idx = RankIndex.empty()
        else:
            idx = RankIndex.from_mapping(build_rank_index())
        _rank_idx_cache = (token, idx)
    return _rank_idx_cache[1]


# Supercell tags draw from a fixed 14-character alphabet. Anything outside it (or absurdly
# short/long) cannot name a player in our data, so a dataset scan for it is pure waste — and
# SHORT garbage is the worst case: the scan's raw-substring prefilter skips ~99.9% of lines for
# a real tag, but a 1-2 char string appears in nearly every line, degenerating the scan to
# json-parsing all ~1.6M matches. Both personal-stats entry points gate on this.
_TAG_CHARS = frozenset("0289PYLQGRJCUV")


def _plausible_tag(t: str) -> bool:
    return 3 <= len(t) <= 14 and set(t) <= _TAG_CHARS


def _personal_for(tag: Optional[str]):
    """Return recent cached history immediately; build misses in the bounded background pool.

    Roster readiness and ownership remain available while history warms. An occupied pool
    never turns a new visitor into another full-file scan or a waiter on one.
    """
    t = normalize_tag(tag or "")
    if not t or not _plausible_tag(t) or _engine is None:
        return None
    hit = _personal_cache.get(t)
    if hit is not None and hit[0] == _last_change:
        return hit[1]
    _warm_personal(t)
    if hit is not None and _last_change - hit[0] <= _STALE_LAG_MAX_SECONDS:
        return hit[1]
    return None


def _rebuild_personal(t: str, extra_matches=None):
    """Build at most two histories globally, and never wait behind another tag/build.

    This bound also covers direct callers (including startup), rather than only /api/warm.
    Per-tag locks remain single-flight; callers joining a running scan return without blocking.
    """
    if not _personal_build_slots.acquire(blocking=False):
        return None
    try:
        with _personal_locks_guard:
            if len(_personal_locks) > 512:
                for k in [k for k, v in _personal_locks.items() if not v.locked()]:
                    del _personal_locks[k]
            lock = _personal_locks.setdefault(t, threading.Lock())
            if not lock.acquire(blocking=False):
                return None
        try:
            hit = _personal_cache.get(t)
            if hit is not None and hit[0] == _last_change:
                return hit[1]
            if len(_personal_cache) >= 256:
                _personal_cache.pop(next(iter(list(_personal_cache)), ""), None)
            version = _last_change
            ps = build_personal_stats(
                t, fallback=_engine.stats, extra_matches=extra_matches,
                analysis_start_ts=getattr(_engine.stats, "analysis_start_ts", 0),
            )
            _personal_cache[t] = (version, ps)
            return ps
        finally:
            lock.release()
    finally:
        _personal_build_slots.release()


# Cap how many personal-stats warms run at once. Each warm streams the full ``matches.jsonl`` to
# filter one tag's games (seconds on the cloud dataset, on the free tier's CPU sliver), so an
# unbounded thread-per-LOAD would let a burst of distinct tags — many drafters, or a bot hitting
# /api/rank — spawn a scan pile-up that starves the box and can time out the health check. Warming
# is best-effort (later best-effort warms retry a miss), so a full pool skips rather than queues.
# 2 keeps a normal LOAD warm without saturating the instance; tune if needed.
_WARM_MAX_CONCURRENCY = 2
_warm_slots = threading.BoundedSemaphore(_WARM_MAX_CONCURRENCY)
_personal_build_slots = threading.BoundedSemaphore(_WARM_MAX_CONCURRENCY)
_warm_inflight: set = set()  # tags with a spawned warm worker; guarded by _personal_locks_guard

# A recent cached history can serve during revalidation. Older history is omitted until the
# bounded background rebuild completes, so pressure cannot force an unbounded foreground scan.
_STALE_LAG_MAX_SECONDS = 3600.0


def _warm_personal(tag: Optional[str]) -> None:
    """Warm history without blocking advice. Duplicate tags and a full pool no-op.

    /api/rank and /api/warm can start this before picks. Cold/stale pick reads also request
    a warm; ownership and readiness still work while the history signal is unavailable.
    """
    t = normalize_tag(tag or "")
    if (not t or not _plausible_tag(t) or _engine is None
            or _personal_cache.get(t, (None,))[0] == _last_change):
        return
    # A build already in flight for this tag must not cost a second pool slot: the joining
    # worker would just park on the per-tag lock while HOLDING it — two near-simultaneous
    # triggers for one tag (multi-tab picks, boot + map pings) could pin the whole pool doing
    # no work. The in-flight check has to be ATOMIC with taking the slot: `lock.locked()`
    # alone races the window between Thread.start() and the worker actually acquiring the
    # lock (reproduced at ~15-30% with two concurrent stale reads, 2026-08-25), so a spawned-
    # worker registry (_warm_inflight) is written under the same guard that checks it. The
    # locked() check also catches direct/startup builds, which the registry does not see.
    with _personal_locks_guard:
        if t in _warm_inflight:
            return
        lock = _personal_locks.get(t)
        if lock is not None and lock.locked():
            return
        if not _warm_slots.acquire(blocking=False):
            return  # warm pool full — skip; a later request can retry warming
        _warm_inflight.add(t)

    def _run() -> None:
        try:
            # This worker owns a warm slot; the shared build bound also covers direct callers.
            _rebuild_personal(t)
        except Exception as exc:
            logger.warning("personal history warm failed (%s)", type(exc).__name__)
        finally:
            with _personal_locks_guard:
                _warm_inflight.discard(t)
            _warm_slots.release()

    try:
        threading.Thread(target=_run, daemon=True, name="warm-personal").start()
    except RuntimeError:      # thread table exhausted — release the slot and skip, don't 500 the LOAD
        with _personal_locks_guard:
            _warm_inflight.discard(t)
        _warm_slots.release()


async def _refresh_loop() -> None:
    """Periodically re-sync the dataset (and model) and hot-swap rebuilt stats / a reloaded
    model into the live engine, so a fresh crawl or retrain rolls out with no restart."""
    global _last_check, _last_change, _rank_version, _dataset_count
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(settings.refresh_seconds)
        try:
            data_changed = (await loop.run_in_executor(None, sync.sync_matches, settings.data_url)
                            if settings.data_url else False)
            _last_check = time.time()
            if data_changed and _engine is not None:
                _last_change = time.time()  # invalidate the rank / meta / personal caches
                _dataset_count = await loop.run_in_executor(None, count_matches)  # refresh headline count
            # Refresh the published meta-drift report (a few KB) — /api/meta reads the file per
            # request, so a changed artifact is picked up with no eager work here.
            if settings.meta_report_url:
                if await loop.run_in_executor(None, sync.sync_meta_report, settings.meta_report_url):
                    logger.info("meta report artifact updated")
            # Refresh the precomputed rank-index artifact (loaded, not rebuilt). A change just bumps
            # the version so the lazy _rank_index() reloads on the next /api/rank — no eager work.
            if settings.rank_index_url and _engine is not None:
                if await loop.run_in_executor(None, sync.sync_rank_index, settings.rank_index_url):
                    _rank_version += 1
                    logger.info("rank index artifact updated")
            # Refresh the empirical stats from their source: the published artifact (STATS_URL,
            # loaded — no in-memory rebuild) or, failing that, a local rebuild from the matches.
            stats_changed = (await loop.run_in_executor(None, sync.sync_stats, settings.stats_url)
                             if settings.stats_url and _engine is not None else False)
            if _engine is not None and (stats_changed or (data_changed and
                    (not settings.stats_url or _stats_source == "rebuild"))):
                g, br = await loop.run_in_executor(None, _build_stats)
                _engine.stats, _engine.bracket_stats = g, br
                logger.info("draft stats reloaded: %d matches, %d bracket(s)", g.n, len(br))
            if (settings.model_url or settings.model_manifest_url) and _engine is not None:
                if await loop.run_in_executor(None, _sync_model):
                    _engine.model = await loop.run_in_executor(None, _load_model)
                    logger.info("win-prob model hot-swapped (available=%s)", _engine.model.available)
            # Item win-rate table: just refresh the file — the loadout loader reloads it on mtime
            # change per request, so there's no engine object to hot-swap.
            if settings.itemstats_url:
                if await loop.run_in_executor(None, sync.sync_itemstats, settings.itemstats_url):
                    logger.info("item win-rate table updated")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — a refresh hiccup must not kill the loop
            logger.warning("refresh loop error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _last_check, _last_change, _dataset_count
    loop = asyncio.get_running_loop()
    if settings.data_url:
        await loop.run_in_executor(None, sync.sync_matches, settings.data_url)
        _last_check = _last_change = time.time()
    if settings.model_url or settings.model_manifest_url:
        await loop.run_in_executor(None, _sync_model)
    if settings.stats_url:
        await loop.run_in_executor(None, sync.sync_stats, settings.stats_url)
    if settings.rank_index_url:
        await loop.run_in_executor(None, sync.sync_rank_index, settings.rank_index_url)
    if settings.meta_report_url:
        await loop.run_in_executor(None, sync.sync_meta_report, settings.meta_report_url)
    if settings.itemstats_url:
        await loop.run_in_executor(None, sync.sync_itemstats, settings.itemstats_url)
    g, br = _build_stats()
    _engine = DraftEngine(g, _load_model(), bracket_stats=br)
    _dataset_count = await loop.run_in_executor(None, count_matches)  # headline count over the full dataset
    if settings.player_tag:
        ptag = normalize_tag(settings.player_tag)
        try:
            async with BrawlStarsClient() as client:
                _engine.roster, _engine.roster_name = await mastery.fetch_roster(client, settings.player_tag)
                # Prime personal stats with the player's freshest games (needs the live key,
                # so local/home only); the public host falls back to dataset-derived stats.
                try:
                    extra = matches_from_battlelog(await client.get_battlelog(ptag), ptag)
                    await loop.run_in_executor(None, _rebuild_personal, ptag, extra)
                except Exception:
                    pass
        except Exception:
            _engine.roster, _engine.roster_name = None, ""
    task = None
    if (settings.data_url or settings.model_url or settings.model_manifest_url or settings.stats_url
            or settings.rank_index_url or settings.meta_report_url or settings.itemstats_url
            ) and settings.refresh_seconds > 0:
        task = asyncio.create_task(_refresh_loop())
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Brawl Stars Draft Tool", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    era = current_balance_era()
    ready = bool(_engine and _engine.stats.n > 0)
    newest_ts = 0
    if settings.meta_report_url and sync.META_REPORT_PATH.exists():
        try:
            newest_ts = load_report(sync.META_REPORT_PATH).newest_ts
        except Exception:
            pass
    age = max(0, int(time.time()) - newest_ts) if newest_ts else None
    artifacts = sync.sync_status()
    bundle_status = bundles.status() if settings.model_manifest_url else {}
    if bundle_status:
        artifacts["model bundle"] = bundle_status
    degraded = any(value.get("error") for value in artifacts.values())
    payload = {
        "status": ("degraded" if degraded else "ok") if ready else "unavailable",
        "ready": ready,
        "model": bool(_engine and _engine.model and _engine.model.available),
        "model_identity": {k: v for k, v in model_status().items() if k in ("sha256", "release_id")},
        "model_era": getattr(getattr(_engine, "model", None), "analysis_era_id", ""),
        "matches": _dataset_count or (_engine.stats.n if _engine else 0),
        "stats_source": _stats_source,
        "stats_n": _engine.stats.n if _engine else 0,
        "stats_era": getattr(_engine.stats, "analysis_era_id", "") if _engine else "",
        "stats_start_ts": getattr(_engine.stats, "analysis_start_ts", 0) if _engine else 0,
        "current_era": era.id if era else "",
        "current_era_start_ts": era.start_ts if era else 0,
        "newest_match_ts": newest_ts or None,
        "data_age_seconds": age,
        "data_stale": age > 6 * 3600 if age is not None else None,
        "artifacts": artifacts,
        "roster": bool(_engine and _engine.roster),
        "refresh_seconds": settings.refresh_seconds if settings.data_url else 0,
        "last_check": _last_check or None,
        "last_change": _last_change or None,
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/api/model")
def model_status():
    """Statistics for the exact loaded model; legacy artifacts have no verified evaluation."""
    model = _engine.model if _engine else None
    info = getattr(model, "release_info", {})
    identity = sync.artifact_identity(sync.MODEL_PATH, sync._MODEL_SHA_PATH) if not info else {}
    return {"available": bool(model and model.available),
            "analysis_era_id": getattr(model, "analysis_era_id", ""),
            "release_id": info.get("release_id"), "sha256": info.get("sha256", identity.get("sha256")),
            "published_at": info.get("published_at"), "metrics": info.get("metrics"),
            "evaluation_status": "verified_bundle" if info else "unavailable",
            "note": None if info else "No evaluation report verified against the served weights. Historical reports are not current model metrics."}



@app.get("/api/meta", response_model=S.MetaResponse)
def meta():
    """Has the meta shifted (balance change / new brawler) recently? Served from the published
    meta-report artifact when META_REPORT_URL is set (the home crawler computes it each cycle) —
    computing here streams the full dataset twice, minutes per data change on the free tier's
    CPU sliver, so the local compute (cached per data version) is only the fallback."""
    global _meta_cache
    rep = None
    if settings.meta_report_url and sync.META_REPORT_PATH.exists():
        try:
            rep = load_report(sync.META_REPORT_PATH)
        except Exception as e:  # noqa: BLE001 — a corrupt/old artifact must fall back, not 500
            logger.warning("meta report load failed (%s)", type(e).__name__)
    if rep is None and settings.meta_report_url:
        return S.MetaResponse(shifted=False, n_recent=0, n_prior=0,
                              note="Meta report temporarily unavailable; current draft statistics remain active.")
    if rep is None:
        if _meta_cache is None or _meta_cache[0] != _last_change:
            _meta_cache = (_last_change, detect_drift())
        rep = _meta_cache[1]
    names = {b.id: b.name for b in R.load_brawlers()}
    return S.MetaResponse(
        shifted=rep.shifted, n_recent=rep.n_recent, n_prior=rep.n_prior,
        new_brawlers=[names.get(b, str(b)) for b in rep.new_brawlers],
        shifts=[
            S.MetaShift(
                brawler_id=s.brawler_id, name=s.name, kind=s.kind,
                wr_before=round(s.wr_before, 4), wr_after=round(s.wr_after, 4),
                use_before=round(s.use_before, 4), use_after=round(s.use_after, 4),
                z=round(s.z, 2),
            )
            for s in rep.shifts
        ],
        note=rep.note,
    )


@app.get("/api/reference", response_model=S.ReferenceResponse)
def reference():
    # pickable_brawlers() (not load_brawlers()) so the grid, its count, and any client-built
    # candidate list drop unreleased catalog entries — same "filter at the boundary, not in the
    # vocab-building loader" reasoning applied to maps just below.
    brawlers = [
        S.BrawlerRef(id=b.id, name=b.name, cls=b.cls, rarity=b.rarity, image_url=b.image_url)
        for b in R.pickable_brawlers()
    ]
    # `load_ranked_maps()` is every catalog map in a ranked mode (~440, including ones upstream
    # flags `disabled` — that flag has hidden live Ranked maps three times). Ranked only rotates a
    # handful per mode per season, so showing the catalog offers map/mode pairs nobody can queue
    # (e.g. "Heist: Pit Stop"), and the model has nothing to say about them anyway. Collected
    # ranked games are the only rotation signal we have — a map with none is one we have never
    # seen played. Falls back to the upstream-enabled maps when stats aren't loaded yet, so a
    # cold start shows too much rather than nothing.
    #
    # Deliberately filtered *here* and not in `load_ranked_maps()`: that function is also the
    # model's map vocabulary, which should hold every map that could ever carry training games.
    #
    # Two signals, per mode. The recent window (`map_games_recent`, last RECENT_WINDOW_DAYS of
    # battle time) is the primary one: a map added mid-season starts filling it immediately and
    # crosses the cut after a few hours of crawl — same-day, where the cumulative cut needed
    # ~2-3 days (2026-08-25: Brawl Ball gained Beach Ball + Pinhole Punt at ~18:00 UTC) — and a
    # dropped map drains out of it within the window's ~3 days instead of decaying below the
    # cumulative threshold over ~8 weeks. The cumulative share cut is the fallback for when the
    # recent signal is absent (pre-2026-08-25 artifact) or the mode's window is too thin to
    # trust (post-outage refill, or a crawl slowdown — see RECENT_TRUST_MIN_LEADER; a thin window
    # can still admit a clearly-played map and prune a clearly-absent one, see
    # RECENT_PRUNE_MIN_MODE_GAMES in `data/ranked_maps.py`): the cumulative stats span more
    # history than one rotation, so "any games at all" would readmit retirees' decaying residue —
    # the separation there is per-mode and enormous (2026-08-20: Heist ran four maps at
    # 1954-2026 games with retired Pit Stop on 90). Both cuts are shares of the mode's leader,
    # not absolute counts, so the thresholds ride the crawl's volume instead of needing a retune
    # every time the dataset grows.
    stats = _engine.stats if _engine else None
    played = select_current_ranked_maps(R.load_ranked_maps(), stats)
    maps = [
        S.MapRef(id=m.id, name=m.name, mode=m.mode, image_url=m.image_url,
                 games=int(_engine.stats.map_games.get(m.id, 0)) if _engine else 0)
        for m in played
    ]
    brackets = [b for b in BRACKETS if _engine and b in _engine.bracket_stats]
    # Same free/"boosted" set the recommender folds into a roster (hand-maintained list ∪
    # data-derived), so the client's grid (placeable + "free" badge) and the picks agree on what
    # is free — including unannounced mid-season grants the release notes miss.
    pickable_ids = {b.id for b in brawlers}
    seasonal_boosted = [bid for bid in R.load_ranked_rotation() if bid in pickable_ids]
    return S.ReferenceResponse(brawlers=brawlers, maps=maps, modes=list(RANKED_MODES),
                               brackets=brackets, boosted=list(_free_brawler_ids()),
                               seasonal_boosted=seasonal_boosted)


def _parse_id_csv(raw: Optional[str], cap: int = 5) -> List[int]:
    """Defensive CSV-of-ints parser for the loadout ``enemies`` param: junk tokens are skipped and
    the list is capped — never a 4xx, because the hover popover must degrade quietly."""
    out: List[int] = []
    for tok in (raw or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out[:cap]


@app.get("/api/loadout", response_model=S.LoadoutResponse)
def loadout(brawler: int, mode: str, map_id: Optional[int] = None, enemies: Optional[str] = None):
    """Which gadget / star power / gear to equip on a drafted brawler, given the mode. Effect-based
    heuristic (see :mod:`bsdraft.engine.loadout`) — the client overlays the user's owned items on
    their own pick. ``enemies`` (CSV of the queried brawler's opponents' ids, seat-flip resolved by
    the client) turns on the comp-aware overlay; optional, so old clients keep byte-identical
    comp-blind behavior and both deploy-skew directions are silent no-ops. Returns an empty (but
    well-formed) body for an unknown brawler so the hover popover degrades quietly rather than
    erroring."""
    adv = loadout_advice(brawler, mode, map_id, enemies=_parse_id_csv(enemies))
    if adv is None:
        return S.LoadoutResponse(brawler_id=brawler, brawler_name="", mode=mode)
    return S.LoadoutResponse(**adv)


class _LiveProfileAdapter:
    async def get_player(self, tag):
        return await _profiles.get(tag, client_factory=BrawlStarsClient)


@app.get("/api/roster", response_model=S.RosterResponse)
async def roster(tag: Optional[str] = None):
    """The given player's roster — owned brawlers, loadout completeness, and mastery — fetched
    live from Supercell (needs the IP-locked key, so local/home only). The frontend re-polls this
    so a long session stays current; a successful result is cached for ``roster_ttl_seconds`` so
    the polling doesn't hammer the live API.

    No ``settings.player_tag`` fallback: this endpoint is public via the roster tunnel, so a
    tag-less request must NOT resolve to the operator's own account — that leaked the operator's
    identity and roster to every visitor who hadn't entered their own tag."""
    t = (tag or "").strip()
    if not t:
        return S.RosterResponse(loaded=False, tag="", name="", error="no player tag")
    key = normalize_tag(t)
    hit = _roster_cache.get(key)
    if hit is not None and (time.time() - hit[0]) < settings.roster_ttl_seconds:
        return hit[1]
    try:
        r, name = await mastery.fetch_roster(_LiveProfileAdapter(), t)
        # Deliberately do NOT write r onto _engine. This is a read endpoint, and _engine is one
        # process-global DraftEngine shared by every request: persisting the fetched roster let
        # whichever tag last hit /api/roster become the engine default that _roster_for() folds
        # into /api/recommend — cross-visitor contamination the moment this host serves more
        # than one person. The roster is returned per-request (and cached by tag below);
        # personalization takes it explicitly via RecommendRequest.roster.
        owned = [
            S.OwnedBrawler(
                id=bid, mastery=round(m.score, 3), gaps=m.gaps(),
                owned_star_powers=list(m.owned_star_powers),
                owned_gadgets=list(m.owned_gadgets),
                owned_gears=[S.OwnedGear(**g) for g in m.owned_gears],
                # Progression state the purchase advisor needs (already parsed by Mastery).
                power=m.power, has_hypercharge=m.has_hypercharge, buffies=m.buffies,
            )
            for bid, m in r.items()
        ]
        resp = S.RosterResponse(loaded=True, tag=t, name=name, owned=owned)
        # Cache only successful loads; errors retry next poll. Evict stale/oldest entries so a
        # stream of distinct tags can't grow this unbounded over the process lifetime (the TTL
        # alone never removes anything — it's only checked on read).
        now = time.time()
        if len(_roster_cache) >= _ROSTER_CACHE_MAX:
            for k in [k for k, (ts, _) in _roster_cache.items()
                      if (now - ts) >= settings.roster_ttl_seconds]:
                del _roster_cache[k]
            if len(_roster_cache) >= _ROSTER_CACHE_MAX:  # all still fresh — drop the oldest
                del _roster_cache[min(_roster_cache, key=lambda k: _roster_cache[k][0])]
        _roster_cache[key] = (now, resp)
        return resp
    except Exception as e:  # noqa: BLE001
        return S.RosterResponse(loaded=False, tag=t, name="",
                                error="Live roster lookup is temporarily unavailable. Try again shortly.")


@app.post("/api/purchases", response_model=S.PurchasesResponse)
def purchases(req: S.PurchaseRequest):
    """Rank a player's most efficient next purchases (power climbs, gadgets, star powers, gears,
    hypercharges, new-brawler unlocks) from their ownership snapshot. Like /api/recommend,
    the client sends the roster it fetched from the keyed tunnel — the public host can't fetch it
    itself. Scored by win-rate value per coin-equivalent with every prerequisite (power climb to
    the item gate and to the bracket's Ranked power floor, a core build, the unlock) priced into
    the package; balances stay unknowable. See :mod:`bsdraft.engine.purchases`."""
    owned = {
        e.id: purchases_mod.OwnedState(
            power=e.power,
            star_powers=frozenset(e.owned_star_powers),
            gadgets=frozenset(e.owned_gadgets),
            gears=frozenset(purchases_mod._norm(g.name) for g in e.owned_gears),
            has_hypercharge=e.has_hypercharge,
        )
        for e in req.roster
    }
    bracket = req.rank_bracket if req.rank_bracket in BRACKETS else None
    floor = purchases_mod.resolve_floor(bracket, req.power_floor, R.load_economy())
    recs = _engine.recommend_purchases(owned, top=max(0, min(req.top, 200)), rank_bracket=bracket,
                                       power_floor=floor, min_per_kind=max(0, min(req.min_per_kind, 5)))
    return S.PurchasesResponse(
        tag=req.tag or "", name=req.name or "", scope="ranked",
        rank_bracket=bracket, power_floor=floor,
        recommendations=[S.PurchaseRec(**r) for r in recs],
    )


async def _live_rank(tag_n: str) -> Tuple[str, Optional[S.RankResponse]]:
    """Current Ranked tier from a live profile fetch (needs the IP-locked key, so it only
    works local/home or via the keyed tunnel). Reads the profile's ``rankedRank`` — the tier
    the player is at *now* — rather than the battle log, whose per-game tier over-states anyone
    who lost a promotion game (see :func:`current_ranked_tier`). Cached briefly
    (``roster_ttl_seconds``) so the frontend re-polling the same tag spares the live API.

    Returns ``(status, response)`` rather than an Optional, because the caller must tell three
    outcomes apart and only one of them justifies trusting the dataset:

    * ``"ok"`` — a tier came back; serve it.
    * ``"unplaced"`` — the profile loaded fine and carries no ``rankedRank``: the player has not
      placed this season. Their dataset row is from *before* the reset, so falling back to it
      would report a tier they no longer hold — the exact over-statement this whole live-first
      path exists to avoid.
    * ``"unavailable"`` — the lookup could not be served at all (no key on this host, an IP-lock
      403, a network blip). Nothing was learned about the player, so the dataset is the best
      guess we have — flagged stale, since it may also predate a reset.
    """
    hit = _rank_cache.get(tag_n)
    if hit is not None and (time.time() - hit[0]) < settings.roster_ttl_seconds:
        return hit[1]
    try:
        player = await _profiles.get(tag_n, client_factory=BrawlStarsClient)
        t = current_ranked_tier(player)
    except Exception:  # noqa: BLE001 — keyless/offline host, IP-lock 403, or API hiccup
        return ("unavailable", None)
    if not t:
        # A successful fetch that says "no tier this season" is real information, not a miss.
        resp = S.RankResponse(found=False, tag=tag_n, source="live",
                              error="no Ranked games yet this season — place a few to set a tier")
        out = ("unplaced", resp)
    else:
        resp = S.RankResponse(found=True, tag=tag_n, tier=t, tier_label=tier_label(t),
                              bracket=bracket_of_tier(t), source="live")
        out = ("ok", resp)
    if len(_rank_cache) > 512:   # bound growth — one entry per unique tag, TTL alone never frees it
        _rank_cache.clear()
    _rank_cache[tag_n] = (time.time(), out)
    return out


@app.get("/api/rank", response_model=S.RankResponse)
async def rank(tag: str):
    """Resolve a player's current Ranked tier. We try a live profile fetch first whenever a
    key is configured (local/home, or the keyed roster tunnel), because its ``rankedRank`` is the
    player's tier *right now* — the collected match data is a crawl snapshot that goes stale across
    a Ranked season reset, where a player can drop several tiers, so a pre-reset row over-states
    them. The dataset is the fallback: it needs no key (the only source on the public host) and
    covers players with no recent ranked games."""
    tag_n = normalize_tag(tag)
    if not tag_n:
        return S.RankResponse(found=False, tag="", error="enter a player tag")
    live_tried = False
    if settings.brawlstars_api_token:
        live_tried = True
        status, live = await _live_rank(tag_n)
        # "ok" and "unplaced" are both answers about *this* season — return them as-is. Only
        # "unavailable" (no answer at all) may fall through to the pre-reset crawl snapshot.
        if status in ("ok", "unplaced"):
            # The player just entered their tag (LOAD) — warm this tag's personal stats now, in the
            # background, so a personalized seat's first pick doesn't block on the scan later (see
            # _warm_personal). Only helps when this host also serves /api/recommend (local / the
            # home stack): in production, rank resolves through the keyed roster tunnel while
            # recommends are scored elsewhere, so the client warms that host via /api/warm instead.
            if is_mythic_plus(live.tier):
                _warm_personal(tag_n)
            return live
    t = _rank_index().get(tag_n)
    if t:
        if is_mythic_plus(t):
            _warm_personal(tag_n)   # Mythic+ only, and off the critical path — see the live branch
        return S.RankResponse(found=True, tag=tag_n, tier=t, tier_label=tier_label(t),
                              bracket=bracket_of_tier(t), source="dataset",
                              # A dataset row is a crawl snapshot with no season stamp: after a
                              # reset it over-states. Say so whenever the live check that would
                              # have corrected it could not run.
                              stale=live_tried)
    return S.RankResponse(
        found=False, tag=tag_n,
        error="no recent ranked games found" if settings.brawlstars_api_token
        else "not in our data, and live lookup isn't available here")


@app.get("/api/warm")
async def warm(tag: str):
    """Pre-build a tag's personal stats in the background, so the first personalized recommend
    doesn't block on the dataset scan. Exists because the warm that /api/rank fires can land on
    the wrong machine: in production the client resolves rank through the keyed roster tunnel
    (the home host), while /api/recommend is scored here — so the tunnel's cache got warmed and
    this host's stayed cold, and the day's first personalized pick paid the full matches.jsonl
    scan in-request. The client pings this endpoint (on the scoring host) whenever a tag
    resolves and again on each map switch, so a data-refresh cache invalidation mid-session is
    also re-warmed before the next draft's pick phase.

    No Mythic+ gate, unlike the /api/rank warm: blind-pick brackets send ``personal_tag`` too
    (the dual-column personal rail), so any resolvable tag is worth warming. Always returns
    immediately — a full warm pool, unknown tag, or unbooted engine just means a later
    request retries warming, exactly as before. Unauthenticated, so bounded twice: implausible
    tags (wrong length/alphabet — see ``_plausible_tag``) are dropped before spending anything,
    and at most ``_WARM_MAX_CONCURRENCY`` scans run in flight; extra requests no-op."""
    _warm_personal(normalize_tag(tag))
    return {"ok": True}


@app.post("/api/top_picks", response_model=S.TopPicksResponse)
def top_picks(req: S.TopPicksRequest):
    """The strongest picks for the *current board*, with every brawler judged at a full
    loadout (all gadgets, gears & star powers) and **no roster** — so nothing is filtered by
    ownership or mastery. It re-ranks as the draft fills in: brawlers already picked/banned
    drop out, and synergy with your team / counters to theirs fold into the score. This is
    the pure population meta ('who's strongest here right now'), the deliberate counterpart
    to /api/recommend, which personalizes to the player's roster & history."""
    state = DraftState(
        map_id=req.map_id, mode=req.mode,
        our_team=list(req.our_team), their_team=list(req.their_team), bans=list(req.bans),
        rank_bracket=req.rank_bracket,
    )
    picks = _engine.recommend_picks(state, top=req.top, roster=None)  # roster=None ⇒ full loadout
    return S.TopPicksResponse(
        map_id=req.map_id, mode=req.mode, rank_bracket=req.rank_bracket,
        picks=[
            S.TopPick(brawler_id=p.brawler_id, name=p.name, cls=p.cls,
                      score=round(p.score, 4), map_winrate=round(p.map_winrate, 4))
            for p in picks
        ],
    )


class _ReqMastery:
    """Lite stand-in for :class:`engine.mastery.Mastery` built from a client-sent roster entry.

    Exposes the ``.score`` the roster UI displays, the ``.gaps()`` it shows as chips, and the
    ``.fielded()`` readiness view the scorer prices. Lets the public backend personalize from a
    roster the client fetched (via the keyed tunnel) but the backend itself can't reach.

    Power, gear count, and optional Buffy ownership are kept, not just the gate's verdict on them:
    the floor decides whether a brawler is *selectable*, while readiness prices how far the
    selectable copy is from the maxed one the meta table describes. An absent Buffy object means
    an older roster schema and stays neutral; explicit false flags are joined to curated
    availability by the scorer."""
    __slots__ = ("score", "_gaps", "_power", "_n_gears", "_buffies")

    def __init__(self, score: float, gaps: List[str], power: int = 0, n_gears: int = 0,
                 buffies: Optional[dict] = None):
        self.score = max(0.0, min(1.0, float(score)))
        self._gaps = list(gaps or [])
        self._power = int(power or 0)
        self._n_gears = int(n_gears or 0)
        self._buffies = None if buffies is None else dict(buffies)

    def gaps(self) -> List[str]:
        return self._gaps

    def fielded(self) -> Fielded:
        return Fielded.from_gaps(self._power, self._gaps, self._n_gears, self._buffies)


class _BoostedMastery:
    """Mastery stand-in for a season's free/"boosted" brawler — owned or not.

    Ranked hands these out fully maxed to *everyone* — Power 11, every star power / gadget / gear /
    hypercharge — regardless of whether the player owns the brawler, or at what level. The live
    signal in :mod:`bsdraft.engine.freebrawlers` is exactly this: *every* ranked slot for a free
    brawler reads Power 11, with no levelling tail, which can only be true if owners field the maxed
    loan too. So ``.fielded()`` is fully ready and the brawler takes **no** readiness deficit, and
    ``.gaps()`` is empty — for an owned copy just as much as an unowned one. That is the whole point:
    a free maxed brawler is exactly the copy the meta win rate describes.

    ``.score`` is the display-only investment index. It defaults to 0.60 (full *build*, zero
    *comfort*) for the unowned case, where the player has no history; an owned free brawler keeps its
    real score, so the roster UI still shows the comfort the player has actually earned. Either way
    the score no longer touches the pick score, so pricing the copy as ready costs nothing there."""
    __slots__ = ("score",)

    def __init__(self, score: float = 0.60):
        self.score = float(score)

    def gaps(self) -> List[str]:
        return []

    def fielded(self) -> Fielded:
        return Fielded.ready()


def _roster_for(req: S.RecommendRequest):
    """Roster dict ``{brawler_id: mastery-like}`` to personalize against, or None. Prefers the
    client-sent roster (the only source on the public host), then the server's own roster
    (local/home, where the IP-locked key can fetch it). This season's free/"boosted" brawlers are
    folded in as available-at-full-loadout so they're recommendable even when unowned — and an owned
    copy is priced as fully maxed too, since Ranked loans everyone the maxed brawler (it keeps only
    its real display score). Returns None unless ``personalize`` is set.

    The server-roster fallback applies only when the ``roster`` field is *omitted* (None). An
    explicitly sent empty list means "this player fields nothing" (the client's power-floor filter
    can empty a real roster) and must personalize against exactly that, never the operator's
    roster. ``_engine.roster`` itself is only ever the operator's own, preloaded once at startup
    from PLAYER_TAG (local/home) — ``/api/roster`` no longer writes here, so no per-request path
    can put a visitor's roster on the shared engine.

    Owned brawlers below the bracket's power floor are dropped: Ranked hard-blocks selecting a
    brawler under Power 9 (through Diamond) / Power 11 (Mythic up), so recommending one the player
    couldn't field is a bug — the very report that motivated this gate. Boosted brawlers arrive at
    Power 11 and are added *after* the filter, so they always clear it. A reported power of 0 means
    "unknown" (an older client that omits the field) and is left in, so the gate never empties a
    roster that simply predates power being sent."""
    if not req.personalize:
        return None
    floor = min_power_for_bracket(req.rank_bracket)
    fieldable = lambda power: power == 0 or power >= floor
    if req.roster is not None:
        roster = {
            e.id: _ReqMastery(
                e.mastery, e.gaps, e.power, len(e.owned_gears or ()),
                e.buffies.model_dump(exclude_none=True) if e.buffies is not None else None,
            )
            for e in req.roster if fieldable(e.power)
        }
    elif _engine.roster:
        roster = {bid: m for bid, m in _engine.roster.items() if fieldable(m.power)}
    else:
        roster = None
    if roster is None:
        return None
    for bid in _free_brawler_ids():
        # Ranked loans the free brawler fully maxed to everyone, owners included, so an owned copy —
        # even under-levelled or half-built — must be priced as ready with no gaps, not by its real
        # deficient mastery (owning a weak copy would otherwise recommend it *worse* than not owning
        # it). Overwrite, not setdefault; keep the owned copy's real display score for the roster UI.
        prior = roster.get(bid)
        roster[bid] = _BoostedMastery(prior.score if prior is not None else 0.60)
    return roster


def _free_brawler_ids() -> tuple:
    """Ids of brawlers that are free/"boosted" in Ranked right now, folded into a personalized
    roster at full loadout so an unowned-or-underlevelled free brawler is still recommendable.

    Two sources, unioned — belt and suspenders because a wrong *omission* here silently deletes
    the map's best pick (the Nori report), while a wrong *inclusion* only recommends a brawler the
    player can't field, and both sources are conservative:

      * The hand-maintained ``ranked_boosted.json`` (:func:`R.load_ranked_boosted`) — the leading
        signal, and the only one that knows *next* season before any games are played.
      * The data-derived set (:mod:`bsdraft.engine.freebrawlers`, computed at stats-build time and
        carried in the stats artifact) — the authoritative signal for what is free *now*, which
        catches unannounced mid-season grants the release notes never mention.
    """
    ids = set(R.load_ranked_boosted())
    stats = getattr(_engine, "stats", None)
    if stats is not None:
        ids |= set(getattr(stats, "free_brawler_ids", ()))
    return tuple(ids)


@app.post("/api/recommend", response_model=S.RecommendResponse)
def recommend(req: S.RecommendRequest):
    state = DraftState(
        map_id=req.map_id, mode=req.mode,
        our_team=list(req.our_team), their_team=list(req.their_team), bans=list(req.bans),
        we_pick_first=req.we_pick_first, solo_queue=req.solo_queue, rank_bracket=req.rank_bracket,
    )
    roster = _roster_for(req)
    composition = _engine.composition(state)
    warnings = _engine.composition_report(state)["warnings"]
    game_plan = S.GamePlan(**_engine.game_plan(state))
    next_to_act = state.next_to_act()

    if req.phase == "ban":
        # The roster matters during bans too: a brawler the player can't field is free to ban,
        # while banning one of their own projected picks costs them.
        bans = _engine.recommend_bans(state, top=req.top, roster=roster)
        return S.RecommendResponse(
            phase="ban", bans=[S.BanRec(**vars(b)) for b in bans],
            composition=composition, warnings=warnings, game_plan=game_plan, next_to_act=next_to_act,
        )

    # Personal win-rate signal — only feeds pick scoring, and the build scans the dataset, so don't
    # pay for it during the ban phase (the result would be discarded there) — that scan, fired on
    # every ban placement, was the bulk of the blind-pick "analyzing…" stall before the first pick.
    personal = _personal_for(req.personal_tag)
    picks = [S.PickRec(**vars(p))
             for p in _engine.recommend_picks(state, top=req.top, roster=roster, personal=personal)]

    return S.RecommendResponse(
        phase="pick", picks=picks,
        composition=composition, warnings=warnings, game_plan=game_plan, next_to_act=next_to_act,
    )
