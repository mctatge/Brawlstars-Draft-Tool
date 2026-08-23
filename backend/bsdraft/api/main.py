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

from bsdraft.api import schemas as S
from bsdraft.collect.client import BrawlStarsClient, normalize_tag
from bsdraft.config import settings
from bsdraft.constants import RANKED_MODES
from bsdraft.data import reference as R
from bsdraft.data import sync
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
from bsdraft.engine.tiers import BRACKETS, bracket_of_tier, min_power_for_bracket, tier_label
from bsdraft.models.serve import WinProbModel

logger = logging.getLogger("bsdraft.api")

_engine: Optional[DraftEngine] = None
_last_check: float = 0.0   # epoch of the last sync attempt (liveness)
_last_change: float = 0.0  # epoch of the last actual data change
_meta_cache = None         # (data_version, MetaReport); recomputed lazily when data changes
_rank_idx_cache = None     # (token, RankIndex); recomputed lazily when the data/artifact changes
_rank_version = 0          # bumps when the synced rank-index artifact changes (invalidates the cache)
_personal_cache: dict = {} # tag -> (data_version, PersonalStats|None); rebuilt when data changes
_personal_locks: dict = {} # tag -> Lock; single-flights the per-tag dataset scan (no stampede)
_personal_locks_guard = threading.Lock()
_roster_cache: dict = {}   # normalized tag -> (fetched_at, RosterResponse); short TTL spares the live API
# Share of its mode's busiest map that a map must reach to count as "in the current rotation".
# The observed gap between a live map and a retired one is ~20x, so anything from ~0.1 to ~0.5
# separates them; 0.15 leans toward keeping a map that is merely quiet.
MIN_SHARE_OF_MODE_LEADER = 0.15

_rank_cache: dict = {}     # normalized tag -> (fetched_at, RankResponse); short TTL on live rank lookups


def _build_stats():
    """Produce ``(global_stats, {bracket: stats})``. When STATS_URL is set the API **loads** the
    precomputed stats artifact (built off-box from *all* matches, ~tens of MB, no OOM); otherwise
    it **rebuilds** them from the synced matches, capped to STATS_MAX_MATCHES to bound peak RAM."""
    if settings.stats_url and sync.STATS_PATH.exists():
        try:
            return load_stats(sync.STATS_PATH)
        except Exception as e:  # noqa: BLE001 — a corrupt/old artifact must not break startup
            logger.warning("stats load failed (%s); rebuilding from matches", e)
    return build_bracketed(halflife_days=settings.stats_halflife_days,
                           max_matches=settings.stats_max_matches)


def _rank_index() -> RankIndex:
    """Cached ``tag -> tier`` lookup. When RANK_INDEX_URL is set the API **loads** the precomputed
    artifact (a compact NumPy-backed ~20 MB structure, no in-memory build); otherwise it **builds**
    the index from the synced matches (~1.3M-entry dict, ~200 MB) and packs it down. Rebuilt lazily
    when the synced artifact changes (``_rank_version``) or, for the local build, when the matches
    change (``_last_change``)."""
    global _rank_idx_cache
    use_artifact = bool(settings.rank_index_url) and sync.RANK_INDEX_PATH.exists()
    token = (use_artifact, _rank_version if use_artifact else _last_change)
    if _rank_idx_cache is None or _rank_idx_cache[0] != token:
        if use_artifact:
            try:
                idx = load_rank_index(sync.RANK_INDEX_PATH)
            except Exception as e:  # noqa: BLE001 — a corrupt/old artifact must fall back, not 500
                logger.warning("rank index load failed (%s); building from matches", e)
                idx = RankIndex.from_mapping(build_rank_index())
        else:
            idx = RankIndex.from_mapping(build_rank_index())
        _rank_idx_cache = (token, idx)
    return _rank_idx_cache[1]


def _personal_for(tag: Optional[str]):
    """Cached personal stats for ``tag``, derived from the synced dataset (key-free, so it
    works on the public host) and rebuilt when the data changes. Returns None when the tag
    is empty or the player has no labeled games in our data. A live battle-log augment can
    pre-seed a richer entry at startup (see lifespan), which this cache then serves."""
    t = normalize_tag(tag or "")
    if not t or _engine is None:
        return None
    hit = _personal_cache.get(t)
    if hit is not None and hit[0] == _last_change:
        return hit[1]
    # Building scans the dataset for this tag's games (seconds on the full cloud dataset). Single-
    # flight per tag so a burst of requests for the same uncached tag — rapid picks, the frontend
    # re-polling, multiple tabs — waits on one build instead of each launching its own redundant
    # scan (a cache stampede; the cache is only written once the scan finishes).
    with _personal_locks_guard:
        if len(_personal_locks) > 512:   # bound growth — one tiny Lock per unique tag
            _personal_locks.clear()
        lock = _personal_locks.setdefault(t, threading.Lock())
    with lock:
        hit = _personal_cache.get(t)     # another thread may have built it while we waited
        if hit is not None and hit[0] == _last_change:
            return hit[1]
        if len(_personal_cache) > 256:   # simple bound — tags are cheap to rebuild
            _personal_cache.clear()
        ps = build_personal_stats(t, fallback=_engine.stats)
        _personal_cache[t] = (_last_change, ps)
        return ps


async def _refresh_loop() -> None:
    """Periodically re-sync the dataset (and model) and hot-swap rebuilt stats / a reloaded
    model into the live engine, so a fresh crawl or retrain rolls out with no restart."""
    global _last_check, _last_change, _rank_version
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(settings.refresh_seconds)
        try:
            data_changed = (await loop.run_in_executor(None, sync.sync_matches, settings.data_url)
                            if settings.data_url else False)
            _last_check = time.time()
            if data_changed and _engine is not None:
                _last_change = time.time()  # invalidate the rank / meta / personal caches
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
            if settings.stats_url and _engine is not None:
                if await loop.run_in_executor(None, sync.sync_stats, settings.stats_url):
                    g, br = await loop.run_in_executor(None, _build_stats)
                    _engine.stats, _engine.bracket_stats = g, br
                    logger.info("draft stats reloaded: %d matches, %d bracket(s)", g.n, len(br))
            elif data_changed and _engine is not None:
                g, br = await loop.run_in_executor(None, _build_stats)
                _engine.stats, _engine.bracket_stats = g, br
                logger.info("draft stats rebuilt: %d matches, %d bracket(s)", g.n, len(br))
            if settings.model_url and _engine is not None:
                if await loop.run_in_executor(None, sync.sync_model, settings.model_url):
                    _engine.model = await loop.run_in_executor(None, WinProbModel)  # atomic swap
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
    global _engine, _last_check, _last_change
    loop = asyncio.get_running_loop()
    if settings.data_url:
        await loop.run_in_executor(None, sync.sync_matches, settings.data_url)
        _last_check = _last_change = time.time()
    if settings.model_url:
        await loop.run_in_executor(None, sync.sync_model, settings.model_url)
    if settings.stats_url:
        await loop.run_in_executor(None, sync.sync_stats, settings.stats_url)
    if settings.rank_index_url:
        await loop.run_in_executor(None, sync.sync_rank_index, settings.rank_index_url)
    if settings.meta_report_url:
        await loop.run_in_executor(None, sync.sync_meta_report, settings.meta_report_url)
    if settings.itemstats_url:
        await loop.run_in_executor(None, sync.sync_itemstats, settings.itemstats_url)
    g, br = _build_stats()
    _engine = DraftEngine(g, WinProbModel(), bracket_stats=br)
    if settings.player_tag:
        ptag = normalize_tag(settings.player_tag)
        try:
            async with BrawlStarsClient() as client:
                _engine.roster, _engine.roster_name = await mastery.fetch_roster(client, settings.player_tag)
                # Prime personal stats with the player's freshest games (needs the live key,
                # so local/home only); the public host falls back to dataset-derived stats.
                try:
                    extra = matches_from_battlelog(await client.get_battlelog(ptag), ptag)
                    _personal_cache[ptag] = (_last_change, build_personal_stats(
                        ptag, fallback=_engine.stats, extra_matches=extra))
                except Exception:
                    pass
        except Exception:
            _engine.roster, _engine.roster_name = None, ""
    task = None
    if (settings.data_url or settings.model_url or settings.stats_url
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
    return {
        "status": "ok",
        "model": bool(_engine and _engine.model and _engine.model.available),
        "matches": _engine.stats.n if _engine else 0,
        "roster": bool(_engine and _engine.roster),
        "refresh_seconds": settings.refresh_seconds if settings.data_url else 0,
        "last_check": _last_check or None,
        "last_change": _last_change or None,
    }


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
            logger.warning("meta report load failed (%s); computing from matches", e)
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
    brawlers = [
        S.BrawlerRef(id=b.id, name=b.name, cls=b.cls, rarity=b.rarity, image_url=b.image_url)
        for b in R.load_brawlers()
    ]
    # `load_ranked_maps()` is the catalog's *not-retired* set — every map still in the game's
    # files across all modes, ~113 of them. Ranked only rotates a handful per mode per season, so
    # showing the catalog offers map/mode pairs nobody can queue (e.g. "Heist: Pit Stop"), and the
    # model has nothing to say about them anyway. Collected ranked games are the only rotation
    # signal we have — a map with none is one we have never seen played. Same idiom as
    # `engine.py`'s Brawl Ball pick. Falls back to the full list when stats aren't loaded yet, so
    # a cold start shows too much rather than nothing.
    #
    # Deliberately filtered *here* and not in `load_ranked_maps()`: that function also builds the
    # model's pinned map vocabulary (`encoders.py`, `export_model.py`), where dropping entries
    # would shift every embedding row out from under the trained checkpoint.
    #
    # The cost of a relative cut is that a map added mid-rotation stays hidden until it has
    # accumulated its share of games. That is the right way to be wrong here: a map we have
    # barely seen is one the model has nothing to say about either.
    all_maps = R.load_ranked_maps()
    games = {m.id: (_engine.stats.map_games.get(m.id, 0) if _engine else 0) for m in all_maps}
    # "Has any games at all" is too weak a cut: the stats span more history than one rotation, so
    # a map retired seasons ago keeps a decaying residue and drifts back into the list. The real
    # separation is per-mode and enormous — every map in the live rotation sits within ~8% of its
    # mode's busiest map, while a retiree sits 20x below it (2026-08-20: Heist ran four maps at
    # 1954-2026 games with Pit Stop on 90, Brawl Ball four at 2059-2183 with Spiraling Out on 71).
    # Cut on a share of the mode's leader, not an absolute count, so the threshold rides the
    # crawl's volume instead of needing a retune every time the dataset grows.
    top_per_mode = {}
    for m in all_maps:
        top_per_mode[m.mode] = max(top_per_mode.get(m.mode, 0), games[m.id])
    played = [m for m in all_maps
              if games[m.id] >= max(1, MIN_SHARE_OF_MODE_LEADER * top_per_mode[m.mode])]
    maps = [
        S.MapRef(id=m.id, name=m.name, mode=m.mode, image_url=m.image_url,
                 games=int(_engine.stats.map_games.get(m.id, 0)) if _engine else 0)
        for m in (played or all_maps)
    ]
    brackets = [b for b in BRACKETS if _engine and b in _engine.bracket_stats]
    return S.ReferenceResponse(brawlers=brawlers, maps=maps, modes=list(RANKED_MODES),
                               brackets=brackets, boosted=list(R.load_ranked_boosted()))


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
        async with BrawlStarsClient() as client:
            r, name = await mastery.fetch_roster(client, t)
        _engine.roster, _engine.roster_name = r, name
        owned = [
            S.OwnedBrawler(
                id=bid, mastery=round(m.score, 3), gaps=m.gaps(),
                owned_star_powers=list(m.owned_star_powers),
                owned_gadgets=list(m.owned_gadgets),
                owned_gears=[S.OwnedGear(**g) for g in m.owned_gears],
                # Progression state the purchase advisor needs (already parsed by Mastery).
                power=m.power, has_hypercharge=m.has_hypercharge,
            )
            for bid, m in r.items()
        ]
        resp = S.RosterResponse(loaded=True, tag=t, name=name, owned=owned)
        if len(_roster_cache) > 512:   # bound growth — one entry per unique tag, TTL alone never frees it
            _roster_cache.clear()
        _roster_cache[key] = (time.time(), resp)  # cache only successful loads; errors retry next poll
        return resp
    except Exception as e:  # noqa: BLE001
        return S.RosterResponse(loaded=False, tag=t, name="", error=str(e))


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
        async with BrawlStarsClient() as client:
            player = await client.get_player(tag_n)
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
            return live
    t = _rank_index().get(tag_n)
    if t:
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

    Power and the gear count are kept, not just the gate's verdict on them: the floor decides
    whether a brawler is *selectable*, while readiness prices how far the selectable copy is from
    the maxed one the meta table describes. Both read the same wire field."""
    __slots__ = ("score", "_gaps", "_power", "_n_gears")

    def __init__(self, score: float, gaps: List[str], power: int = 0, n_gears: int = 0):
        self.score = max(0.0, min(1.0, float(score)))
        self._gaps = list(gaps or [])
        self._power = int(power or 0)
        self._n_gears = int(n_gears or 0)

    def gaps(self) -> List[str]:
        return self._gaps

    def fielded(self) -> Fielded:
        return Fielded.from_gaps(self._power, self._gaps, self._n_gears)


class _BoostedMastery:
    """Mastery stand-in for a season's free/"boosted" brawler the player doesn't own.

    Ranked hands these out fully maxed — Power 11, every star power / gadget / gear / hypercharge —
    so ``.fielded()`` is fully ready and the brawler takes **no** readiness deficit. That is the
    whole point: a free maxed brawler is exactly the copy the meta win rate describes.

    ``.score`` stays 0.60 (full *build*, zero *comfort*) because it is a display-only investment
    index and the player genuinely has no history here. It no longer touches the pick score, so the
    old worry — that this constant quietly depressed free maxed brawlers — is gone with the term."""
    __slots__ = ()
    score = 0.60

    def gaps(self) -> List[str]:
        return []

    def fielded(self) -> Fielded:
        return Fielded.ready()


def _roster_for(req: S.RecommendRequest):
    """Roster dict ``{brawler_id: mastery-like}`` to personalize against, or None. Prefers the
    client-sent roster (the only source on the public host), then the server's own roster
    (local/home, where the IP-locked key can fetch it). This season's free/"boosted" brawlers are
    folded in as available-at-full-loadout so they're recommendable even when unowned (an owned
    one keeps its real mastery). Returns None unless ``personalize`` is set.

    The server-roster fallback applies only when the ``roster`` field is *omitted* (None). An
    explicitly sent empty list means "this player fields nothing" (the client's power-floor filter
    can empty a real roster) and must personalize against exactly that — on a multi-visitor host,
    ``_engine.roster`` holds whichever roster ``/api/roster`` fetched *last*, so falling through
    on ``[]`` would score one player's draft against another player's brawlers.

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
        roster = {e.id: _ReqMastery(e.mastery, e.gaps, e.power, len(e.owned_gears or ()))
                  for e in req.roster if fieldable(e.power)}
    elif _engine.roster:
        roster = {bid: m for bid, m in _engine.roster.items() if fieldable(m.power)}
    else:
        roster = None
    if roster is None:
        return None
    for bid in R.load_ranked_boosted():
        roster.setdefault(bid, _BoostedMastery())  # owned boosted brawler keeps its real mastery
    return roster


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
