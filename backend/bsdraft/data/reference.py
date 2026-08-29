"""Load and clean Brawl Stars reference data (brawlers, maps, modes).

Source: the keyless Brawlify/BrawlAPI JSON snapshots in ``data/reference/``. Provides
clean, typed accessors plus a stable contiguous brawler index for model embeddings.

Pure stdlib so it runs without installing the ML/runtime dependencies:

    PYTHONPATH=backend python -m bsdraft.data.reference
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

from bsdraft.constants import (
    BRAWLER_CLASSES,
    RANKED_MODES,
    REFERENCE_DIR,
    UNCLASSIFIED,
)

CLASS_OVERRIDES_PATH = Path(__file__).resolve().parent / "class_overrides.json"
RANKED_BOOSTED_PATH = REFERENCE_DIR / "ranked_boosted.json"
ECONOMY_PATH = REFERENCE_DIR / "economy.json"

# How long a rotation / grant with no explicit ``valid_until`` may keep serving after its start.
# A Ranked season runs ~4-5 weeks, so this never truncates a live one; it exists so an
# unmaintained file goes QUIET rather than asserting a long-expired free set forever.
_MAX_UNBOUNDED_ROTATION_DAYS = 45

# Maps that are live in the ranked rotation but whose upstream Brawlify/BrawlAPI ``disabled``
# flag lags behind it (the events feed tracks casual rotation and can miss a ranked-only pool).
# Force-enabling here — rather than editing ``maps.json`` — keeps the fix from being clobbered
# the next time ``scripts/refresh_reference.py`` rewrites the snapshot from upstream. The map
# still only surfaces on the site once the crawler has accumulated its share of games for the
# mode (see the reference endpoint), so this just lets it back into the pool to be collected.
#   15000886  Safe(r) Zone (Heist) — confirmed live 2026-08-23; upstream still flags disabled.
RANKED_MAP_ENABLE_OVERRIDES = frozenset({15000886})


@dataclass(frozen=True)
class Accessory:
    id: int
    name: str
    kind: str  # "star_power" | "gadget"
    image_url: str = ""
    description: str = ""  # raw catalog text (carries unfilled `x`/`<!card...>` value tokens)


@dataclass(frozen=True)
class Brawler:
    id: int
    name: str
    cls: str  # a member of BRAWLER_CLASSES, or UNCLASSIFIED
    rarity: str
    star_powers: tuple
    gadgets: tuple
    image_url: str
    # False for a catalog entry still flagged unreleased upstream (a datamined / leaked-early
    # brawler that sits in the snapshot before it is live in the game). Such an entry stays in the
    # vocabulary (see load_brawlers / brawler_index) but must be kept out of every pickable /
    # recommendable pool — see pickable_brawlers(). Defaults True so a snapshot missing the key
    # can't silently hide a real brawler.
    released: bool = True


@dataclass(frozen=True)
class GameMap:
    id: int
    name: str
    mode: str
    environment: str
    image_url: str


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def class_overrides() -> dict:
    """Manual class assignments for brawlers Brawlify hasn't tagged yet."""
    if not CLASS_OVERRIDES_PATH.exists():
        return {}
    data = _load_json(CLASS_OVERRIDES_PATH)
    if isinstance(data, dict):
        return data.get("overrides", data)
    return {}


def _resolve_class(raw_brawler: dict, overrides: dict) -> str:
    cls = (raw_brawler.get("class") or {}).get("name")
    if not cls or cls == "Unknown":
        cls = overrides.get(raw_brawler["name"], UNCLASSIFIED)
    return cls if cls in BRAWLER_CLASSES else UNCLASSIFIED


@lru_cache(maxsize=1)
def load_brawlers() -> tuple:
    """All brawlers, sorted by id, with classes cleaned via overrides."""
    raw = _load_json(REFERENCE_DIR / "brawlers.json")["list"]
    overrides = class_overrides()
    brawlers = [
        Brawler(
            id=x["id"],
            name=x["name"],
            cls=_resolve_class(x, overrides),
            rarity=(x.get("rarity") or {}).get("name", ""),
            star_powers=tuple(
                Accessory(sp["id"], sp["name"], "star_power",
                          sp.get("imageUrl", ""), sp.get("description", ""))
                for sp in (x.get("starPowers") or [])
            ),
            gadgets=tuple(
                Accessory(g["id"], g["name"], "gadget",
                          g.get("imageUrl", ""), g.get("description", ""))
                for g in (x.get("gadgets") or [])
            ),
            image_url=x.get("imageUrl", ""),
            released=bool(x.get("released", True)),
        )
        for x in raw
    ]
    brawlers.sort(key=lambda b: b.id)
    return tuple(brawlers)


@lru_cache(maxsize=1)
def pickable_brawlers() -> tuple:
    """Released brawlers only — the draftable / recommendable / UI-pickable pool.

    Excludes any catalog entry still flagged ``released: false`` (a datamined or leaked-early
    brawler that sits in the snapshot before it is live in the game). Those have no match data,
    so every data-driven signal falls back to a neutral prior and the entry is never *penalised*
    the way a real brawler with mixed stats is — on an otherwise-empty board it floats to the top
    of the pick list while not even being selectable in-game.

    Deliberately filtered **here** and not in :func:`load_brawlers` / :func:`brawler_index`: those
    build the model's pinned embedding vocabulary, where dropping an entry would shift every
    higher-id brawler onto a neighbour's trained row and desync from ``winprob.npz``. This is the
    same "filter at the boundary, not in the vocab-building loader" idiom the reference endpoint
    uses for maps (see ``api/main.py``). A leaked entry keeps its pinned row and re-enters this
    pool automatically the moment ``released`` flips true upstream — no code change, no retrain."""
    return tuple(b for b in load_brawlers() if b.released)


@lru_cache(maxsize=1)
def brawler_index() -> dict:
    """Stable brawler id -> contiguous index (0..N-1), sorted by id. For embeddings."""
    return {b.id: i for i, b in enumerate(load_brawlers())}


@lru_cache(maxsize=1)
def _by_name() -> dict:
    return {b.name.lower(): b for b in load_brawlers()}


def brawler_by_name(name: str) -> Optional[Brawler]:
    return _by_name().get(name.strip().lower())


@lru_cache(maxsize=1)
def load_ranked_maps() -> tuple:
    """Active maps belonging to the 5 ranked modes, sorted by (mode, name)."""
    raw = _load_json(REFERENCE_DIR / "maps.json")["list"]
    maps = []
    for x in raw:
        if x.get("disabled") and x.get("id") not in RANKED_MAP_ENABLE_OVERRIDES:
            continue
        mode = (x.get("gameMode") or {}).get("name")
        if mode not in RANKED_MODES:
            continue
        maps.append(
            GameMap(
                id=x["id"],
                name=x["name"],
                mode=mode,
                environment=(x.get("environment") or {}).get("name", ""),
                image_url=x.get("imageUrl", ""),
            )
        )
    maps.sort(key=lambda m: (m.mode, m.name))
    return tuple(maps)


@lru_cache(maxsize=1)
def _ranked_boosted_doc() -> Optional[dict]:
    """Parsed ``ranked_boosted.json``, or None when absent/unreadable. Only the file parse is
    cached — the date logic (:func:`_rotation_for_now`) runs per call, because the deployed API
    is deliberately kept warm for days: a process-start-only check would keep serving an expired
    rotation (or miss a staged season handover) long after the boundary."""
    if not RANKED_BOOSTED_PATH.exists():
        return None
    try:
        return _load_json(RANKED_BOOSTED_PATH)
    except (json.JSONDecodeError, OSError):
        return None


def _now_utc() -> datetime:
    """Serving-side clock, anchored to **UTC** explicitly (test seam). Render runs UTC but the
    home Mac runs local time, and Supercell's own season boundaries are UTC-hour events — an
    implicit local ``date.today()`` would make the two serving hosts flip the FREE set hours
    apart, and hand-staged dates untargetable."""
    return datetime.now(timezone.utc)


def _parse_boundary(value, *, end_of_day: bool) -> Optional[datetime]:
    """A hand-staged rotation boundary → aware UTC datetime, or None when the value isn't a
    parseable string (the *callers* pick the fail-safe direction for None). Two forms:
    a bare ISO date (``2026-08-19``) means the whole **UTC** day — its first instant as a start,
    its last as an end (``end_of_day``) — and a full ISO datetime (``2026-08-19T10:00:00Z`` or
    with an offset; naive means UTC) pins the exact instant, for when the in-game flip hour is
    known. Type-checked because the file is hand-edited: a non-string here must fail safe, not
    500 every request that loads the rotation."""
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    try:
        d = date.fromisoformat(s)  # strict: bare dates only
        return datetime.combine(d, dt_time.max if end_of_day else dt_time.min, tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _rotation_for_now(doc: dict, now: datetime) -> Optional[dict]:
    """The rotation to serve at ``now`` (aware UTC): ``active`` while its ``valid_until``
    boundary hasn't passed, else — and this is what makes a season flip hands-off — the
    ``upcoming`` entry whose hand-staged ``active_from`` has arrived. No API or blog exposes the
    live season, so the flip moment is human knowledge; staging it in the data lets the handover
    happen at the boundary with no midnight edit-and-deploy. Where windows overlap, the rotation
    with the latest start wins (a staged season takes over from a sloppy ``valid_until``).

    Fail-safes are asymmetric on purpose: a malformed ``valid_until`` keeps ``active`` serving
    (expiring early misleads no one), but a malformed / absent ``active_from`` keeps an upcoming
    rotation *unserved* (starting early would mislead — the exact case this file's fail-safes
    exist to prevent). Between an expired ``valid_until`` and a not-yet-arrived ``active_from``
    nothing serves — a deliberate gap for when the exact flip hour is unknown.

    An entry may carry its own ``valid_until``; a *promoted* upcoming entry that carries none
    would otherwise serve forever, because only ``active`` was ever bounded (an unmaintained file
    then keeps asserting a long-dead free set — fail-dangerous, the one thing this file must not
    do). So an un-ended rotation is capped at ``_MAX_UNBOUNDED_ROTATION_DAYS`` past its start:
    well beyond a real season, short enough that a stale file goes quiet instead of lying."""
    best = None  # (start, rotation) — later start wins; an upcoming entry beats `active` on a tie
    active = doc.get("active") or {}
    if active:
        # Entry-level end wins over the legacy top-level one, which only ever bounded `active`.
        until = _parse_boundary(active.get("valid_until"), end_of_day=True)
        if until is None:
            until = _parse_boundary(doc.get("valid_until"), end_of_day=True)
        if until is None or now <= until:  # unset/malformed boundary — keep serving
            best = (datetime.min.replace(tzinfo=timezone.utc), active)
    for entry in doc.get("upcoming") or []:
        if not isinstance(entry, dict):
            continue
        start = _parse_boundary(entry.get("active_from"), end_of_day=False)
        if start is None:
            continue  # not staged (or malformed) — never risk serving a rotation early
        end = _parse_boundary(entry.get("valid_until"), end_of_day=True)
        if end is None:
            end = start + timedelta(days=_MAX_UNBOUNDED_ROTATION_DAYS)
        if start <= now <= end and (best is None or start >= best[0]):
            best = (start, entry)
    return best[1] if best else None


def _active_grants(doc: dict, now: datetime) -> list:
    """Brawler names free right now via a **grant** — free outside the seasonal rotation.

    The release notes' "Free Brawler Rotation" is not the whole free set: a brawler can be made
    free mid-season with no announcement at all (observed 2026-08-25, when Nori went to 100%
    Power 11 in every ranked match five days after the Season 2 flip). Those grants are recorded
    here as ``{"brawler", "since", "valid_until"}`` entries, separate from ``active``/``upcoming``
    so a boosted-watch rewrite (which only knows what the notes print) can't destroy them.

    Same asymmetric fail-safes as the rotation: a missing/malformed ``since`` never serves
    (starting early misleads), while ``valid_until`` is optional and capped the same way an
    un-ended rotation is, so a forgotten grant expires instead of lying forever."""
    out = []
    for entry in doc.get("grants") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("brawler")
        if not isinstance(name, str) or not name.strip():
            continue
        start = _parse_boundary(entry.get("since"), end_of_day=False)
        if start is None or start > now:
            continue
        end = _parse_boundary(entry.get("valid_until"), end_of_day=True)
        if end is None:
            end = start + timedelta(days=_MAX_UNBOUNDED_ROTATION_DAYS)
        if now <= end:
            out.append(name)
    return out


def load_ranked_boosted() -> tuple:
    """Brawler ids of the current season's Ranked **free / "boosted" brawlers** — the maxed
    brawlers everyone may use in Ranked regardless of ownership. Read from the committed
    ``ranked_boosted.json`` (maintained by the ``boosted-watch`` scraper, see
    :mod:`bsdraft.collect.boosted`); the rotation's **names** are resolved to ids here so a
    catalog rename can't strand a hard-coded id. Which rotation serves — ``active``, or an
    ``upcoming`` entry whose staged ``active_from`` has arrived — is decided per call by
    :func:`_rotation_for_now` against a UTC clock, so season flips happen at the staged boundary
    even on a long-lived, kept-warm process, and every serving host agrees on the FREE set.

    The seasonal rotation is **not** the whole free set — a brawler can be granted free
    mid-season with no announcement (see :func:`_active_grants`), so any currently-serving
    ``grants`` entries are unioned in. This file is only the hand-maintained half of the answer;
    the authoritative live signal is derived from match data (see
    :mod:`bsdraft.engine.freebrawlers`), which catches unannounced grants on its own.

    Fail-safe (the list must never *mislead* — telling a player they can freely pick a brawler
    they actually can't is worse than showing none): returns ``()`` when the file is absent /
    unreadable, when no name resolves, or when nothing covers the current moment."""
    doc = _ranked_boosted_doc()
    if doc is None:
        return ()
    now = _now_utc()
    rotation = _rotation_for_now(doc, now) or {}
    names = list(rotation.get("brawlers", []) or []) + _active_grants(doc, now)
    ids = []
    for name in names:
        b = brawler_by_name(name) if isinstance(name, str) else None
        if b is not None and b.id not in ids:
            ids.append(b.id)
    return tuple(ids)


@lru_cache(maxsize=1)
def load_economy() -> dict:
    """Curated progression-economy table for the purchase advisor (costs, power gates, impact
    priors, hypercharge availability). Hand-maintained — none of this is in any catalog and the
    API never exposes prices; see ``economy.json`` and ``docs/purchase-advisor.md``.

    Fail-safe to ``{}`` so a missing/broken file degrades the advisor to 'no cost/gate context'
    rather than erroring the endpoint (the consumer treats every section as optional)."""
    if not ECONOMY_PATH.exists():
        return {}
    try:
        doc = _load_json(ECONOMY_PATH)
    except (json.JSONDecodeError, OSError):
        return {}
    return doc if isinstance(doc, dict) else {}


def summary() -> str:
    brawlers = load_brawlers()
    maps = load_ranked_maps()
    cls_counts = Counter(b.cls for b in brawlers)
    unclassified = [b.name for b in brawlers if b.cls == UNCLASSIFIED]
    map_counts = Counter(m.mode for m in maps)

    lines = [
        f"Brawlers: {len(brawlers)}  (embedding index 0..{len(brawlers) - 1})",
        "  classes: " + ", ".join(f"{k}={cls_counts[k]}" for k in BRAWLER_CLASSES),
    ]
    if unclassified:
        lines.append(f"  UNCLASSIFIED ({len(unclassified)}): " + ", ".join(unclassified))
    else:
        lines.append("  UNCLASSIFIED: 0  (all brawlers classified)")
    lines.append(f"Ranked maps: {len(maps)}")
    for mode in RANKED_MODES:
        lines.append(f"  {mode}: {map_counts[mode]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
