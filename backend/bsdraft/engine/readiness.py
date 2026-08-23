"""How far a *particular player's copy* of a brawler is from the brawler the meta number describes.

Every objective signal in :mod:`bsdraft.engine.scoring` — the map win rate, the model's marginal,
synergy, counter — describes a Power 11 brawler on a full loadout, because that is what the corpus
is: 97.3% of collected player-slots are Power 11. Scoring a Power 9 copy off that table is
extrapolation, and until now it was silent. This module prices the gap.

The output is a **deficit in win-rate points**, always >= 0, subtracted from the base score, plus a
list of :class:`Reason` chips explaining it. Every reason carries its provenance:

    measured   estimated from the match log with a placebo gate (docs/readiness.md)
    estimated  a declared prior, capped so it can never outrank a measurement
    unpriced   shown to the user, contributes exactly 0.0

That third source exists because being honest about an unknown beats pretending it is zero.
Hypercharge is the case: battle logs carry no hypercharge field, so no estimator exists on the
data in hand. It ships visible and unpriced rather than silently ignored.

Pure stdlib and safe on the serve path — the numpy-backed estimator that *produces* the constants
is :mod:`bsdraft.data.readiness_build`, which this module must never import. The constants below
are code defaults; ``data/reference/readiness.json`` overlays them per key when present, and a
missing or malformed file degrades to the defaults rather than erroring a request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from bsdraft.constants import REFERENCE_DIR

READINESS_PATH = REFERENCE_DIR / "readiness.json"

# --- provenance tags -------------------------------------------------------------------
MEASURED = "measured"
ESTIMATED = "estimated"
UNPRICED = "unpriced"

# --- the gap strings, defined once ------------------------------------------------------
# :meth:`engine.mastery.Mastery.gaps` emits these and the client echoes them back on the recommend
# request, so the wire contract and the scorer must agree on the exact text. One definition.
GAP_NO_STAR_POWER = "no star power"
GAP_NO_GADGET = "no gadget"
GAP_NO_HYPERCHARGE = "no hypercharge"

MAX_POWER = 11

# Measured power deficit, in win-rate points. Within-player Mantel-Haenszel on brawler-residualized
# outcomes over 1.43M ranked matches; placebo +0.07pp on 993k strata. See docs/readiness.md for the
# estimator, the three-estimator progression, and the conservative shipping rule (the point
# estimates were -9.4 and -5.9; these are haircut by the methodological margin).
POWER_DEFICIT: Dict[int, float] = {11: 0.0, 10: 0.040, 9: 0.075}

# Declared priors for an incomplete loadout, converted from the retired 3:2:2 investment index into
# win-rate points. UNIT is the per-share value; a star power is worth 3 shares, a gadget 2, a gear
# slot 1. These are NOT measured — the ordering invariant below pins them under any measurement so
# a guess can never outrank data.
UNIT = 0.007
MISSING_STAR_POWER = 3 * UNIT   # 0.021
MISSING_GADGET = 2 * UNIT       # 0.014
MISSING_GEAR_SLOT = 1 * UNIT    # 0.007, per empty slot
MISSING_HYPERCHARGE = 0.0       # unpriced — no estimator exists (see module docstring)

# Ranked opens the two gear slots at Power 8 and Power 10 (mirrors engine.purchases._GEAR_SLOT_POWERS).
# Used to charge only for slots the brawler can actually fill: a Power 9 copy has one slot, so
# penalising it for an empty second slot would double-charge what POWER_DEFICIT already prices.
_GEAR_SLOT_POWERS = (8, 10)

READY_CAP = 0.12        # most a readiness deficit can subtract (measured; ~1.6x the P9 effect)
ITEM_EDGE_CAP = 0.05    # most measured item quality can move a score, either direction

# Most the player's own record can move a score. Deliberately tight, and an order below the
# measured power effect, because unlike readiness this is an unvalidated product knob: a personal
# win rate is confounded with when you played, who you queued with, and meta drift since.
#
# It is also the signal that was cut on 2026-08-17 for out-driving the objective blend, so the size
# is a product decision, not just a statistical one — personalization is meant to be a nudge that
# breaks near-ties, never something that floats a sub-50% brawler into the top ten. Calibration
# check on a real account: a 4-game record at 97% raw shrinks to +8.1 points of raw edge, which
# this cap holds to +2.0. If that still proves too hot in use, lower it before widening it.
HISTORY_CAP = 0.02


@dataclass(frozen=True)
class Reason:
    """One line of the deficit, ready to render as a chip."""
    label: str
    points: float      # signed, in win-rate points; <= 0 for a deficit, 0.0 when unpriced
    source: str        # MEASURED | ESTIMATED | UNPRICED


@dataclass(frozen=True)
class Fielded:
    """What the player can actually put on the board, as opposed to what the meta table assumes.

    ``power == 0`` means *unknown*, not "power level zero" — it is the wire default for a client
    that predates the field, and the power-floor gate deliberately keeps such entries. Unknown
    power draws no penalty, and suppresses the gear term too (slot count is underivable)."""
    power: int = MAX_POWER
    has_starpower: bool = True
    has_gadget: bool = True
    n_gears: int = 2
    has_hypercharge: bool = True

    @property
    def power_known(self) -> bool:
        return self.power > 0

    @property
    def gear_slots(self) -> int:
        """Slots Ranked has actually opened at this power level."""
        if not self.power_known:
            return 0
        return sum(1 for p in _GEAR_SLOT_POWERS if self.power >= p)

    @classmethod
    def ready(cls) -> "Fielded":
        """A fully-fielded copy — the brawler the meta number already describes. Zero deficit."""
        return cls()

    @classmethod
    def from_gaps(cls, power: int, gaps: Optional[List[str]], n_gears: int = 2) -> "Fielded":
        """Build from the client-sent wire shape: a power level, the gap strings, and how many
        gears the player owns. Gaps are the only signal for star power / gadget / hypercharge —
        :meth:`Mastery.gaps` emits no gear string, so the count is passed separately."""
        g = set(gaps or ())
        return cls(
            power=int(power or 0),
            has_starpower=GAP_NO_STAR_POWER not in g,
            has_gadget=GAP_NO_GADGET not in g,
            n_gears=int(n_gears or 0),
            has_hypercharge=GAP_NO_HYPERCHARGE not in g,
        )

    @classmethod
    def from_mastery(cls, m) -> "Fielded":
        """Build from an :class:`engine.mastery.Mastery` (the home host's server-roster path)."""
        return cls(
            power=int(getattr(m, "power", 0) or 0),
            has_starpower=bool(getattr(m, "has_starpower", True)),
            has_gadget=bool(getattr(m, "has_gadget", True)),
            n_gears=len(getattr(m, "owned_gears", ()) or ()),
            has_hypercharge=bool(getattr(m, "has_hypercharge", True)),
        )


@lru_cache(maxsize=1)
def load_readiness() -> dict:
    """The measured constants artifact, or ``{}``.

    Fail-safe on purpose: a missing or malformed file degrades to the code defaults rather than
    erroring a request. The file is checked into ``data/reference/`` (not synced like the
    ``data/processed/`` artifacts), so absence means an old checkout, not a cold cache."""
    if not READINESS_PATH.exists():
        return {}
    try:
        doc = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


@lru_cache(maxsize=1)
def power_deficit_table() -> Dict[int, float]:
    """``power -> deficit``, code defaults overlaid per key by the artifact.

    Overlaid per key rather than replaced wholesale so a partial or older artifact can't blank out
    a level. Values are validated: a non-numeric, negative, or absurd entry is ignored in favour of
    the default, because a bad constant here silently distorts every personalized score."""
    table = dict(POWER_DEFICIT)
    for p in range(1, 9):                       # below the floor rides Power 9's constant
        table.setdefault(p, POWER_DEFICIT[9])
    raw = (load_readiness().get("power_deficit") or {})
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                power, val = int(k), float(v)
            except (TypeError, ValueError):
                continue
            if 1 <= power <= MAX_POWER and 0.0 <= val <= READY_CAP:
                table[power] = val
    return table


def _power_points(power: int) -> float:
    """Deficit for a power level. Unknown (0) draws nothing — see :class:`Fielded`."""
    if power <= 0:
        return 0.0
    return power_deficit_table().get(min(power, MAX_POWER), 0.0)


def readiness(f: Optional[Fielded], confidence: float = 0.0) -> Tuple[float, List[Reason]]:
    """``(deficit, reasons)`` for one candidate. Deficit is >= 0 and is *subtracted* by the caller.

    ``confidence`` is the player's own sample confidence on this brawler, and it **fades the
    deficit out**: a record built on this exact copy already contains the handicap, so charging it
    again double-counts. The caveat is a brawler you just upgraded, whose old record drags for a
    while — transient, self-correcting, and it errs toward under-recommending something you just
    invested in rather than over-recommending something you cannot field.

    Reason points are already faded and capped, so ``sum(r.points) == -deficit`` always holds.
    """
    if f is None:
        return 0.0, []

    raw: List[Tuple[str, float, str]] = []
    pp = _power_points(f.power)
    if pp > 0:
        raw.append((f"P{f.power}", pp, MEASURED))
    if not f.has_starpower:
        raw.append((GAP_NO_STAR_POWER, MISSING_STAR_POWER, ESTIMATED))
    if not f.has_gadget:
        raw.append((GAP_NO_GADGET, MISSING_GADGET, ESTIMATED))
    empty_slots = max(0, f.gear_slots - f.n_gears)
    if empty_slots > 0:
        label = "no gear" if f.n_gears == 0 else f"{empty_slots} empty gear slot"
        raw.append((label, empty_slots * MISSING_GEAR_SLOT, ESTIMATED))
    if not f.has_hypercharge:
        raw.append((GAP_NO_HYPERCHARGE, MISSING_HYPERCHARGE, UNPRICED))

    total = sum(pts for _, pts, _ in raw)
    if total <= 0:
        # Nothing priced. Unpriced reasons still surface, at exactly 0.0 points.
        return 0.0, [Reason(lbl, 0.0, src) for lbl, _, src in raw]

    fade = max(0.0, 1.0 - max(0.0, min(1.0, confidence)))
    deficit = min(READY_CAP, total) * fade
    # Distribute the capped+faded total back over the priced reasons so the chips sum to the score
    # movement the user sees. Unpriced reasons stay at 0.0 and are excluded from the scaling.
    scale = (deficit / total) if total > 0 else 0.0
    # `if pts else 0.0` keeps an unpriced reason at a clean +0.0 rather than -0.0, which compares
    # equal but serializes as "-0.0" and reads like a rounding bug in the UI.
    reasons = [Reason(lbl, (-pts * scale) if pts else 0.0, src) for lbl, pts, src in raw]
    return deficit, reasons


def item_edge(*_args, **_kwargs) -> Optional[float]:
    """Measured per-item quality, in win-rate points, signed. ``None`` = no measurement.

    Deliberately inert: the per-item win-rate table (``itemstats.json``) has never been built —
    there is no ownership-profile crawl and ``ITEMSTATS_URL`` is unset on every host — so wiring
    a real body here today would ship arithmetic nothing can exercise. The call site, the cap and
    the tests are in place so switching it on is a contained change. See docs/item-winrate.md."""
    return None


def clamp_score(v: float) -> float:
    """Keep a fused score inside [0, 1].

    Adding signed adjustments to a weighted average makes the sum unbounded, and the frontend
    renders it as a percentage — an uncapped deficit could print a negative win rate."""
    return max(0.0, min(1.0, v))


def _ordering_invariant_holds() -> bool:
    """No declared prior may outrank a measurement. Asserted in the tests, not at import."""
    priors = (MISSING_STAR_POWER, MISSING_GADGET, MISSING_GEAR_SLOT)
    measured = [v for v in power_deficit_table().values() if v > 0]
    return bool(measured) and max(priors) < min(measured)
