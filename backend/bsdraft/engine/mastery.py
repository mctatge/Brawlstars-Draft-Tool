"""Player roster & mastery.

Parses the live `/players/{tag}` roster into per-brawler ownership, and carries two views of it.

``score`` is an **investment index** in [0,1] — owned star powers / gadgets / gears (``build``)
plus personal trophies (``comfort``). It is DISPLAY ONLY: nothing multiplies it into a pick score
any more, because a unitless 0..1 index blended with win-rate-shaped signals overstates a built
brawler by tens of points. Power level and hypercharge stay out of it.

``fielded`` is the view the scorer actually uses — power level, owned loadout and gear count,
priced in win-rate points by :mod:`bsdraft.engine.readiness`. That is where power and hypercharge
are accounted for: power as a measured deficit, hypercharge as an explicitly *unpriced* reason,
since battle logs never record it. See docs/readiness.md.

Buffies are left out too. The `/players/{tag}` roster does carry a per-brawler
`buffies: {"gadget": bool, "starPower": bool, "hyperCharge": bool}` object, but its `True` flags
only tell us which buffies the player *owns* — never how many *exist* for that brawler. A brawler
with no buffie released (e.g. R-T) returns all-`False`, which is indistinguishable from one whose
buffies you simply haven't unlocked yet. With no reliable slot total, a "missing buffie" signal
misfires on every brawler that has none (verified against maxed top-100 rosters), so buffies are
left out of both the build score and the loadout gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from bsdraft.engine.readiness import (
    GAP_NO_GADGET,
    GAP_NO_HYPERCHARGE,
    GAP_NO_STAR_POWER,
    Fielded,
)


@dataclass
class Mastery:
    brawler_id: int
    power: int
    rank: int
    trophies: int
    highest_trophies: int
    has_starpower: bool
    has_gadget: bool
    has_gears: bool
    has_hypercharge: bool
    # The *specific* owned items (ids), retained so the UI can suggest only what the player can
    # actually equip on their own pick. Gears carry names+levels since no catalog lists them.
    # These are also the raw ingredient for the planned single-item-owner win-rate inference.
    owned_star_powers: Tuple[int, ...] = ()
    owned_gadgets: Tuple[int, ...] = ()
    owned_gears: Tuple[dict, ...] = ()  # each {"id", "name", "level"}

    @property
    def comfort(self) -> float:  # how much the player has played/succeeded on it
        return min(1.0, self.highest_trophies / 1000.0)

    @property
    def build(self) -> float:  # how fully built the brawler is, over the loadout the API can measure
        # Star power weighted 1.5× a gadget or gear — the original 3:2:2 split, with the buffie term
        # dropped (see the module docstring) and the rest renormalized to reach 1.0 when fully built.
        return (
            3 * (1.0 if self.has_starpower else 0.0)
            + 2 * (1.0 if self.has_gadget else 0.0)
            + 2 * (1.0 if self.has_gears else 0.0)
        ) / 7.0

    @property
    def score(self) -> float:
        # An *investment* index in [0,1] — DISPLAY ONLY as of the readiness change. Nothing
        # multiplies this into a pick score any more: it is a unitless fraction, and blending it
        # with win-rate-shaped signals is exactly the bug `engine/scoring.py` now avoids. What the
        # score path uses instead is :meth:`fielded`, priced in win-rate points by
        # :mod:`bsdraft.engine.readiness`.
        #
        # Power stays out of *this* number because it ranks investment among brawlers that all
        # clear the floor. It emphatically does NOT mean power is free: within-player measurement
        # puts a Power 9 copy ~9 points of win rate below a maxed one (docs/readiness.md), which is
        # what `readiness` now charges. An earlier version of this comment claimed "the dataset's
        # win rates already fold real power in" — 97.3% of the corpus is Power 11, so they do not.
        return max(0.0, min(1.0, 0.60 * self.build + 0.40 * self.comfort))

    def fielded(self) -> "Fielded":
        """The readiness view: what this copy can actually put on the board. Used by the scorer on
        the home host, where the server roster hands real :class:`Mastery` objects to scoring."""
        return Fielded.from_mastery(self)

    def gaps(self) -> List[str]:
        # Loadout gaps only. Power isn't listed here because an under-floor brawler is unfieldable
        # and filtered out before scoring — but the Power 9-10 window that *survives* the floor is
        # priced by `readiness`, and surfaced there as its own reason rather than as a gap string.
        out: List[str] = []
        if not self.has_starpower:
            out.append(GAP_NO_STAR_POWER)
        if not self.has_gadget:
            out.append(GAP_NO_GADGET)
        if not self.has_hypercharge:
            out.append(GAP_NO_HYPERCHARGE)
        return out


def _ids(items) -> Tuple[int, ...]:
    return tuple(i["id"] for i in items if isinstance(i, dict) and i.get("id") is not None)


def _gears(items) -> Tuple[dict, ...]:
    out = []
    for g in items or []:
        if isinstance(g, dict) and g.get("id") is not None:
            out.append({"id": g["id"], "name": g.get("name", ""), "level": g.get("level", 0)})
    return tuple(out)


def parse_roster(player: dict) -> Dict[int, Mastery]:
    roster: Dict[int, Mastery] = {}
    for b in player.get("brawlers", []):
        star_powers = b.get("starPowers") or []
        gadgets = b.get("gadgets") or []
        gears = b.get("gears") or []
        roster[b["id"]] = Mastery(
            brawler_id=b["id"],
            power=b.get("power", 0),
            rank=b.get("rank", 0),
            trophies=b.get("trophies", 0),
            highest_trophies=b.get("highestTrophies", 0),
            has_starpower=bool(star_powers),
            has_gadget=bool(gadgets),
            has_gears=bool(gears),
            has_hypercharge=bool(b.get("hyperCharges")),
            owned_star_powers=_ids(star_powers),
            owned_gadgets=_ids(gadgets),
            owned_gears=_gears(gears),
        )
    return roster


async def fetch_roster(client, tag: str) -> Tuple[Dict[int, Mastery], str]:
    player = await client.get_player(tag)
    return parse_roster(player), player.get("name", "")
