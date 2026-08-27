"""Detect Ranked **free / "boosted" brawlers** directly from match data.

The release-notes "Free Brawler Rotation" list is *incomplete*: a brawler can be handed out free
mid-season with no announcement (observed 2026-08-25, when Nori went fully free five days after
the Season 2 flip — nothing in any release note said so). Scraping alone therefore under-reports
the free set, and a player who owns that brawler at low power — or doesn't own it — is wrongly
told they can't field it, so it never gets recommended even when it is the map's best pick.

Match data settles it unambiguously. A free brawler is handed out **fully maxed (Power 11)**, so
in Ranked *every* slot for it reads ``power: 11``. A non-free brawler always carries a tail of
players still levelling it (Power 9/10), because owners grind up from below. Measured over a
recent window the two populations don't overlap: on 2026-08-26 the four free brawlers sat at
0.00–0.07% sub-Power-11 while the most-maxed non-free brawler was at 1.2% — a wide, stable gap.

So: a brawler with enough recent Ranked volume and **essentially zero** sub-Power-11 slots is
free right now. This is the authoritative *live* signal; the hand-maintained ``ranked_boosted.json``
(see :mod:`bsdraft.collect.boosted`) stays useful as a *leading* signal for next season, which the
data can't know until games are played.

The detector is a pure function over an already-accumulated power histogram, so it costs nothing
on the serving path — :class:`bsdraft.engine.stats.DraftStats` accumulates the histogram while it
already iterates every match, computes the free set once at build time, and ships it in the stats
artifact the API loads. See ``docs/backend-architecture.md`` and the memory note
"free-brawlers-detectable-from-match-power".
"""
from __future__ import annotations

from typing import Mapping

# The power a free brawler is handed out at (fully maxed). The signal is the *absence* of the
# sub-max tail that levelling owners produce.
FIELDED_POWER = 11

# Recent-window horizon for the signal, anchored to the newest match (NOT the wall clock, so a
# stalled crawl freezes the live set instead of draining it — same rule as map_games_recent).
# 2 days: enough Ranked volume for a stable read (tens of thousands of slots/day) while still
# reacting to a fresh grant within a day, and short enough not to smear a brawler's own
# free/not-free transition.
FREE_WINDOW_DAYS = 2.0

# Gate. Below MIN_SLOTS the read is too thin to trust (small-sample brawlers can hit 0% by luck).
# A free brawler shows ~0% sub-max; the nearest non-free sits above 1%, so 0.5% is mid-gap — well
# clear of the free cluster (≤0.07%) and of the non-free floor (≥1.2%). Asserting "free" wrongly
# is the cardinal error (it recommends a brawler the player can't field), so both gates are
# deliberately conservative; a truly-free brawler clears them with a wide margin.
MIN_SLOTS = 300
MAX_SUBMAX_FRAC = 0.005


def submax_fraction(power_hist: Mapping[int, float]) -> float:
    """Fraction of a brawler's recent slots below :data:`FIELDED_POWER`. ``0.0`` for an empty
    histogram (no evidence — the caller's MIN_SLOTS gate rejects it before this matters)."""
    total = sum(power_hist.values())
    if total <= 0:
        return 0.0
    sub = sum(c for p, c in power_hist.items() if p < FIELDED_POWER)
    return sub / total


def detect_free(
    power_counts: Mapping[int, Mapping[int, float]],
    *,
    min_slots: int = MIN_SLOTS,
    max_submax_frac: float = MAX_SUBMAX_FRAC,
) -> frozenset:
    """Brawler ids that look **free right now**, from ``{brawler_id: {power: recent_slot_count}}``.

    A brawler qualifies when it has at least ``min_slots`` recent Ranked slots and at most
    ``max_submax_frac`` of them below :data:`FIELDED_POWER`. Both gates must hold — the sample
    gate keeps a lucky small sample out, the tail gate is the actual free signal.
    """
    out = set()
    for bid, hist in power_counts.items():
        total = sum(hist.values())
        if total < min_slots:
            continue
        if submax_fraction(hist) <= max_submax_frac:
            out.add(bid)
    return frozenset(out)


def free_report(power_counts: Mapping[int, Mapping[int, float]], names: Mapping[int, str]) -> str:
    """Human-readable audit of the detector over a histogram — every brawler sorted by sub-max
    fraction, with the free/not-free line marked. For ``scripts`` / CI diagnostics, not serving."""
    rows = []
    for bid, hist in power_counts.items():
        total = sum(hist.values())
        rows.append((submax_fraction(hist), total, names.get(bid, str(bid))))
    rows.sort()
    free = detect_free(power_counts)
    lines = [f"free brawlers ({len(free)}): " +
             ", ".join(sorted(names.get(b, str(b)) for b in free))]
    for frac, total, name in rows:
        tag = "FREE" if any(names.get(b, str(b)) == name for b in free) else ""
        flag = "" if total >= MIN_SLOTS else " (thin)"
        lines.append(f"  {name:<16} slots={int(total):<7} subP11={100*frac:.3f}%{flag}  {tag}")
    return "\n".join(lines)
