"""Ranked-map rotation helpers shared by the API and world-model tooling.

``reference.load_ranked_maps()`` returns the stable map vocabulary: every catalog map in a
supported Ranked mode, whatever the upstream ``disabled`` flag says.  The board shows the much
smaller live rotation inferred from collected Ranked games.  This module owns that selection
logic so authoring tools map the same map set the product exposes.
"""
from __future__ import annotations

from typing import Iterable, Tuple

from bsdraft.data.reference import GameMap

# Share of its mode's busiest map that a map must reach to count as "in the current rotation".
# The observed gap between a live map and a retired one is ~20x, so anything from ~0.1 to ~0.5
# separates them; 0.15 leans toward keeping a map that is merely quiet.
MIN_SHARE_OF_MODE_LEADER = 0.15

# The recent-window (rotation-liveness) cut, used for a mode only when its window leader clears
# the trust floor. A retired map draws exactly ZERO ranked games post-flip (measured 2026-08-25:
# 1-2 stray rows per map per 6 days), so the share term buys no readmission protection — it only
# delays a just-added map's appearance while it ramps from zero. 0.02 of a ~5k steady-state
# leader is ~100 games, a few hours of crawl; the 25-game floor is what actually separates live
# from stray when the share term rounds low. The trust floor exists because a thin window can't
# tell live from straggler: right after a >RECENT_WINDOW_DAYS outage the window restarts from
# empty, and while per-map counts sit in the Poisson noise band (~1-30) a leader-relative cut
# would prune live maps that a fuller window keeps. At the floor itself (leader = 500) same-rate
# live maps sit ~440+, far above both thresholds, so the gate never bites in a healthy window.
RECENT_MIN_SHARE_OF_MODE_LEADER = 0.02
RECENT_MIN_GAMES = 25.0
RECENT_TRUST_MIN_LEADER = 500.0

# Below the trust floor a mode starts from the cumulative cut (after an outage that is exactly
# the right guess: the pre-outage rotation), but the window still carries one-directional
# evidence the cumulative cut must not override. The cumulative stats lag a rotation flip by
# days to weeks, and a thin window is not only a post-outage state: 2026-09-23 on, the crawl's
# fresh Ranked games fell ~10x (every mode's 3-day leader ~130), and the cumulative-only
# fallback hid a map live for 9 days (Quick Travel: 116 games in the window, leader 127) while
# still serving five maps that had drawn zero games in 3 days.
#   * Admit: RECENT_MIN_GAMES in the window is proof of play at any volume — a retired map
#     draws 1-2 stray rows per 6 days, never 25.
#   * Prune: once the mode's window holds RECENT_PRUNE_MIN_MODE_GAMES, a map the cumulative cut
#     keeps that shows at most RECENT_ABSENT_MAX has left. Gated on the mode's TOTAL, not its
#     leader: a full rotation flip splits the window between outgoing and incoming maps and
#     halves the leader while the total holds. At 200 games over even an 8-map mid-flip pool, a
#     live map expects ~25 (a 0.35x-quiet one in a 4-map pool ~20): P(Poisson(20) <= 2) ~ 5e-7.
#     Under that total the window is too empty to call absence (post-outage refill), so nothing
#     is pruned.
RECENT_PRUNE_MIN_MODE_GAMES = 200.0
RECENT_ABSENT_MAX = 2.0


def select_current_ranked_maps(
    all_maps: Iterable[GameMap],
    stats,
) -> Tuple[GameMap, ...]:
    """Filter a ranked-map catalog to the currently observed Ranked rotation.

    ``stats`` is intentionally duck-typed: the API passes ``DraftStats`` while
    tests and scripts often pass a tiny object with ``map_games`` and
    ``map_games_recent`` dicts.  If there is no usable signal, return the maps the
    upstream catalog does not flag disabled — too many beats none, and the full vocab
    (every map ever in a ranked mode, ~440) would bury the rotation.
    """
    maps = tuple(all_maps)
    fallback = tuple(m for m in maps if not m.catalog_disabled) or maps
    if not maps or stats is None:
        return fallback

    totals = getattr(stats, "map_games", {}) or {}
    recent_totals = getattr(stats, "map_games_recent", {}) or {}
    games = {m.id: totals.get(m.id, 0) for m in maps}
    recent = {m.id: recent_totals.get(m.id, 0) for m in maps}

    by_mode: dict[str, list[GameMap]] = {}
    for map_ref in maps:  # caller order is preserved within each mode
        by_mode.setdefault(map_ref.mode, []).append(map_ref)

    played: list[GameMap] = []
    for mode_maps in by_mode.values():
        top_recent = max(recent[m.id] for m in mode_maps)
        if top_recent >= RECENT_TRUST_MIN_LEADER:
            threshold = max(
                RECENT_MIN_GAMES,
                RECENT_MIN_SHARE_OF_MODE_LEADER * top_recent,
            )
            played.extend(m for m in mode_maps if recent[m.id] >= threshold)
            continue

        top = max(games[m.id] for m in mode_maps)
        threshold = max(1, MIN_SHARE_OF_MODE_LEADER * top)
        can_prune = sum(recent[m.id] for m in mode_maps) >= RECENT_PRUNE_MIN_MODE_GAMES
        played.extend(
            m for m in mode_maps
            if recent[m.id] >= RECENT_MIN_GAMES
            or (games[m.id] >= threshold
                and not (can_prune and recent[m.id] <= RECENT_ABSENT_MAX))
        )

    return tuple(played or fallback)
