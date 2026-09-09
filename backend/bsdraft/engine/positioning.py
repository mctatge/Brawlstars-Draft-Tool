"""Actionable, rule-based opening assignments for a finished draft.

The match corpus records draft -> outcome, not movement, lanes, or timings.  Exact map
directions therefore live in a small curated registry keyed by stable map id, while every
other map falls back to mode-level jobs that do not pretend to know its geometry.

Assignments are entirely curated/rule-based: the blueprint decides where to start and what job
to do, and an exact map matchup may add one coordinated rotation. Aggregate head-to-head results
remain in the measured panel; they are team outcomes and cannot establish lane causality.

Keeping this outside :mod:`gameplan` makes the map registry easy to extend without turning
the statistical read in that module into a store of hand-authored map facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Mapping, Sequence, Tuple

from bsdraft.constants import TEAM_SIZE


@dataclass(frozen=True)
class PositionSlot:
    key: str
    label: str
    job: str
    class_fit: Mapping[str, float]


@dataclass(frozen=True)
class Blueprint:
    mode: str
    formation: str
    slots: Tuple[PositionSlot, PositionSlot, PositionSlot]


@dataclass(frozen=True)
class MatchupRule:
    """A hand-authored swap call for one exact map, ally, and enemy combination."""
    weight: float
    tracker_slot: str
    backfill_slot: str
    support_slot: str
    tracker_template: str
    backfill_template: str


def _fit(**scores: float) -> Mapping[str, float]:
    """Write class-fit tables with Python-safe keys (``Damage_Dealer`` -> display name)."""
    return {name.replace("_", " "): score for name, score in scores.items()}


_CLASS_EXECUTION = {
    "Tank": "Own the front edge with your health, but reset before you hand over a free Super.",
    "Assassin": "Stay hidden until they spend ammo, then enter and leave instead of anchoring in open ground.",
    "Marksman": "Hold the longest safe sightline and move up only after your damage forces them back.",
    "Controller": "Spend ammo on the route into the objective, not on targets already leaving it.",
    "Artillery": "Keep hard cover between you and their lane and deny the landing area before they step in.",
    "Support": "Play one retreat path behind the teammate you enable so one engage cannot catch both of you.",
    "Damage Dealer": "Take the first trade from cover, then use your burst to force them off the objective.",
    "Unclassified": "Play from cover, preserve a retreat path, and move only after your team wins the first trade.",
}

# Hand-authored matchup assignments are part of a map's tactical profile, not learned 1v1 claims.
# Parallel Plays is the motivating case: R-T can mark Bibi's approach at range and contain her so
# Sprout is not isolated into the tank. Only one tracker is ever emitted, because three simultaneous
# "follow your matchup" calls would destroy the formation instead of coordinating it.
_CURATED_TRACKS = {
    (15000293, "Hot Zone", 16000066, 16000026): MatchupRule(
        weight=1.0,  # R-T -> Bibi
        tracker_slot="home_zone",
        backfill_slot="away_zone",
        support_slot="flex",
        tracker_template=(
            "Poke {enemy} before she reaches the zone wall. If she commits far, swap with "
            "{backfill} only on a full reset or respawn; never trail her through center mid-fight. "
            "Play ring-around-cover and do not overlap both halves into one bat swing—call help "
            "instead of yielding the circle."
        ),
        backfill_template=(
            "On {tracker}'s {enemy} swap call, hold far until {tracker} starts crossing, then "
            "rotate into near behind the handoff. {support} stays on far support."
        ),
    ),
}

# The official seven-class label is too coarse for a few map jobs. R-T's long-range mark plus
# close-range split makes it the reliable home anchor on Parallel Plays; encode that tactical
# fact here instead of pretending every Damage Dealer fills the slot identically.
_CURATED_SLOT_BONUS = {
    (15000293, "Hot Zone", 16000066, "home_zone"): 0.50,  # R-T
}

# A profiled map can go beyond class-level prose when one brawler has a well-defined job there.
# These replace (rather than append to) the generic class sentence, keeping the three cards concise.
_CURATED_JOB_COPY = {
    (15000293, "Hot Zone", 16000066, "home_zone"): (
        "Hold the near bottom-left zone and mark its near-side entrance. Use the wall to protect "
        "your body; staying alive on the circle matters more than chasing a low target."
    ),
    (15000293, "Hot Zone", 16000002, "away_zone"): (
        "Enter the far top-right zone through right-side cover and screen for the center support. "
        "Contest from the near edge; retreat toward center instead of feeding on their side."
    ),
    (15000293, "Hot Zone", 16000037, "flex"): (
        "Set up behind the center fence and lob onto the far top-right zone from a separate angle. "
        "Stay within the far player's peel; rotate near only if the anchor is doubled or dies."
    ),
}

_MODE_BLUEPRINTS: Dict[str, Blueprint] = {
    "Gem Grab": Blueprint(
        mode="Gem Grab",
        formation="Open 1–1–1: one brawler owns the mine, one creates side pressure, and one keeps the carrier's exit open.",
        slots=(
            PositionSlot("mine", "CENTER · MINE", "Start near the mine, collect only when the lane is safe, and retreat first once you hold most gems or the countdown begins.",
                         _fit(Controller=8, Support=7, Marksman=6, Damage_Dealer=5, Artillery=5, Tank=3, Assassin=2)),
            PositionSlot("pressure_lane", "PRESSURE LANE", "Take the lane that gives you cover into mid and pinch the enemy mine player after winning it.",
                         _fit(Tank=8, Assassin=8, Damage_Dealer=6, Controller=6, Marksman=4, Artillery=4, Support=3)),
            PositionSlot("cover_lane", "CARRIER COVER", "Hold the other lane and stay available to peel; do not cross the mine while your carrier has no exit.",
                         _fit(Support=8, Marksman=7, Controller=6, Artillery=6, Damage_Dealer=5, Tank=4, Assassin=4)),
        ),
    ),
    "Brawl Ball": Blueprint(
        mode="Brawl Ball",
        formation="Open with a ball winner, a scoring angle, and a last defender; only the last defender moves up after your team wins control.",
        slots=(
            PositionSlot("ball_lane", "BALL LANE", "Contest the first ball touch without spending your escape, then carry only after an enemy is forced back.",
                         _fit(Tank=8, Assassin=8, Controller=6, Damage_Dealer=6, Support=5, Marksman=3, Artillery=3)),
            PositionSlot("scoring_angle", "SCORING ANGLE", "Take the safer side route, pressure the goal-side defender, and become the pass option rather than stacking on the ball.",
                         _fit(Damage_Dealer=8, Controller=7, Assassin=7, Tank=6, Marksman=5, Artillery=5, Support=4)),
            PositionSlot("last_defender", "LAST DEFENDER", "Open goal-side of your teammates and stay behind the ball until the first enemy is down or forced to heal.",
                         _fit(Controller=8, Marksman=7, Damage_Dealer=6, Artillery=6, Support=6, Tank=5, Assassin=3)),
        ),
    ),
    "Knockout": Blueprint(
        mode="Knockout",
        formation="Build a cross-fire without isolating: one safe sightline, one pressure angle, and one teammate close enough to trade either lane.",
        slots=(
            PositionSlot("safe_sightline", "SAFE SIGHTLINE", "Take the route with the cleanest retreat and preserve your life while you establish the first line of fire.",
                         _fit(Marksman=8, Controller=7, Artillery=7, Damage_Dealer=6, Support=5, Tank=3, Assassin=3)),
            PositionSlot("crossfire", "CROSS-FIRE ANGLE", "Open on a separate angle, pressure sideways across their cover, and collapse only after somebody is low.",
                         _fit(Assassin=8, Marksman=7, Damage_Dealer=7, Controller=6, Artillery=6, Tank=5, Support=4)),
            PositionSlot("trade", "TRADE / PEEL", "Stay one rotation from both teammates, enter trade range before first contact, and punish the enemy who commits first.",
                         _fit(Support=8, Controller=7, Damage_Dealer=6, Tank=6, Marksman=5, Assassin=5, Artillery=4)),
        ),
    ),
    "Heist": Blueprint(
        mode="Heist",
        formation="Assign a first responder before anyone attacks: one safe-side defender, one damage lane, and one flex who rotates on the first won or lost lane.",
        slots=(
            PositionSlot("safe_defense", "SAFE DEFENSE", "Start on the route with the shortest path back to your safe and clear the first diver before joining offense.",
                         _fit(Artillery=8, Controller=8, Damage_Dealer=7, Marksman=6, Support=5, Tank=4, Assassin=3)),
            PositionSlot("damage_lane", "SAFE-DAMAGE LANE", "Take the route that can turn one won fight into safe damage; ignore low enemies when the safe is free.",
                         _fit(Damage_Dealer=8, Marksman=8, Tank=7, Assassin=7, Controller=5, Artillery=5, Support=3)),
            PositionSlot("flex", "FLEX ROTATION", "Open between the two routes, help secure the first lane, then rotate back before an enemy reaches your safe.",
                         _fit(Support=8, Assassin=7, Controller=7, Damage_Dealer=6, Tank=6, Marksman=5, Artillery=5)),
        ),
    ),
    "Hot Zone": Blueprint(
        mode="Hot Zone",
        formation="Start with one zone anchor, one off-angle, and one flex; rotate because a zone is being lost, not because a kill looks available.",
        slots=(
            PositionSlot("zone_anchor", "ZONE ANCHOR", "Open on the safest edge of your assigned uncaptured zone and keep capture progress running while your teammates take angles.",
                         _fit(Controller=8, Tank=7, Support=6, Damage_Dealer=6, Artillery=5, Marksman=5, Assassin=3)),
            PositionSlot("off_angle", "OFF-ANGLE", "Take a different route into the objective and force enemies to turn away from your anchor before stepping in.",
                         _fit(Artillery=8, Marksman=7, Assassin=7, Damage_Dealer=6, Controller=6, Tank=5, Support=3)),
            PositionSlot("flex", "FLEX ROTATION", "Start close enough to help either fight and turn toward the objective that is losing control first.",
                         _fit(Support=8, Assassin=7, Controller=7, Damage_Dealer=6, Tank=6, Marksman=5, Artillery=5)),
        ),
    ),
    "Bounty": Blueprint(
        mode="Bounty",
        formation="Open 1–1–1 with center control, a safe lane, and a pressure angle; keep every retreat inside a teammate's firing line.",
        slots=(
            PositionSlot("center_control", "CENTER CONTROL", "Secure the center without spending your escape, then hold the route that protects both side lanes.",
                         _fit(Controller=8, Marksman=7, Support=6, Damage_Dealer=6, Artillery=5, Tank=4, Assassin=3)),
            PositionSlot("safe_lane", "SAFE LANE", "Take the longest safe sightline, win damage trades, and play behind teammates once your bounty is high.",
                         _fit(Marksman=8, Artillery=8, Controller=7, Damage_Dealer=6, Support=5, Tank=3, Assassin=3)),
            PositionSlot("pressure_angle", "PRESSURE ANGLE", "Use the separate angle to pinch cover and finish low targets; do not chase beyond your teammates' trade range.",
                         _fit(Assassin=8, Damage_Dealer=7, Marksman=7, Controller=6, Tank=6, Artillery=5, Support=4)),
        ),
    ),
}


# Exact directions are relative to the player's normal in-game view (their team spawning at the
# bottom).  The mode guard matters: RecommendRequest accepts map_id and mode independently, and a
# malformed pair must fall back rather than emit Hot Zone geometry in another mode.
_MAP_BLUEPRINTS: Dict[Tuple[int, str], Blueprint] = {
    (15000293, "Hot Zone"): Blueprint(
        mode="Hot Zone",
        formation=(
            "Start 1–2: one anchor holds the near bottom-left zone, one player enters the far "
            "top-right zone, and one supports far from behind the center fence. Once near is "
            "complete, collapse far—unless the enemy can finish near before you win far; then "
            "the anchor stalls."
        ),
        slots=(
            PositionSlot("home_zone", "BOTTOM-LEFT · NEAR ZONE", "Open on the near bottom-left zone and hold its near-side entrance. Staying alive near the circle matters more than chasing a low target.",
                         _fit(Controller=8, Damage_Dealer=7, Artillery=6, Marksman=5, Support=4, Tank=3, Assassin=2)),
            PositionSlot("away_zone", "TOP-RIGHT · FAR ZONE", "Take the right route into the far top-right zone and contest from its near edge. Retreat through center rather than feeding on their side.",
                         _fit(Tank=8, Assassin=8, Controller=7, Damage_Dealer=6, Artillery=5, Marksman=4, Support=3)),
            PositionSlot("flex", "CENTER FENCE · SUPPORT FAR", "Set up behind the center fence and pressure the far zone from a separate angle. Rotate near only if the anchor dies or gets a 2v1; do not ping-pong between even fights.",
                         _fit(Support=8, Assassin=7, Controller=6, Damage_Dealer=5, Tank=5, Marksman=4, Artillery=4)),
        ),
    ),
}


def _slot_score(map_id: int, mode: str, slot: PositionSlot, ally: dict) -> float:
    base = slot.class_fit.get(ally["cls"], 1.0)
    curated = _CURATED_SLOT_BONUS.get((map_id, mode, ally["id"], slot.key), 0.0)
    return base + curated


def _assign_slots(map_id: int, mode: str, blueprint: Blueprint,
                  allies: Sequence[dict]) -> List[Tuple[dict, PositionSlot]]:
    """Maximum-score slot assignment, stable under request/team ordering."""
    ordered = sorted(allies, key=lambda a: (a["id"], a["name"]))
    best = None
    for perm in permutations(ordered, len(blueprint.slots)):
        score = sum(_slot_score(map_id, mode, slot, ally)
                    for slot, ally in zip(blueprint.slots, perm))
        # Prefer lower stable ids in earlier slots on a true tie; never inherit array order.
        key = (round(score, 9), tuple(-a["id"] for a in perm))
        if best is None or key > best[0]:
            best = (key, perm)
    return list(zip(best[1], blueprint.slots)) if best is not None else []


def _primary_track(map_id: int, mode: str, allies: Sequence[dict],
                   enemies: Sequence[dict]) -> Dict[int, dict]:
    """Return at most one exact, hand-authored enemy rotation for this map."""
    if len(allies) != TEAM_SIZE or len(enemies) != TEAM_SIZE:
        return {}
    ally_by_id = {a["id"]: a for a in allies}
    enemy_by_id = {e["id"]: e for e in enemies}
    curated = [
        (rule, ally_by_id[our_id], enemy_by_id[enemy_id])
        for (mid, mmode, our_id, enemy_id), rule in _CURATED_TRACKS.items()
        if mid == map_id and mmode == mode
        and our_id in ally_by_id and enemy_id in enemy_by_id
    ]
    if curated:
        rule, ally, enemy = max(
            curated, key=lambda row: (row[0].weight, -row[1]["id"], -row[2]["id"])
        )
        return {ally["id"]: {
            "enemy": enemy,
            "rule": rule,
        }}
    return {}


def opening_plan(*, map_id: int, mode: str, allies: Sequence[dict],
                 enemies: Sequence[dict]) -> Tuple[str, List[dict]]:
    """Return ``(formation, assignments)`` for a complete allied draft.

    Waiting for all three allies prevents jobs from visibly reshuffling while the draft is still
    being entered.  Blind pick still gets the full positional opening; it simply has no named
    matchup adjustments because there are no revealed enemies.
    """
    allies = list(allies[:TEAM_SIZE])
    enemies = list(enemies[:TEAM_SIZE])
    if len(allies) != TEAM_SIZE:
        return "", []
    blueprint = _MAP_BLUEPRINTS.get((map_id, mode)) or _MODE_BLUEPRINTS.get(mode)
    if blueprint is None:
        return "", []

    assignments = []
    slotted = _assign_slots(map_id, mode, blueprint, allies)
    targets = _primary_track(map_id, mode, allies, enemies)
    tracker_index = next((i for i, (ally, _) in enumerate(slotted) if ally["id"] in targets), None)
    backfill = support = tracker = tracked_enemy = None
    if tracker_index is not None:
        tracker = slotted[tracker_index][0]
        target = targets[tracker["id"]]
        rule = target["rule"]
        by_slot = {slot.key: ally for ally, slot in slotted}
        tracker_slot = slotted[tracker_index][1].key
        if (tracker_slot == rule.tracker_slot
                and rule.backfill_slot in by_slot and rule.support_slot in by_slot):
            backfill = by_slot[rule.backfill_slot]
            support = by_slot[rule.support_slot]
            tracked_enemy = target["enemy"]
        else:
            # A future class/blueprint change must not reinterpret an exact rotation on the fly.
            targets = {}
    for ally, slot in slotted:
        target = targets.get(ally["id"])
        tracks = None
        adjust = ""
        if target is not None:
            enemy = target["enemy"]
            tracks = enemy["name"]
            adjust = target["rule"].tracker_template.format(
                enemy=enemy["name"], backfill=backfill["name"])
        elif backfill is not None and ally["id"] == backfill["id"]:
            tracker_target = targets[tracker["id"]]
            adjust = tracker_target["rule"].backfill_template.format(
                tracker=tracker["name"], enemy=tracked_enemy["name"], support=support["name"])
        job = _CURATED_JOB_COPY.get((map_id, mode, ally["id"], slot.key))
        if job is None:
            execution = _CLASS_EXECUTION.get(ally["cls"], _CLASS_EXECUTION["Unclassified"])
            job = f"{slot.job} {execution}"
        assignments.append({
            "name": ally["name"],
            "cls": ally["cls"],
            "start": slot.key,
            "position": slot.label,
            "job": job,
            "tracks": tracks,
            "adjust": adjust,
        })
    return blueprint.formation, assignments
