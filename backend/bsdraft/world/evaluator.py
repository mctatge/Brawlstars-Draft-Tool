"""Factorized offline evaluation on top of verified mechanics and map geometry.

These functions return physical measurements, not calibrated win probabilities.  They are kept
outside ``engine.scoring`` deliberately: the first world-model build cannot change a live pick's
score, ordering, or explanation breakdown until real data passes the evaluation gate in
``docs/spatial-world-model.md``.

The current kernel models a Power-11 base-attack duel at fixed positions.  Every projectile is
assumed to hit the target point unless the caller supplies a lower explicit hit fraction.  Splash
radius is retained in the source profile but is not interpreted as damage or area coverage here;
healing, shields, supers, loadouts, player policies, and moving targets likewise remain uncovered
rather than silently invented.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from .geometry import (
    attack_in_range,
    can_target_point,
    position_occupiable,
    shortest_path_distance_to_any,
    world_to_cell,
)
from .schema import BrawlerMechanics, MapGeometry, Vec2, WorldModel


MAX_TEAM_SIZE = 3


def _complete_geometry(
    world: WorldModel, map_id: int, mode: str
) -> Optional[MapGeometry]:
    """Return fully verified geometry, never a partially unknown numeric input."""
    geometry = world.map_for(map_id, mode)
    if geometry is None or any("?" in row for row in geometry.terrain):
        return None
    return geometry


@dataclass(frozen=True, slots=True)
class PositionedBrawler:
    """A covered brawler placed at a continuous world-space point."""

    brawler_id: int
    position: Vec2

    def __post_init__(self) -> None:
        if not isinstance(self.brawler_id, int) or isinstance(self.brawler_id, bool):
            raise ValueError("PositionedBrawler.brawler_id must be an integer")
        if self.brawler_id <= 0:
            raise ValueError("PositionedBrawler.brawler_id must be positive")


@dataclass(frozen=True, slots=True)
class DuelRead:
    """A static base-attack duel; positive ``edge`` favours ``first_brawler_id``."""

    first_brawler_id: int
    second_brawler_id: int
    first_can_hit: bool
    second_can_hit: bool
    first_ttk_seconds: Optional[float]
    second_ttk_seconds: Optional[float]
    edge: float


def _fraction(value: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("hit_fraction must be a number in (0, 1]") from exc
    if not math.isfinite(out) or not 0.0 < out <= 1.0:
        raise ValueError("hit_fraction must be finite and in (0, 1]")
    return out


def _positive(value: float, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive number") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return out


def shots_to_defeat(
    attacker: BrawlerMechanics,
    defender: BrawlerMechanics,
    *,
    hit_fraction: float = 1.0,
) -> int:
    """Theoretical ammo attacks needed to remove the defender's base maximum health."""
    fraction = _fraction(hit_fraction)
    damage_per_ammo = attacker.attack.damage * attacker.attack.projectiles * fraction
    if not math.isfinite(damage_per_ammo) or damage_per_ammo <= 0.0:
        raise ValueError("hit_fraction is too small for finite damage arithmetic")
    required = defender.max_health / damage_per_ammo
    if not math.isfinite(required):
        raise ValueError("hit_fraction is too small for finite shot-count arithmetic")
    return max(1, int(math.ceil(required)))


def _time_to_fire(attacker: BrawlerMechanics, attacks: int) -> float:
    """Elapsed time from the first shot until the killing shot is released.

    The initial ammo is fired at ``unload_seconds`` cadence.  Once it is exhausted, the next
    attack waits one reload interval and later attacks wait the slower of reload and unload.  It
    is a documented baseline—not a claim about every brawler's bespoke reload behavior.
    """
    if attacks <= 1:
        return 0.0
    try:
        initial = min(attacks, attacker.attack.ammo)
        elapsed = (initial - 1) * attacker.attack.unload_seconds
        remaining = attacks - initial
        if remaining:
            elapsed += attacker.attack.reload_seconds
            elapsed += (remaining - 1) * max(
                attacker.attack.reload_seconds, attacker.attack.unload_seconds
            )
    except OverflowError:
        return math.inf
    return elapsed


def theoretical_time_to_kill(
    attacker: BrawlerMechanics,
    defender: BrawlerMechanics,
    distance_world: float,
    *,
    hit_fraction: float = 1.0,
) -> float:
    """Point-target base-attack TTK, or ``math.inf`` outside the shared range band.

    Damage is applied only for an assumed direct hit on the target point.  The
    profile's splash radius does not expand reach or add damage in this model.
    """
    try:
        distance = float(distance_world)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance_world must be a finite non-negative number") from exc
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance_world must be finite and non-negative")
    attack = attacker.attack
    if not attack_in_range(attack, distance):
        return math.inf
    attacks = shots_to_defeat(attacker, defender, hit_fraction=hit_fraction)
    travel = distance / attack.projectile_speed_world_per_second
    result = _time_to_fire(attacker, attacks) + travel
    if not math.isfinite(result):
        raise ValueError("hit_fraction is too small for finite TTK arithmetic")
    return result


def _edge_from_times(first_ttk: float, second_ttk: float, temperature: float) -> float:
    if math.isinf(first_ttk) and math.isinf(second_ttk):
        return 0.0
    if math.isinf(first_ttk):
        return -1.0
    if math.isinf(second_ttk):
        return 1.0
    return math.tanh((second_ttk - first_ttk) / temperature)


def evaluate_duel(
    world: WorldModel,
    map_id: int,
    mode: str,
    first: PositionedBrawler,
    second: PositionedBrawler,
    *,
    hit_fraction: float = 1.0,
    temperature_seconds: float = 2.0,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> Optional[DuelRead]:
    """Evaluate one positioned point-target duel.

    ``None`` means the requested map/mechanics are missing or any cell in the
    map remains unknown.  Positions that a brawler's circular body cannot
    occupy are invalid queries and raise ``ValueError``.  Splash is unmodeled.
    """
    geometry = _complete_geometry(world, map_id, mode)
    first_profile = world.brawler_for(first.brawler_id)
    second_profile = world.brawler_for(second.brawler_id)
    if geometry is None or first_profile is None or second_profile is None:
        return None

    fraction = _fraction(hit_fraction)
    temperature = _positive(temperature_seconds, "temperature_seconds")
    destroyed = tuple(destroyed_walls) if destroyed_walls is not None else ()
    for label, item, profile in (
        ("first", first, first_profile),
        ("second", second, second_profile),
    ):
        if not position_occupiable(
            geometry,
            item.position,
            body_radius_world=profile.collision_radius_world,
            destroyed_walls=destroyed,
        ):
            raise ValueError(f"{label} brawler position is not occupiable")
    dx = second.position.x - first.position.x
    dy = second.position.y - first.position.y
    distance = math.hypot(dx, dy)

    first_can_hit = can_target_point(
        geometry,
        first_profile.attack,
        first.position,
        second.position,
        destroyed_walls=destroyed,
    )
    second_can_hit = can_target_point(
        geometry,
        second_profile.attack,
        second.position,
        first.position,
        destroyed_walls=destroyed,
    )
    first_ttk = (
        theoretical_time_to_kill(
            first_profile, second_profile, distance, hit_fraction=fraction
        )
        if first_can_hit
        else math.inf
    )
    second_ttk = (
        theoretical_time_to_kill(
            second_profile, first_profile, distance, hit_fraction=fraction
        )
        if second_can_hit
        else math.inf
    )
    return DuelRead(
        first_brawler_id=first.brawler_id,
        second_brawler_id=second.brawler_id,
        first_can_hit=first_can_hit,
        second_can_hit=second_can_hit,
        first_ttk_seconds=None if math.isinf(first_ttk) else first_ttk,
        second_ttk_seconds=None if math.isinf(second_ttk) else second_ttk,
        edge=_edge_from_times(first_ttk, second_ttk, temperature),
    )


def _position_key(value: PositionedBrawler) -> Tuple[int, float, float]:
    return value.brawler_id, float(value.position.x), float(value.position.y)


def _validate_team(team: Sequence[PositionedBrawler], label: str) -> Tuple[PositionedBrawler, ...]:
    values = tuple(team)
    if len(values) > MAX_TEAM_SIZE:
        raise ValueError(f"{label} has {len(values)} brawlers; maximum is {MAX_TEAM_SIZE}")
    ids = [value.brawler_id for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains a duplicate brawler id")
    return tuple(sorted(values, key=_position_key))


def _directed_team_edge(
    world: WorldModel,
    map_id: int,
    mode: str,
    first_team: Tuple[PositionedBrawler, ...],
    second_team: Tuple[PositionedBrawler, ...],
    *,
    hit_fraction: float,
    temperature_seconds: float,
    destroyed_walls: Optional[Iterable[Vec2]],
) -> Optional[float]:
    edges = []
    for first in first_team:
        for second in second_team:
            duel = evaluate_duel(
                world,
                map_id,
                mode,
                first,
                second,
                hit_fraction=hit_fraction,
                temperature_seconds=temperature_seconds,
                destroyed_walls=destroyed_walls,
            )
            if duel is None:
                return None
            edges.append(duel.edge)
    return math.fsum(edges) / len(edges) if edges else 0.0


def team_duel_edge(
    world: WorldModel,
    map_id: int,
    mode: str,
    first_team: Sequence[PositionedBrawler],
    second_team: Sequence[PositionedBrawler],
    *,
    hit_fraction: float = 1.0,
    temperature_seconds: float = 2.0,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> Optional[float]:
    """Mean cross-team duel edge, exactly order-invariant and antisymmetric.

    This is intentionally only the pairwise combat term of the planned factorization.  Empty or
    one-sided boards have no cross-team evidence and therefore return exactly zero.  Missing map
    or brawler coverage returns ``None`` rather than a neutral-looking fabricated measurement.
    """
    geometry = _complete_geometry(world, map_id, mode)
    if geometry is None:
        return None
    first = _validate_team(first_team, "first_team")
    second = _validate_team(second_team, "second_team")
    fraction = _fraction(hit_fraction)
    temperature = _positive(temperature_seconds, "temperature_seconds")
    destroyed = tuple(destroyed_walls) if destroyed_walls is not None else ()
    for label, item in (("first_team", member) for member in first):
        profile = world.brawler_for(item.brawler_id)
        if profile is None:
            return None
        if not position_occupiable(
            geometry,
            item.position,
            body_radius_world=profile.collision_radius_world,
            destroyed_walls=destroyed,
        ):
            raise ValueError(f"{label} brawler position is not occupiable")
    for label, item in (("second_team", member) for member in second):
        profile = world.brawler_for(item.brawler_id)
        if profile is None:
            return None
        if not position_occupiable(
            geometry,
            item.position,
            body_radius_world=profile.collision_radius_world,
            destroyed_walls=destroyed,
        ):
            raise ValueError(f"{label} brawler position is not occupiable")

    first_key = tuple(_position_key(item) for item in first)
    second_key = tuple(_position_key(item) for item in second)
    if first_key == second_key:
        return 0.0

    # Evaluate one canonical orientation, then attach the caller's sign.  Apart from avoiding
    # floating summation-order drift, this makes swap-negation a structural guarantee.
    if first_key < second_key:
        canonical_first, canonical_second, sign = first, second, 1.0
    else:
        canonical_first, canonical_second, sign = second, first, -1.0
    value = _directed_team_edge(
        world,
        map_id,
        mode,
        canonical_first,
        canonical_second,
        hit_fraction=fraction,
        temperature_seconds=temperature,
        destroyed_walls=destroyed,
    )
    return None if value is None else sign * value


def objective_coverage(
    world: WorldModel,
    map_id: int,
    mode: str,
    brawler_id: int,
    origin: Vec2,
    *,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> Optional[float]:
    """Fraction of objective-cell centres targetable from ``origin``.

    This is centreline point coverage, not splash or area-of-effect coverage.
    A partially unknown map yields ``None`` rather than numeric evidence.
    """
    geometry = _complete_geometry(world, map_id, mode)
    profile = world.brawler_for(brawler_id)
    if geometry is None or profile is None or not geometry.objective_cells:
        return None
    destroyed = tuple(destroyed_walls) if destroyed_walls is not None else ()
    if not position_occupiable(
        geometry,
        origin,
        body_radius_world=profile.collision_radius_world,
        destroyed_walls=destroyed,
    ):
        raise ValueError("brawler origin is not occupiable")
    covered = 0
    for cell in geometry.objective_cells:
        if can_target_point(
            geometry,
            profile.attack,
            origin,
            geometry.cell_center_world(cell),
            destroyed_walls=destroyed,
        ):
            covered += 1
    return covered / len(geometry.objective_cells)


def objective_travel_time(
    world: WorldModel,
    map_id: int,
    mode: str,
    brawler_id: int,
    origin: Vec2,
    *,
    can_cross_water: bool = False,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> Optional[float]:
    """Shortest cell-centre travel time to a verified objective, or ``None`` if uncovered.

    ``math.inf`` is a real covered result meaning every objective is unreachable.  ``None`` means
    the artifact lacks the requested map/mode, brawler, or objective geometry.  The continuous
    origin is validated for body clearance, then quantized to its containing cell centre; this v1
    measurement intentionally omits sub-cell travel.
    """
    geometry = _complete_geometry(world, map_id, mode)
    profile = world.brawler_for(brawler_id)
    if geometry is None or profile is None or not geometry.objective_cells:
        return None
    destroyed = tuple(destroyed_walls) if destroyed_walls is not None else ()
    if not position_occupiable(
        geometry,
        origin,
        body_radius_world=profile.collision_radius_world,
        can_cross_water=can_cross_water,
        destroyed_walls=destroyed,
    ):
        raise ValueError("brawler origin is not occupiable")
    distance = shortest_path_distance_to_any(
        geometry,
        world_to_cell(geometry, origin),
        geometry.objective_cells,
        body_radius_world=profile.collision_radius_world,
        can_cross_water=can_cross_water,
        destroyed_walls=destroyed,
    )
    return distance / profile.move_speed_world_per_second


__all__ = [
    "DuelRead",
    "MAX_TEAM_SIZE",
    "PositionedBrawler",
    "evaluate_duel",
    "objective_coverage",
    "objective_travel_time",
    "shots_to_defeat",
    "team_duel_edge",
    "theoretical_time_to_kill",
]
