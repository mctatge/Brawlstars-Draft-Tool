"""Dependency-free spatial geometry for the draft-scale world model.

The map coordinate system has its origin at the top-left: ``x`` grows to the
right, ``y`` grows downward, and one grid cell spans ``cell_size_world`` world
units on each axis.  Projectile operations use continuous world coordinates;
pathfinding operates between grid-cell centres and returns world-unit distance.

Unknown terrain (``?``) always fails closed.  A destructible wall (``+``) is
open only when its cell is explicitly present in ``destroyed_walls``.
"""
from __future__ import annotations

import heapq
import math
from functools import lru_cache
from itertools import count
from typing import FrozenSet, Iterable, Optional, Sequence, Tuple

from .schema import (
    LINE_OF_SIGHT_BLOCKING_TERRAIN,
    MAX_CELLS_PER_REGION,
    MOVEMENT_BLOCKING_TERRAIN,
    AttackProfile,
    MapGeometry,
    Vec2,
)


Cell = Tuple[int, int]
_EMPTY_DESTROYED: FrozenSet[Cell] = frozenset()
_DIRECT_BLOCKERS = LINE_OF_SIGHT_BLOCKING_TERRAIN
_LOB_BLOCKERS = frozenset(("?",))
_SQRT_2 = math.sqrt(2.0)
_EPSILON = 1e-12
_NEIGHBOURS: Sequence[Cell] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


def _xy(value: Vec2) -> Tuple[float, float]:
    """Return finite numeric coordinates from a ``Vec2``-compatible value."""
    try:
        x = float(value.x)
        y = float(value.y)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("coordinate must provide numeric x and y values") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("coordinates must be finite")
    return x, y


def _cell(value: Vec2, *, label: str = "cell") -> Cell:
    x, y = _xy(value)
    if not (x.is_integer() and y.is_integer()):
        raise ValueError(f"{label} coordinates must be integers, got ({x}, {y})")
    return int(x), int(y)


def _normalise_destroyed_walls(
    destroyed_walls: Optional[Iterable[Vec2]],
) -> FrozenSet[Cell]:
    if destroyed_walls is None:
        return _EMPTY_DESTROYED
    cells = []
    for value in destroyed_walls:
        # Accept internal ``(x, y)`` keys as well as the public Vec2 form.  This
        # lets higher-level operations normalise a generator once, then reuse it.
        if isinstance(value, tuple) and len(value) == 2:
            try:
                x, y = float(value[0]), float(value[1])
            except (TypeError, ValueError) as exc:
                raise TypeError("destroyed wall coordinates must be numeric") from exc
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("destroyed wall coordinates must be finite")
            if not (x.is_integer() and y.is_integer()):
                raise ValueError("destroyed wall coordinates must be integers")
            cells.append((int(x), int(y)))
        else:
            cells.append(_cell(value, label="destroyed wall"))
    return frozenset(cells)


def in_bounds(map_geometry: MapGeometry, cell: Vec2) -> bool:
    """Whether an integer grid coordinate lies inside ``map_geometry``.

    Non-integral, non-finite, or otherwise malformed coordinates are not cells
    and therefore return ``False`` rather than being silently truncated.
    """
    try:
        x, y = _cell(cell)
    except (TypeError, ValueError):
        return False
    return 0 <= x < map_geometry.width and 0 <= y < map_geometry.height


def _world_in_bounds(map_geometry: MapGeometry, point: Vec2) -> bool:
    x, y = _xy(point)
    max_x = map_geometry.width * map_geometry.cell_size_world
    max_y = map_geometry.height * map_geometry.cell_size_world
    # The right and bottom edges belong to no cell.
    return 0.0 <= x < max_x and 0.0 <= y < max_y


def world_to_cell(map_geometry: MapGeometry, point: Vec2) -> Vec2:
    """Convert a world-space point to its containing integer cell.

    ``ValueError`` is raised outside the half-open map rectangle instead of
    clamping to an unrelated edge cell.
    """
    x, y = _xy(point)
    if not _world_in_bounds(map_geometry, point):
        raise ValueError(f"world point ({x}, {y}) is outside the map")
    size = map_geometry.cell_size_world
    return Vec2(x=int(math.floor(x / size)), y=int(math.floor(y / size)))


def cell_center(map_geometry: MapGeometry, cell: Vec2) -> Vec2:
    """Return the continuous world-space centre of an in-bounds grid cell."""
    x, y = _cell(cell)
    if not (0 <= x < map_geometry.width and 0 <= y < map_geometry.height):
        raise ValueError(f"cell ({x}, {y}) is outside the map")
    size = map_geometry.cell_size_world
    return Vec2(x=(x + 0.5) * size, y=(y + 0.5) * size)


def _validate_world_endpoint(map_geometry: MapGeometry, point: Vec2, label: str) -> None:
    x, y = _xy(point)
    if not _world_in_bounds(map_geometry, point):
        raise ValueError(f"{label} world point ({x}, {y}) is outside the map")


def _non_negative_finite(value: float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def attack_in_range(attack: AttackProfile, distance_world: float) -> bool:
    """Whether an exact centre-to-centre distance is inside ``attack``'s range band.

    This is the single range predicate shared by reachability and time-to-kill
    evaluation.  Bounds are inclusive and deliberately have no separate
    tolerance that could make an attack reachable but give it an infinite TTK.
    """
    distance = _non_negative_finite(distance_world, "distance_world")
    min_range = _non_negative_finite(attack.min_range_world, "attack.min_range_world")
    max_range = _non_negative_finite(attack.range_world, "attack.range_world")
    if max_range < min_range:
        raise ValueError("attack range must satisfy min range <= max range")
    return min_range <= distance <= max_range


def _segment_intersects_aabb(
    start: Tuple[float, float],
    end: Tuple[float, float],
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> bool:
    """Inclusive segment/AABB intersection using symmetric slab clipping.

    Endpoints are canonicalised before clipping.  Besides making the intent
    explicit, this guarantees identical floating-point operations when callers
    reverse the segment.  All inequalities are inclusive, so merely touching
    an inflated obstacle boundary counts as a collision.
    """
    if end < start:
        start, end = end, start
    x0, y0 = start
    dx, dy = end[0] - x0, end[1] - y0
    t_min, t_max = 0.0, 1.0

    for origin, delta, low, high in (
        (x0, dx, left, right),
        (y0, dy, top, bottom),
    ):
        if abs(delta) <= _EPSILON:
            if origin < low - _EPSILON or origin > high + _EPSILON:
                return False
            continue
        t1 = (low - origin) / delta
        t2 = (high - origin) / delta
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max + _EPSILON:
            return False
    return t_max >= -_EPSILON and t_min <= 1.0 + _EPSILON


def _segment_clear(
    map_geometry: MapGeometry,
    start: Vec2,
    end: Vec2,
    radius: float,
    destroyed_walls: FrozenSet[Cell],
    blockers: FrozenSet[str],
) -> bool:
    a = _xy(start)
    b = _xy(end)
    size = map_geometry.cell_size_world
    # Only cells whose (inflated) boxes can overlap the segment's bounding box are relevant.
    # The one-cell halo preserves conservative boundary contact when a bound lies exactly on a
    # cell edge, without scanning the whole map for every candidate/projectile query.
    min_x = max(0, math.floor((min(a[0], b[0]) - radius) / size) - 1)
    max_x = min(map_geometry.width - 1,
                math.floor((max(a[0], b[0]) + radius) / size) + 1)
    min_y = max(0, math.floor((min(a[1], b[1]) - radius) / size) - 1)
    max_y = min(map_geometry.height - 1,
                math.floor((max(a[1], b[1]) + radius) / size) + 1)

    for y in range(min_y, max_y + 1):
        row = map_geometry.terrain[y]
        for x in range(min_x, max_x + 1):
            terrain = row[x]
            if terrain not in blockers:
                continue
            if terrain == "+" and (x, y) in destroyed_walls:
                continue
            if _segment_intersects_aabb(
                a,
                b,
                x * size - radius,
                y * size - radius,
                (x + 1) * size + radius,
                (y + 1) * size + radius,
            ):
                return False
    return True


def projectile_clear(
    map_geometry: MapGeometry,
    start: Vec2,
    end: Vec2,
    projectile_radius_world: float = 0.0,
    *,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> bool:
    """Whether a direct projectile can traverse ``start`` to ``end``.

    Collision is continuous: the segment is tested against every blocking
    cell's axis-aligned box inflated by the projectile radius.  ``#``, intact
    ``+``, and unknown ``?`` cells block; bushes and water do not.  Boundary
    contact is conservatively considered blocked.
    """
    _validate_world_endpoint(map_geometry, start, "start")
    _validate_world_endpoint(map_geometry, end, "end")
    radius = float(projectile_radius_world)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("projectile_radius_world must be finite and non-negative")
    destroyed = _normalise_destroyed_walls(destroyed_walls)
    return _segment_clear(
        map_geometry, start, end, radius, destroyed, _DIRECT_BLOCKERS
    )


def can_target_point(
    map_geometry: MapGeometry,
    attack: AttackProfile,
    origin: Vec2,
    target: Vec2,
    *,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> bool:
    """Whether an attack's centreline can reach one exact target point.

    Range is centre-to-centre and inclusive at both the minimum and maximum.
    Direct attacks obey ordinary projectile occlusion.  Lobbed attacks pass
    over walls but still fail closed when their trajectory crosses unknown
    terrain.  This predicate intentionally does not expand the target by
    ``splash_radius_world`` or infer area-of-effect behavior; unsupported attack
    kinds raise rather than guessing mechanics.
    """
    _validate_world_endpoint(map_geometry, origin, "origin")
    _validate_world_endpoint(map_geometry, target, "target")
    ox, oy = _xy(origin)
    tx, ty = _xy(target)
    distance = math.hypot(tx - ox, ty - oy)
    if not attack_in_range(attack, distance):
        return False

    radius = float(attack.projectile_radius_world)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("attack projectile radius must be finite and non-negative")
    destroyed = _normalise_destroyed_walls(destroyed_walls)
    kind = attack.kind.strip().lower()
    if kind == "direct":
        blockers = _DIRECT_BLOCKERS
    elif kind == "lobbed":
        blockers = _LOB_BLOCKERS
    else:
        raise ValueError(f"unsupported attack kind: {attack.kind!r}")
    return _segment_clear(map_geometry, origin, target, radius, destroyed, blockers)


def can_attack(
    map_geometry: MapGeometry,
    attack: AttackProfile,
    origin: Vec2,
    target: Vec2,
    *,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> bool:
    """Compatibility name for exact point-target reachability.

    A ``True`` result means the attack centreline can reach ``target``.  It does
    not model splash propagation, splash damage falloff, or nearby targets; use
    :func:`can_target_point` when the distinction matters at the call site.
    """
    return can_target_point(
        map_geometry,
        attack,
        origin,
        target,
        destroyed_walls=destroyed_walls,
    )


def attackable_cells(
    map_geometry: MapGeometry,
    attack: AttackProfile,
    origin: Vec2,
    *,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> Tuple[Vec2, ...]:
    """Return cells whose centre points the attack centreline can reach.

    Results are row-major.  They are not splash/area-of-effect coverage: the
    attack's ``splash_radius_world`` is intentionally unused.
    """
    _validate_world_endpoint(map_geometry, origin, "origin")
    destroyed = _normalise_destroyed_walls(destroyed_walls)
    result = []
    for y in range(map_geometry.height):
        for x in range(map_geometry.width):
            cell = Vec2(x=x, y=y)
            target = cell_center(map_geometry, cell)
            if can_target_point(
                map_geometry,
                attack,
                origin,
                target,
                destroyed_walls=destroyed,
            ):
                result.append(cell)
    return tuple(result)


def _movement_blockers(can_cross_water: bool) -> FrozenSet[str]:
    if can_cross_water:
        return _DIRECT_BLOCKERS
    return MOVEMENT_BLOCKING_TERRAIN


def _point_has_clearance(
    map_geometry: MapGeometry,
    point: Tuple[float, float],
    body_radius: float,
    can_cross_water: bool,
    destroyed_walls: FrozenSet[Cell],
) -> bool:
    px, py = point
    size = map_geometry.cell_size_world
    map_right = map_geometry.width * size
    map_bottom = map_geometry.height * size
    if not (0.0 <= px < map_right and 0.0 <= py < map_bottom):
        return False
    if (
        px - body_radius < -_EPSILON
        or py - body_radius < -_EPSILON
        or px + body_radius > map_right + _EPSILON
        or py + body_radius > map_bottom + _EPSILON
    ):
        return False

    blockers = _movement_blockers(can_cross_water)
    radius_sq = body_radius * body_radius
    min_x = max(0, math.floor((px - body_radius) / size) - 1)
    max_x = min(map_geometry.width - 1, math.floor((px + body_radius) / size) + 1)
    min_y = max(0, math.floor((py - body_radius) / size) - 1)
    max_y = min(map_geometry.height - 1, math.floor((py + body_radius) / size) + 1)
    for obstacle_y in range(min_y, max_y + 1):
        row = map_geometry.terrain[obstacle_y]
        for obstacle_x in range(min_x, max_x + 1):
            obstacle = row[obstacle_x]
            if obstacle not in blockers:
                continue
            if obstacle == "+" and (obstacle_x, obstacle_y) in destroyed_walls:
                continue
            left = obstacle_x * size
            right = (obstacle_x + 1) * size
            top = obstacle_y * size
            bottom = (obstacle_y + 1) * size
            nearest_x = min(max(px, left), right)
            nearest_y = min(max(py, top), bottom)
            distance_sq = (px - nearest_x) ** 2 + (py - nearest_y) ** 2
            if distance_sq <= radius_sq + _EPSILON:
                return False
    return True


def position_occupiable(
    map_geometry: MapGeometry,
    point: Vec2,
    *,
    body_radius_world: float = 0.0,
    can_cross_water: bool = False,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> bool:
    """Whether a circular body can occupy a continuous world-space point.

    Solid, intact destructible, and unknown terrain always block.  Water also
    blocks unless ``can_cross_water`` is explicit.  A body must fit inside the
    map and clear blocking cell boxes by its radius; boundary contact with a
    blocker fails closed.
    """
    position = _xy(point)
    radius = _non_negative_finite(body_radius_world, "body_radius_world")
    destroyed = _normalise_destroyed_walls(destroyed_walls)
    return _point_has_clearance(
        map_geometry, position, radius, can_cross_water, destroyed
    )


def cell_traversable(
    map_geometry: MapGeometry,
    cell: Vec2,
    *,
    body_radius_world: float = 0.0,
    can_cross_water: bool = False,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> bool:
    """Whether a circular body may occupy an integer cell's centre."""
    try:
        key = _cell(cell)
    except (TypeError, ValueError):
        return False
    radius = _non_negative_finite(body_radius_world, "body_radius_world")
    destroyed = _normalise_destroyed_walls(destroyed_walls)
    return _cell_has_clearance(
        map_geometry, key, radius, can_cross_water, destroyed
    )


def _cell_has_clearance(
    map_geometry: MapGeometry,
    cell: Cell,
    body_radius: float,
    can_cross_water: bool,
    destroyed_walls: FrozenSet[Cell],
) -> bool:
    x, y = cell
    if not (0 <= x < map_geometry.width and 0 <= y < map_geometry.height):
        return False

    size = map_geometry.cell_size_world
    px = (x + 0.5) * size
    py = (y + 0.5) * size
    return _point_has_clearance(
        map_geometry, (px, py), body_radius, can_cross_water, destroyed_walls
    )


def _octile_distance(a: Cell, b: Cell, cell_size: float) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    diagonal = min(dx, dy)
    straight = max(dx, dy) - diagonal
    return cell_size * (_SQRT_2 * diagonal + straight)


def shortest_path_distance(
    map_geometry: MapGeometry,
    start: Vec2,
    goal: Vec2,
    *,
    body_radius_world: float = 0.0,
    can_cross_water: bool = False,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> float:
    """Return the shortest cell-centre path distance in world units.

    Eight-neighbour A* uses cardinal cost ``cell_size_world`` and diagonal cost
    ``sqrt(2) * cell_size_world``.  Diagonal steps may not pass between blocked
    orthogonal neighbours.  A body centre must also clear every blocking cell
    by ``body_radius_world``.  Water blocks unless ``can_cross_water`` is true.

    Out-of-bounds or blocked endpoints raise ``ValueError``.  Valid endpoints
    in disconnected regions return ``math.inf``.
    """
    start_cell = _cell(start, label="start")
    goal_cell = _cell(goal, label="goal")
    for label, value in (("start", start_cell), ("goal", goal_cell)):
        if not (
            0 <= value[0] < map_geometry.width
            and 0 <= value[1] < map_geometry.height
        ):
            raise ValueError(f"{label} cell {value} is outside the map")

    radius = float(body_radius_world)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("body_radius_world must be finite and non-negative")
    destroyed = _normalise_destroyed_walls(destroyed_walls)

    @lru_cache(maxsize=None)
    def traversable(value: Cell) -> bool:
        return _cell_has_clearance(
            map_geometry, value, radius, can_cross_water, destroyed
        )

    if not traversable(start_cell):
        raise ValueError(f"start cell {start_cell} is blocked")
    if not traversable(goal_cell):
        raise ValueError(f"goal cell {goal_cell} is blocked")
    if start_cell == goal_cell:
        return 0.0

    cell_size = map_geometry.cell_size_world
    sequence = count()
    g_score = {start_cell: 0.0}
    queue = [
        (
            _octile_distance(start_cell, goal_cell, cell_size),
            next(sequence),
            start_cell,
        )
    ]
    while queue:
        estimate, _, current = heapq.heappop(queue)
        current_cost = g_score.get(current)
        if current_cost is None:
            continue
        expected = current_cost + _octile_distance(current, goal_cell, cell_size)
        if estimate > expected + _EPSILON:
            # A stale heap entry superseded by a shorter route.
            continue
        if current == goal_cell:
            return current_cost

        cx, cy = current
        for dx, dy in _NEIGHBOURS:
            neighbour = (cx + dx, cy + dy)
            if not traversable(neighbour):
                continue
            diagonal = dx != 0 and dy != 0
            if diagonal and (
                not traversable((cx + dx, cy))
                or not traversable((cx, cy + dy))
            ):
                continue
            step = cell_size * (_SQRT_2 if diagonal else 1.0)
            tentative = current_cost + step
            if tentative + _EPSILON >= g_score.get(neighbour, math.inf):
                continue
            g_score[neighbour] = tentative
            heapq.heappush(
                queue,
                (
                    tentative
                    + _octile_distance(neighbour, goal_cell, cell_size),
                    next(sequence),
                    neighbour,
                ),
            )

    return math.inf


def shortest_path_distance_to_any(
    map_geometry: MapGeometry,
    start: Vec2,
    goals: Sequence[Vec2],
    *,
    body_radius_world: float = 0.0,
    can_cross_water: bool = False,
    destroyed_walls: Optional[Iterable[Vec2]] = None,
) -> float:
    """Return one shortest cell-centre path to any occupiable goal.

    A single-source Dijkstra search shares one bounded traversability cache across
    every goal.  Malformed or out-of-bounds goals raise; body-blocked goals are
    skipped, and ``math.inf`` means that no supplied goal is occupiable or
    connected.  The number of goals is capped by the artifact region bound.
    """
    start_cell = _cell(start, label="start")
    if not (
        0 <= start_cell[0] < map_geometry.width
        and 0 <= start_cell[1] < map_geometry.height
    ):
        raise ValueError(f"start cell {start_cell} is outside the map")
    try:
        goal_count = len(goals)
    except TypeError as exc:
        raise TypeError("goals must be a finite sequence of cells") from exc
    if goal_count == 0:
        raise ValueError("goals must not be empty")
    if goal_count > MAX_CELLS_PER_REGION:
        raise ValueError(
            f"goals has {goal_count} entries; maximum is {MAX_CELLS_PER_REGION}"
        )
    goal_cells = []
    seen = set()
    for index, goal in enumerate(goals):
        cell = _cell(goal, label=f"goals[{index}]")
        if not (
            0 <= cell[0] < map_geometry.width
            and 0 <= cell[1] < map_geometry.height
        ):
            raise ValueError(f"goal cell {cell} is outside the map")
        if cell not in seen:
            seen.add(cell)
            goal_cells.append(cell)

    radius = _non_negative_finite(body_radius_world, "body_radius_world")
    destroyed = _normalise_destroyed_walls(destroyed_walls)

    @lru_cache(maxsize=None)
    def traversable(value: Cell) -> bool:
        return _cell_has_clearance(
            map_geometry, value, radius, can_cross_water, destroyed
        )

    if not traversable(start_cell):
        raise ValueError(f"start cell {start_cell} is blocked")
    targets = frozenset(goal for goal in goal_cells if traversable(goal))
    if not targets:
        return math.inf
    if start_cell in targets:
        return 0.0

    cell_size = map_geometry.cell_size_world
    sequence = count()
    distances = {start_cell: 0.0}
    queue = [(0.0, next(sequence), start_cell)]
    while queue:
        queued_cost, _, current = heapq.heappop(queue)
        current_cost = distances.get(current)
        if current_cost is None or queued_cost > current_cost + _EPSILON:
            continue
        if current in targets:
            return current_cost

        cx, cy = current
        for dx, dy in _NEIGHBOURS:
            neighbour = (cx + dx, cy + dy)
            if not traversable(neighbour):
                continue
            diagonal = dx != 0 and dy != 0
            if diagonal and (
                not traversable((cx + dx, cy))
                or not traversable((cx, cy + dy))
            ):
                continue
            step = cell_size * (_SQRT_2 if diagonal else 1.0)
            tentative = current_cost + step
            if tentative + _EPSILON >= distances.get(neighbour, math.inf):
                continue
            distances[neighbour] = tentative
            heapq.heappush(queue, (tentative, next(sequence), neighbour))
    return math.inf


__all__ = [
    "attack_in_range",
    "attackable_cells",
    "can_attack",
    "can_target_point",
    "cell_center",
    "cell_traversable",
    "in_bounds",
    "position_occupiable",
    "projectile_clear",
    "shortest_path_distance",
    "shortest_path_distance_to_any",
    "world_to_cell",
]
