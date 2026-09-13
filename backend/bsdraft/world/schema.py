"""Validated, immutable input schema for the spatial world model.

The world artifact is an *atomic game snapshot*: map collision geometry and brawler
mechanics carry one schema version, game build, timestamp, and Power-11 baseline.  Keeping
the two halves together prevents a map snapshot from being silently combined with mechanics
from another balance patch.

This module is deliberately stdlib-only.  Render's serving image must not need Pillow,
OpenCV, Shapely, torch, or another heavyweight dependency merely to load the artifact.  Any
image interpretation belongs in an offline compiler; serving consumes the semantic grid.

``WorldModel.from_payload`` is the strict boundary used by builders and tests: malformed
input raises :class:`ValueError` with a field path.  ``load_world_model`` is the serving
boundary: missing, oversized, corrupt, or unsupported artifacts log and degrade to ``None``.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

from bsdraft.constants import RANKED_MODES, REFERENCE_DIR

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
POWER_LEVEL = 11
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
DEFAULT_WORLD_MODEL_PATH = REFERENCE_DIR / "world_model.json"

# A compact, auditable semantic layer.  ``?`` is intentionally valid: an extractor can retain
# an uncertain cell without pretending it is open.  Geometry consumers must treat it as blocked.
TERRAIN_ALPHABET = frozenset(".#+~b?")
MOVEMENT_BLOCKING_TERRAIN = frozenset("#+~?")
LINE_OF_SIGHT_BLOCKING_TERRAIN = frozenset("#+?")

# V1 reasons over a 23x35 grid for the first verified image transform.  A 64-cell side and
# 4,096-cell map leave generous room for alternate crops without permitting image-resolution
# grids whose all-cell reach queries become quadratic CPU work.  The aggregate/region limits are
# likewise sized for the active Ranked catalog, not an unbounded geometry database.  The minimum
# world-cell scale keeps the kernel's conservative 1e-12 floating tolerance dimensionally small.
MAX_MAPS = 256
MAX_BRAWLERS = 512
MAX_GRID_DIMENSION = 64
MAX_GRID_CELLS = 4_096
MAX_TOTAL_GRID_CELLS = 131_072
MAX_CELLS_PER_REGION = 512
MIN_CELL_SIZE_WORLD = 1e-3
MAX_IMAGE_DIMENSION_PX = 65_536
MAX_TEXT_LENGTH = 2_048

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _error(path: str, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def _reject_duplicate_json_fields(pairs) -> dict:
    """Build one decoded JSON object while rejecting ambiguous repeated field names."""
    out = {}
    for key, value in pairs:
        if key in out:
            raise _error("world artifact", f"duplicate JSON field {key!r}")
        out[key] = value
    return out


def _synthetic_uri(uri: str) -> bool:
    """Whether provenance explicitly uses the reserved synthetic URI scheme."""
    return uri.casefold().startswith("synthetic:")


def _object(value, path: str, fields: set[str]) -> dict:
    if not isinstance(value, dict):
        raise _error(path, "must be an object")
    keys = set(value)
    missing = fields - keys
    extra = keys - fields
    if missing:
        raise _error(path, f"missing field(s): {', '.join(sorted(missing))}")
    if extra:
        raise _error(path, f"unknown field(s): {', '.join(sorted(map(str, extra)))}")
    return value


def _array(value, path: str, *, nonempty: bool = False, maximum: int) -> list:
    if not isinstance(value, list):
        raise _error(path, "must be an array")
    if nonempty and not value:
        raise _error(path, "must not be empty")
    if len(value) > maximum:
        raise _error(path, f"has {len(value)} entries; maximum is {maximum}")
    return value


def _string(value, path: str, *, maximum: int = MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        raise _error(path, "must be a string")
    if not value or value != value.strip():
        raise _error(path, "must be non-empty and have no surrounding whitespace")
    if len(value) > maximum:
        raise _error(path, f"is longer than {maximum} characters")
    return value


def _integer(value, path: str, *, minimum: int, maximum: int) -> int:
    # bool is an int subclass; accepting True as map id 1 is schema corruption, not convenience.
    if not isinstance(value, int) or isinstance(value, bool):
        raise _error(path, "must be an integer")
    if value < minimum or value > maximum:
        raise _error(path, f"must be between {minimum} and {maximum}")
    return value


def _finite(value, path: str, *, minimum: float, maximum: float,
            minimum_inclusive: bool = True) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _error(path, "must be a number")
    out = float(value)
    if not math.isfinite(out):
        raise _error(path, "must be finite")
    below = out < minimum if minimum_inclusive else out <= minimum
    if below or out > maximum:
        relation = ">=" if minimum_inclusive else ">"
        raise _error(path, f"must be {relation} {minimum} and <= {maximum}")
    return out


def _positive(value, path: str, maximum: float) -> float:
    return _finite(value, path, minimum=0.0, maximum=maximum, minimum_inclusive=False)


@dataclass(frozen=True, slots=True)
class Vec2:
    """A point in grid or world space.

    Region fields use integer-valued cell coordinates, while geometry calculations may use
    fractional world coordinates.  The payload parser enforces the integer-cell distinction.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Vec2.{name}: must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"Vec2.{name}: must be finite")


@dataclass(frozen=True, slots=True)
class SourceImage:
    """Provenance and affine grid-to-source-image transform for one map.

    ``grid_origin_*_px`` is the top-left edge of cell ``(0, 0)`` in the source image.  One
    semantic cell spans ``pixels_per_cell`` pixels on both axes.  Runtime reasoning never reads
    RGB pixels; these fields make every collision cell traceable to the hashed source bytes.
    """

    uri: str
    sha256: str
    width_px: int
    height_px: int
    grid_origin_x_px: float
    grid_origin_y_px: float
    pixels_per_cell: float


@dataclass(frozen=True, slots=True)
class DataSource:
    """Verified source manifest behind the mechanics in this atomic snapshot."""

    uri: str
    sha256: str
    verification_method: str
    verified_at: datetime


@dataclass(frozen=True, slots=True)
class MapGeometry:
    """Semantic collision grid and strategically meaningful cells for one map revision."""

    map_id: int
    name: str
    mode: str
    revision: str
    annotation_method: str
    verified_at: datetime
    cell_size_world: float
    width: int
    height: int
    terrain: Tuple[str, ...]
    objective_cells: Tuple[Vec2, ...]
    ally_spawn_cells: Tuple[Vec2, ...]
    enemy_spawn_cells: Tuple[Vec2, ...]
    source: SourceImage

    def terrain_at(self, x: int, y: int) -> str:
        """Terrain at an integer cell, failing closed outside the grid."""
        if (
            isinstance(x, int)
            and isinstance(y, int)
            and 0 <= x < self.width
            and 0 <= y < self.height
        ):
            return self.terrain[y][x]
        return "?"

    def cell_center_world(self, cell: Vec2) -> Vec2:
        """Center of an integer cell in canonical world coordinates (origin: grid top-left)."""
        return Vec2((float(cell.x) + 0.5) * self.cell_size_world,
                    (float(cell.y) + 0.5) * self.cell_size_world)


@dataclass(frozen=True, slots=True)
class AttackProfile:
    """Power-11 base-attack geometry and timing in canonical world units."""

    kind: str                         # direct | lobbed
    damage: float                     # per projectile
    projectiles: int                  # projectiles per ammo
    ammo: int
    reload_seconds: float
    unload_seconds: float
    range_world: float
    min_range_world: float
    projectile_speed_world_per_second: float
    projectile_radius_world: float
    splash_radius_world: float


@dataclass(frozen=True, slots=True)
class BrawlerMechanics:
    """Power-11 base-kit body and attack mechanics for one stable brawler id."""

    brawler_id: int
    name: str
    max_health: float
    move_speed_world_per_second: float
    collision_radius_world: float
    attack: AttackProfile


@dataclass(frozen=True, slots=True)
class WorldModel:
    """One immutable, internally consistent world snapshot."""

    schema_version: int
    artifact_id: str
    game_build: str
    effective_at: datetime
    power_level: int
    mechanics_source: DataSource
    maps: Tuple[MapGeometry, ...]
    brawlers: Tuple[BrawlerMechanics, ...]

    @classmethod
    def from_payload(cls, payload) -> "WorldModel":
        """Validate and construct a world snapshot.

        This method never leaks ``KeyError``/``TypeError`` from malformed input: every rejected
        payload raises a normalized :class:`ValueError` naming the offending field.
        """
        try:
            return _parse_world(payload)
        except ValueError:
            raise
        except Exception as exc:  # defensive normalization for adversarial direct callers
            raise ValueError(
                f"world artifact: invalid payload ({type(exc).__name__}: {exc})"
            ) from exc

    def map_for(self, map_id: int, mode: Optional[str] = None) -> Optional[MapGeometry]:
        """The map with this stable id, optionally guarded by its mode, or ``None``."""
        for map_geometry in self.maps:
            if map_geometry.map_id == map_id:
                return map_geometry if mode is None or map_geometry.mode == mode else None
        return None

    def brawler_for(self, brawler_id: int) -> Optional[BrawlerMechanics]:
        """Mechanics for this stable brawler id, or ``None`` when coverage is incomplete."""
        for brawler in self.brawlers:
            if brawler.brawler_id == brawler_id:
                return brawler
        return None


_SOURCE_FIELDS = {
    "uri", "sha256", "width_px", "height_px", "grid_origin_x_px", "grid_origin_y_px",
    "pixels_per_cell",
}
_DATA_SOURCE_FIELDS = {"uri", "sha256", "verification_method", "verified_at"}
_MAP_FIELDS = {
    "map_id", "name", "mode", "revision", "annotation_method", "verified_at",
    "cell_size_world", "width", "height", "terrain", "objective_cells",
    "ally_spawn_cells", "enemy_spawn_cells", "source",
}
_ATTACK_FIELDS = {
    "kind", "damage", "projectiles", "ammo", "reload_seconds", "unload_seconds",
    "range_world", "min_range_world", "projectile_speed_world_per_second",
    "projectile_radius_world", "splash_radius_world",
}
_BRAWLER_FIELDS = {
    "brawler_id", "name", "max_health", "move_speed_world_per_second",
    "collision_radius_world", "attack",
}
_WORLD_FIELDS = {
    "schema_version", "artifact_id", "game_build", "effective_at", "power_level",
    "mechanics_source", "maps", "brawlers",
}


def _parse_data_source(value, path: str) -> DataSource:
    doc = _object(value, path, _DATA_SOURCE_FIELDS)
    uri = _string(doc["uri"], f"{path}.uri")
    sha256 = _string(doc["sha256"], f"{path}.sha256", maximum=64)
    if not _SHA256_RE.fullmatch(sha256):
        raise _error(f"{path}.sha256", "must be exactly 64 hexadecimal characters")
    method = _string(
        doc["verification_method"], f"{path}.verification_method", maximum=32
    )
    if method not in {"human_verified", "validated_extractor", "synthetic"}:
        raise _error(
            f"{path}.verification_method",
            "must be 'human_verified', 'validated_extractor', or 'synthetic'",
        )
    return DataSource(
        uri=uri,
        sha256=sha256.lower(),
        verification_method=method,
        verified_at=_utc_datetime(doc["verified_at"], f"{path}.verified_at"),
    )


def _parse_source(value, path: str, *, grid_width: int, grid_height: int) -> SourceImage:
    doc = _object(value, path, _SOURCE_FIELDS)
    uri = _string(doc["uri"], f"{path}.uri")
    sha256 = _string(doc["sha256"], f"{path}.sha256", maximum=64)
    if not _SHA256_RE.fullmatch(sha256):
        raise _error(f"{path}.sha256", "must be exactly 64 hexadecimal characters")
    width_px = _integer(doc["width_px"], f"{path}.width_px", minimum=1,
                        maximum=MAX_IMAGE_DIMENSION_PX)
    height_px = _integer(doc["height_px"], f"{path}.height_px", minimum=1,
                         maximum=MAX_IMAGE_DIMENSION_PX)
    origin_x = _finite(doc["grid_origin_x_px"], f"{path}.grid_origin_x_px", minimum=0.0,
                       maximum=float(width_px))
    origin_y = _finite(doc["grid_origin_y_px"], f"{path}.grid_origin_y_px", minimum=0.0,
                       maximum=float(height_px))
    pixels_per_cell = _positive(doc["pixels_per_cell"], f"{path}.pixels_per_cell",
                                float(MAX_IMAGE_DIMENSION_PX))

    # The transform describes cell edges, so the far grid edge must fit inside the source.  A
    # tiny tolerance absorbs ordinary decimal serialization, not a genuinely cropped grid.
    tolerance = 1e-7 * max(width_px, height_px, 1)
    if origin_x + grid_width * pixels_per_cell > width_px + tolerance:
        raise _error(path, "x transform places the semantic grid outside the source image")
    if origin_y + grid_height * pixels_per_cell > height_px + tolerance:
        raise _error(path, "y transform places the semantic grid outside the source image")
    return SourceImage(
        uri=uri,
        sha256=sha256.lower(),
        width_px=width_px,
        height_px=height_px,
        grid_origin_x_px=origin_x,
        grid_origin_y_px=origin_y,
        pixels_per_cell=pixels_per_cell,
    )


def _parse_cells(value, path: str, *, width: int, height: int, nonempty: bool = True,
                 maximum: int = MAX_CELLS_PER_REGION) -> Tuple[Vec2, ...]:
    cells = _array(value, path, nonempty=nonempty, maximum=maximum)
    out = []
    seen = set()
    for index, raw in enumerate(cells):
        cell_path = f"{path}[{index}]"
        if not isinstance(raw, list) or len(raw) != 2:
            raise _error(cell_path, "must be a two-element [x, y] array")
        x = _integer(raw[0], f"{cell_path}[0]", minimum=0, maximum=width - 1)
        y = _integer(raw[1], f"{cell_path}[1]", minimum=0, maximum=height - 1)
        if (x, y) in seen:
            raise _error(cell_path, f"duplicates cell [{x}, {y}]")
        seen.add((x, y))
        out.append(Vec2(float(x), float(y)))
    return tuple(out)


def _parse_map(value, index: int) -> MapGeometry:
    path = f"world artifact.maps[{index}]"
    doc = _object(value, path, _MAP_FIELDS)
    map_id = _integer(doc["map_id"], f"{path}.map_id", minimum=1, maximum=2_147_483_647)
    name = _string(doc["name"], f"{path}.name", maximum=256)
    mode = _string(doc["mode"], f"{path}.mode", maximum=64)
    if mode not in RANKED_MODES:
        raise _error(f"{path}.mode", f"must be one of {', '.join(RANKED_MODES)}")
    revision = _string(doc["revision"], f"{path}.revision", maximum=256)
    annotation_method = _string(
        doc["annotation_method"], f"{path}.annotation_method", maximum=32
    )
    if annotation_method not in {"human_verified", "validated_decoder", "synthetic"}:
        raise _error(
            f"{path}.annotation_method",
            "must be 'human_verified', 'validated_decoder', or 'synthetic'",
        )
    verified_at = _utc_datetime(doc["verified_at"], f"{path}.verified_at")
    cell_size_world = _finite(
        doc["cell_size_world"],
        f"{path}.cell_size_world",
        minimum=MIN_CELL_SIZE_WORLD,
        maximum=10_000.0,
    )
    width = _integer(doc["width"], f"{path}.width", minimum=1, maximum=MAX_GRID_DIMENSION)
    height = _integer(doc["height"], f"{path}.height", minimum=1, maximum=MAX_GRID_DIMENSION)
    if width * height > MAX_GRID_CELLS:
        raise _error(path, f"grid has {width * height} cells; maximum is {MAX_GRID_CELLS}")

    raw_rows = _array(doc["terrain"], f"{path}.terrain", nonempty=True,
                      maximum=MAX_GRID_DIMENSION)
    if len(raw_rows) != height:
        raise _error(f"{path}.terrain", f"must have exactly height={height} rows")
    rows = []
    for y, row in enumerate(raw_rows):
        row_path = f"{path}.terrain[{y}]"
        if not isinstance(row, str):
            raise _error(row_path, "must be a string")
        if len(row) != width:
            raise _error(row_path, f"must have exactly width={width} cells")
        invalid = sorted(set(row) - TERRAIN_ALPHABET)
        if invalid:
            raise _error(row_path, f"unknown terrain code(s): {''.join(invalid)!r}")
        rows.append(row)

    # Some modes have no fixed spatial objective.  Empty means honestly uncovered/not applicable;
    # spawn regions remain mandatory because every supported battlefield has both teams.
    objective_cells = _parse_cells(doc["objective_cells"], f"{path}.objective_cells",
                                   width=width, height=height, nonempty=False)
    ally_spawns = _parse_cells(doc["ally_spawn_cells"], f"{path}.ally_spawn_cells",
                               width=width, height=height)
    enemy_spawns = _parse_cells(doc["enemy_spawn_cells"], f"{path}.enemy_spawn_cells",
                                width=width, height=height)

    # Spawn/objective labels embedded in a blocked cell are almost certainly a coordinate-system
    # error.  Reject rather than quietly pathing from inside a wall or unknown region.
    for region_name, region in (("objective_cells", objective_cells),
                                ("ally_spawn_cells", ally_spawns),
                                ("enemy_spawn_cells", enemy_spawns)):
        for cell in region:
            x, y = int(cell.x), int(cell.y)
            if rows[y][x] in MOVEMENT_BLOCKING_TERRAIN:
                raise _error(f"{path}.{region_name}",
                             f"cell [{x}, {y}] is not walkable ({rows[y][x]!r})")
    overlap = {(int(c.x), int(c.y)) for c in ally_spawns} & {
        (int(c.x), int(c.y)) for c in enemy_spawns
    }
    if overlap:
        raise _error(path, f"ally and enemy spawn cells overlap at {sorted(overlap)[0]}")

    source = _parse_source(doc["source"], f"{path}.source", grid_width=width,
                           grid_height=height)
    return MapGeometry(
        map_id=map_id,
        name=name,
        mode=mode,
        revision=revision,
        annotation_method=annotation_method,
        verified_at=verified_at,
        cell_size_world=cell_size_world,
        width=width,
        height=height,
        terrain=tuple(rows),
        objective_cells=objective_cells,
        ally_spawn_cells=ally_spawns,
        enemy_spawn_cells=enemy_spawns,
        source=source,
    )


def _parse_attack(value, path: str) -> AttackProfile:
    doc = _object(value, path, _ATTACK_FIELDS)
    kind = _string(doc["kind"], f"{path}.kind", maximum=16)
    if kind not in {"direct", "lobbed"}:
        raise _error(f"{path}.kind", "must be 'direct' or 'lobbed'")
    damage = _positive(doc["damage"], f"{path}.damage", 1_000_000_000.0)
    projectiles = _integer(doc["projectiles"], f"{path}.projectiles", minimum=1, maximum=1_024)
    ammo = _integer(doc["ammo"], f"{path}.ammo", minimum=1, maximum=1_024)
    reload_seconds = _positive(doc["reload_seconds"], f"{path}.reload_seconds", 600.0)
    unload_seconds = _finite(doc["unload_seconds"], f"{path}.unload_seconds", minimum=0.0,
                             maximum=600.0)
    range_world = _positive(doc["range_world"], f"{path}.range_world", 100_000.0)
    min_range_world = _finite(doc["min_range_world"], f"{path}.min_range_world", minimum=0.0,
                              maximum=100_000.0)
    if min_range_world > range_world:
        raise _error(f"{path}.min_range_world", "must not exceed range_world")
    speed = _positive(doc["projectile_speed_world_per_second"],
                      f"{path}.projectile_speed_world_per_second", 1_000_000.0)
    projectile_radius = _finite(doc["projectile_radius_world"],
                                f"{path}.projectile_radius_world", minimum=0.0,
                                maximum=100_000.0)
    splash_radius = _finite(doc["splash_radius_world"], f"{path}.splash_radius_world",
                            minimum=0.0, maximum=100_000.0)
    return AttackProfile(
        kind=kind,
        damage=damage,
        projectiles=projectiles,
        ammo=ammo,
        reload_seconds=reload_seconds,
        unload_seconds=unload_seconds,
        range_world=range_world,
        min_range_world=min_range_world,
        projectile_speed_world_per_second=speed,
        projectile_radius_world=projectile_radius,
        splash_radius_world=splash_radius,
    )


def _parse_brawler(value, index: int) -> BrawlerMechanics:
    path = f"world artifact.brawlers[{index}]"
    doc = _object(value, path, _BRAWLER_FIELDS)
    return BrawlerMechanics(
        brawler_id=_integer(doc["brawler_id"], f"{path}.brawler_id", minimum=1,
                            maximum=2_147_483_647),
        name=_string(doc["name"], f"{path}.name", maximum=256),
        max_health=_positive(doc["max_health"], f"{path}.max_health", 1_000_000_000.0),
        move_speed_world_per_second=_positive(
            doc["move_speed_world_per_second"], f"{path}.move_speed_world_per_second", 1_000_000.0),
        collision_radius_world=_positive(doc["collision_radius_world"],
                                         f"{path}.collision_radius_world", 100_000.0),
        attack=_parse_attack(doc["attack"], f"{path}.attack"),
    )


def _utc_datetime(value, path: str) -> datetime:
    text = _string(value, path, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _error(path, "must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise _error(path, "must include a UTC offset ('Z' or '+00:00')")
    return parsed.astimezone(timezone.utc)


def _parse_world(payload) -> WorldModel:
    path = "world artifact"
    doc = _object(payload, path, _WORLD_FIELDS)
    version = _integer(doc["schema_version"], f"{path}.schema_version", minimum=1,
                       maximum=2_147_483_647)
    if version != SCHEMA_VERSION:
        raise _error(f"{path}.schema_version",
                     f"unsupported version {version}; expected {SCHEMA_VERSION}")
    power_level = _integer(doc["power_level"], f"{path}.power_level", minimum=1, maximum=11)
    if power_level != POWER_LEVEL:
        raise _error(f"{path}.power_level",
                     f"must be {POWER_LEVEL}; the serving baseline is Power {POWER_LEVEL}")

    raw_maps = _array(doc["maps"], f"{path}.maps", nonempty=True, maximum=MAX_MAPS)
    maps = tuple(_parse_map(value, index) for index, value in enumerate(raw_maps))
    map_ids = set()
    total_cells = 0
    for index, map_geometry in enumerate(maps):
        if map_geometry.map_id in map_ids:
            raise _error(f"{path}.maps[{index}].map_id",
                         f"duplicates map id {map_geometry.map_id}")
        map_ids.add(map_geometry.map_id)
        total_cells += map_geometry.width * map_geometry.height
    if total_cells > MAX_TOTAL_GRID_CELLS:
        raise _error(f"{path}.maps",
                     f"contain {total_cells} cells total; maximum is {MAX_TOTAL_GRID_CELLS}")

    raw_brawlers = _array(doc["brawlers"], f"{path}.brawlers", nonempty=True,
                          maximum=MAX_BRAWLERS)
    brawlers = tuple(_parse_brawler(value, index) for index, value in enumerate(raw_brawlers))
    brawler_ids = set()
    for index, brawler in enumerate(brawlers):
        if brawler.brawler_id in brawler_ids:
            raise _error(f"{path}.brawlers[{index}].brawler_id",
                         f"duplicates brawler id {brawler.brawler_id}")
        brawler_ids.add(brawler.brawler_id)

    return WorldModel(
        schema_version=version,
        artifact_id=_string(doc["artifact_id"], f"{path}.artifact_id", maximum=256),
        game_build=_string(doc["game_build"], f"{path}.game_build", maximum=256),
        effective_at=_utc_datetime(doc["effective_at"], f"{path}.effective_at"),
        power_level=power_level,
        mechanics_source=_parse_data_source(
            doc["mechanics_source"], f"{path}.mechanics_source"
        ),
        maps=maps,
        brawlers=brawlers,
    )


def load_world_model(
    path=DEFAULT_WORLD_MODEL_PATH, *, allow_synthetic: bool = False
) -> Optional[WorldModel]:
    """Load a strict world artifact, or ``None`` on every failure.

    The 16 MiB limit is checked from metadata *before* reading or JSON-decoding.  The bounded
    second read protects against a file that changes between ``stat`` and ``open``.  Serving can
    therefore keep the feature dark safely while the real, sourced artifact is still absent.
    Synthetic map annotations or mechanics sources are rejected by default; tests and explicit
    development tools may opt in with ``allow_synthetic=True`` without weakening
    :meth:`WorldModel.from_payload`.
    """
    try:
        artifact_path = Path(path)
    except Exception as exc:  # noqa: BLE001 — this is the intentionally fail-soft serving seam
        logger.warning("world model unavailable: invalid artifact path %r (%s)", path, exc)
        return None
    try:
        size = artifact_path.stat().st_size
    except FileNotFoundError:
        logger.debug("world model unavailable: no artifact at %s", artifact_path)
        return None
    except OSError as exc:
        logger.warning("world model unavailable: cannot stat %s (%s)", artifact_path, exc)
        return None
    if size > MAX_ARTIFACT_BYTES:
        logger.warning("world model unavailable: %s is %.2f MiB (limit %.0f MiB)",
                       artifact_path, size / (1024 * 1024), MAX_ARTIFACT_BYTES / (1024 * 1024))
        return None
    try:
        with artifact_path.open("rb") as handle:
            raw = handle.read(MAX_ARTIFACT_BYTES + 1)
        if len(raw) > MAX_ARTIFACT_BYTES:
            logger.warning("world model unavailable: %s grew beyond the %.0f MiB limit",
                           artifact_path, MAX_ARTIFACT_BYTES / (1024 * 1024))
            return None
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_fields)
        world = WorldModel.from_payload(payload)
        contains_synthetic = (
            world.mechanics_source.verification_method == "synthetic"
            or _synthetic_uri(world.mechanics_source.uri)
            or any(
                map_geometry.annotation_method == "synthetic"
                or _synthetic_uri(map_geometry.source.uri)
                for map_geometry in world.maps
            )
        )
        if allow_synthetic is not True and contains_synthetic:
            logger.warning(
                "world model unavailable: %s contains synthetic mechanics or map annotations",
                artifact_path,
            )
            return None
        return world
    except Exception as exc:  # noqa: BLE001 — serving must survive corrupt input
        logger.warning("world model unavailable: invalid artifact at %s (%s)", artifact_path, exc)
        return None


__all__ = [
    "AttackProfile",
    "BrawlerMechanics",
    "DataSource",
    "DEFAULT_WORLD_MODEL_PATH",
    "LINE_OF_SIGHT_BLOCKING_TERRAIN",
    "MAX_ARTIFACT_BYTES",
    "MAX_BRAWLERS",
    "MAX_CELLS_PER_REGION",
    "MAX_GRID_CELLS",
    "MAX_GRID_DIMENSION",
    "MAX_MAPS",
    "MAX_TOTAL_GRID_CELLS",
    "MIN_CELL_SIZE_WORLD",
    "MOVEMENT_BLOCKING_TERRAIN",
    "MapGeometry",
    "POWER_LEVEL",
    "SCHEMA_VERSION",
    "SourceImage",
    "TERRAIN_ALPHABET",
    "Vec2",
    "WorldModel",
    "load_world_model",
]
