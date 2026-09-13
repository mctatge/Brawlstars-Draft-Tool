"""Contracts for the synthetic-first spatial world-model foundation.

The production catalogs do not contain trustworthy combat mechanics or collision geometry, so
every test uses an explicitly fictional bundle with known answers.  The suite pins strict/fail-soft
loading, exact geometry behavior, factorized antisymmetry, resource bounds, and the dark/disconnected
boundary that keeps the live recommender unchanged.

    PYTHONPATH=backend python -m pytest backend/tests/test_world_model.py -q
"""
from __future__ import annotations

import json
import math
import struct
import zlib
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import bsdraft.world.schema as schema
from bsdraft.engine.engine import DraftEngine
from bsdraft.engine.scoring import DEFAULT_WEIGHTS
from bsdraft.engine.state import DraftState
from bsdraft.engine.stats import DraftStats
from bsdraft.world import (
    PositionedBrawler,
    Vec2,
    WorldModel,
    can_attack,
    cell_center,
    evaluate_duel,
    load_world_model,
    objective_coverage,
    objective_travel_time,
    projectile_clear,
    shortest_path_distance,
    shortest_path_distance_to_any,
    shots_to_defeat,
    team_duel_edge,
    theoretical_time_to_kill,
    world_to_cell,
)
from scripts.bootstrap_map_geometry import PNG_SIGNATURE, build_map_entry


FIXTURE = Path(__file__).parent / "fixtures" / "world_model" / "v1_synthetic.json"
BARRIER_MAP = 99000001
GATE_MAP = 99000002
SHOOTER = 99000101
THROWER = 99000102
TANK = 99000103


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


@pytest.fixture(scope="module")
def world() -> WorldModel:
    loaded = load_world_model(FIXTURE, allow_synthetic=True)
    assert loaded is not None
    return loaded


def test_valid_bundle_is_immutable_versioned_and_mode_guarded(world):
    assert world.schema_version == 1
    assert world.artifact_id == "synthetic-world-v1"
    assert world.power_level == 11
    assert world.effective_at.utcoffset().total_seconds() == 0
    assert world.mechanics_source.uri == "synthetic://mechanics"
    assert world.mechanics_source.verification_method == "synthetic"
    assert world.map_for(BARRIER_MAP, "Heist") is not None
    assert world.map_for(BARRIER_MAP, "Heist").annotation_method == "synthetic"
    assert world.map_for(BARRIER_MAP, "Knockout") is None
    assert world.brawler_for(SHOOTER).name == "Synthetic Line Shooter"
    with pytest.raises((AttributeError, TypeError)):
        world.power_level = 10


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.update(schema_version=2), "unsupported version"),
        (lambda p: p.update(power_level=10), "Power 11"),
        (lambda p: p["maps"].append(p["maps"][0].copy()), "duplicates map id"),
        (lambda p: p["brawlers"].append(p["brawlers"][0].copy()), "duplicates brawler id"),
        (lambda p: p["maps"][0].update(terrain=["bad"]), "height=7"),
        (lambda p: p["maps"][0]["terrain"].__setitem__(0, "......!"), "terrain code"),
        (lambda p: p["brawlers"][0].update(max_health=float("nan")), "must be finite"),
        (lambda p: p["brawlers"][0]["attack"].update(range_world=-1), "must be >"),
        (lambda p: p["maps"][0].update(cell_size_world=1e-4), ">= 0.001"),
        (lambda p: p["maps"][0]["source"].update(sha256="not-a-hash"), "64 hexadecimal"),
        (lambda p: p["mechanics_source"].update(sha256="not-a-hash"), "64 hexadecimal"),
        (
            lambda p: p["mechanics_source"].update(verification_method="unverified"),
            "validated_extractor",
        ),
        (lambda p: p["maps"][0].update(annotation_method="unverified"), "human_verified"),
    ],
)
def test_strict_payload_validation_rejects_corruption(mutate, match):
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        WorldModel.from_payload(payload)


def test_public_loader_fails_soft_for_missing_corrupt_and_oversized(tmp_path, monkeypatch):
    assert load_world_model(tmp_path / "missing.json") is None

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert load_world_model(corrupt) is None

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * 33)
    monkeypatch.setattr(schema, "MAX_ARTIFACT_BYTES", 32)
    assert schema.load_world_model(oversized) is None


def test_public_loader_rejects_synthetic_by_default_and_requires_explicit_opt_in():
    assert load_world_model(FIXTURE) is None
    loaded = load_world_model(FIXTURE, allow_synthetic=True)
    assert loaded is not None
    assert loaded.artifact_id == "synthetic-world-v1"


def test_loader_gates_explicit_synthetic_labels_and_uris_independently(tmp_path):
    payload = _payload()
    for map_payload in payload["maps"]:
        map_payload["annotation_method"] = "human_verified"
    payload["mechanics_source"]["verification_method"] = "validated_extractor"
    relabelled = tmp_path / "relabelled.json"
    relabelled.write_text(json.dumps(payload), encoding="utf-8")
    assert load_world_model(relabelled) is None

    payload["mechanics_source"]["uri"] = "https://example.invalid/mechanics.json"
    mechanics_only = tmp_path / "mechanics-only.json"
    mechanics_only.write_text(json.dumps(payload), encoding="utf-8")
    assert load_world_model(mechanics_only) is None

    for map_payload in payload["maps"]:
        map_payload["source"]["uri"] = (
            f"https://example.invalid/maps/{map_payload['map_id']}.png"
        )
    verified_control = tmp_path / "verified-control.json"
    verified_control.write_text(json.dumps(payload), encoding="utf-8")
    assert load_world_model(verified_control) is not None


def test_public_loader_rejects_duplicate_json_keys_before_schema_validation(tmp_path):
    # Ordinary json.loads silently keeps the last value.  Put a valid value second so this would
    # look like an ordinary v1 artifact unless the decoder explicitly rejects duplicate keys.
    raw = FIXTURE.read_text(encoding="utf-8")
    raw = raw.replace(
        '"schema_version": 1,',
        '"schema_version": 999,\n  "schema_version": 1,',
        1,
    )
    path = tmp_path / "duplicate-key.json"
    path.write_text(raw, encoding="utf-8")
    assert load_world_model(path, allow_synthetic=True) is None


def test_synthetic_fixture_stays_below_the_initial_artifact_budget():
    assert FIXTURE.stat().st_size < schema.MAX_ARTIFACT_BYTES


def test_v1_schema_caps_semantic_geometry_before_any_quadratic_query_runs():
    payload = _payload()
    map_payload = payload["maps"][0]
    width = schema.MAX_GRID_DIMENSION + 1
    map_payload.update(
        width=width,
        terrain=["." * width for _ in range(map_payload["height"])],
    )
    map_payload["source"]["width_px"] = width * 30
    with pytest.raises(ValueError, match="between 1 and 64"):
        WorldModel.from_payload(payload)

    payload = _payload()
    map_payload = payload["maps"][0]
    side = schema.MAX_GRID_DIMENSION
    map_payload.update(
        width=side,
        height=side,
        terrain=["." * side for _ in range(side)],
        objective_cells=[
            [index % side, index // side]
            for index in range(schema.MAX_CELLS_PER_REGION + 1)
        ],
    )
    map_payload["source"].update(width_px=side * 30, height_px=side * 30)
    with pytest.raises(ValueError, match="maximum is 512"):
        WorldModel.from_payload(payload)


def test_v1_schema_caps_aggregate_maps_brawlers_and_cells():
    payload = _payload()
    payload["maps"] = [payload["maps"][0]] * (schema.MAX_MAPS + 1)
    with pytest.raises(ValueError, match=f"maximum is {schema.MAX_MAPS}"):
        WorldModel.from_payload(payload)

    payload = _payload()
    payload["brawlers"] = [payload["brawlers"][0]] * (schema.MAX_BRAWLERS + 1)
    with pytest.raises(ValueError, match=f"maximum is {schema.MAX_BRAWLERS}"):
        WorldModel.from_payload(payload)

    payload = _payload()
    side = schema.MAX_GRID_DIMENSION
    base_map = payload["maps"][0]
    base_map.update(
        width=side,
        height=side,
        terrain=["." * side for _ in range(side)],
    )
    base_map["source"].update(width_px=side * 30, height_px=side * 30)
    count = schema.MAX_TOTAL_GRID_CELLS // schema.MAX_GRID_CELLS + 1
    maps = []
    for index in range(count):
        map_payload = deepcopy(base_map)
        map_payload["map_id"] = base_map["map_id"] + index
        maps.append(map_payload)
    payload["maps"] = maps
    with pytest.raises(ValueError, match=f"maximum is {schema.MAX_TOTAL_GRID_CELLS}"):
        WorldModel.from_payload(payload)


def test_objective_cells_may_be_empty_when_the_mode_has_no_fixed_target():
    payload = _payload()
    payload["maps"][0]["objective_cells"] = []
    loaded = WorldModel.from_payload(payload)
    assert loaded.maps[0].objective_cells == ()


def test_coordinate_transform_is_explicit_and_never_clamps(world):
    geometry = world.map_for(BARRIER_MAP, "Heist")
    assert cell_center(geometry, Vec2(1, 3)) == Vec2(1.5, 3.5)
    assert world_to_cell(geometry, Vec2(1.99, 3.01)) == Vec2(1, 3)
    with pytest.raises(ValueError, match="outside"):
        world_to_cell(geometry, Vec2(7.0, 3.0))


def test_direct_projectile_occlusion_is_symmetric_and_destroyed_wall_opens(world):
    geometry = world.map_for(BARRIER_MAP, "Heist")
    left, right = Vec2(1.5, 3.5), Vec2(5.5, 3.5)
    assert not projectile_clear(geometry, left, right)
    assert not projectile_clear(geometry, right, left)
    assert projectile_clear(geometry, left, right, destroyed_walls=[Vec2(3, 3)])
    assert projectile_clear(geometry, right, left, destroyed_walls=[Vec2(3, 3)])


def test_projectile_radius_uses_a_conservative_swept_volume(world):
    geometry = world.map_for(BARRIER_MAP, "Heist")
    start, end = Vec2(0.5, 0.4), Vec2(6.5, 0.4)
    assert projectile_clear(geometry, start, end, 0.59)
    assert not projectile_clear(geometry, start, end, 0.60)


def test_direct_and_lobbed_attacks_obey_explicit_wall_semantics(world):
    geometry = world.map_for(BARRIER_MAP, "Heist")
    shooter = world.brawler_for(SHOOTER)
    thrower = world.brawler_for(THROWER)
    left, right = Vec2(1.5, 3.5), Vec2(5.5, 3.5)
    assert not can_attack(geometry, shooter.attack, left, right)
    assert can_attack(geometry, thrower.attack, left, right)
    assert can_attack(
        geometry, shooter.attack, left, right, destroyed_walls=[Vec2(3, 3)]
    )


def test_lobbed_attack_still_fails_closed_across_unknown_terrain(world):
    geometry = world.map_for(BARRIER_MAP, "Heist")
    rows = list(geometry.terrain)
    rows[3] = "...?..."
    unknown = replace(geometry, terrain=tuple(rows))
    thrower = world.brawler_for(THROWER)
    assert not can_attack(unknown, thrower.attack, Vec2(1.5, 3.5), Vec2(5.5, 3.5))


def test_attack_range_boundary_is_inclusive(world):
    geometry = world.map_for(BARRIER_MAP, "Heist")
    shooter = world.brawler_for(SHOOTER)
    origin = Vec2(0.5, 0.5)
    assert can_attack(geometry, shooter.attack, origin, Vec2(6.5, 0.5))
    assert not can_attack(geometry, shooter.attack, origin, Vec2(6.50000000001, 0.5))


def test_attack_reach_and_ttk_share_one_range_boundary_contract(world):
    """A geometric hit must never turn into an infinite TTK at the same exact distance.

    This probes both edges inside and just outside the floating tolerance without prescribing
    whether the shared contract is strict or tolerant there; the two public reads only need to
    agree with each other.
    """
    geometry = world.map_for(BARRIER_MAP, "Heist")
    shooter = world.brawler_for(SHOOTER)
    thrower = world.brawler_for(THROWER)

    cases = (
        (shooter, thrower, shooter.attack.range_world, Vec2(0.5, 0.5), 1.0),
        (thrower, shooter, thrower.attack.min_range_world, Vec2(0.5, 0.5), 0.0),
    )
    for attacker, defender, boundary, origin, direction in cases:
        for delta in (-2e-12, -5e-13, 0.0, 5e-13, 2e-12):
            distance = boundary + delta
            target = Vec2(origin.x + direction * distance, origin.y + (1.0 - direction) * distance)
            reaches = can_attack(geometry, attacker.attack, origin, target)
            finite_ttk = math.isfinite(theoretical_time_to_kill(attacker, defender, distance))
            assert reaches is finite_ttk, (attacker.name, boundary, delta, reaches, finite_ttk)


def test_pathfinding_detours_symmetrically_and_destroyed_wall_shortens_path(world):
    geometry = world.map_for(BARRIER_MAP, "Heist")
    left, right = Vec2(1, 3), Vec2(5, 3)
    intact = shortest_path_distance(geometry, left, right, body_radius_world=0.2)
    reverse = shortest_path_distance(geometry, right, left, body_radius_world=0.2)
    opened = shortest_path_distance(
        geometry, left, right, body_radius_world=0.2, destroyed_walls=[Vec2(3, 3)]
    )
    assert intact == pytest.approx(reverse)
    assert intact > opened
    assert opened == pytest.approx(4.0)


def test_body_radius_makes_a_one_cell_gate_physically_meaningful(world):
    geometry = world.map_for(GATE_MAP, "Heist")
    start, goal = Vec2(2, 1), Vec2(2, 3)
    assert shortest_path_distance(
        geometry, start, goal, body_radius_world=0.49
    ) == pytest.approx(2.0)
    assert math.isinf(shortest_path_distance(
        geometry, start, goal, body_radius_world=0.50
    ))


def test_pathfinding_forbids_diagonal_corner_cutting(world):
    base = world.map_for(GATE_MAP, "Heist")
    corner = replace(base, width=2, height=2, terrain=(".#", "#."))
    assert math.isinf(shortest_path_distance(corner, Vec2(0, 0), Vec2(1, 1)))


def test_multi_goal_path_search_is_bounded_and_skips_blocked_goals(world):
    geometry = world.map_for(BARRIER_MAP, "Heist")
    assert shortest_path_distance_to_any(
        geometry,
        Vec2(1, 3),
        (Vec2(3, 3), Vec2(1, 4)),
        body_radius_world=0.2,
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="maximum is 512"):
        shortest_path_distance_to_any(
            geometry,
            Vec2(1, 3),
            (Vec2(1, 4),) * (schema.MAX_CELLS_PER_REGION + 1),
            body_radius_world=0.2,
        )


def test_invalid_positioned_brawler_occupancy_is_rejected(world):
    # The wall is inside the map, so a bounds-only check would accept it and emit a plausible
    # numeric duel read from a physically impossible state.
    in_wall = PositionedBrawler(SHOOTER, Vec2(3.5, 2.5))
    in_open = PositionedBrawler(THROWER, Vec2(1.5, 2.5))
    with pytest.raises(ValueError, match="blocked|occup|walkable"):
        evaluate_duel(world, BARRIER_MAP, "Heist", in_wall, in_open)


def test_theoretical_ttk_is_explicit_about_range_and_hit_fraction(world):
    shooter = world.brawler_for(SHOOTER)
    thrower = world.brawler_for(THROWER)
    assert shots_to_defeat(shooter, thrower) == 3
    assert shots_to_defeat(shooter, thrower, hit_fraction=0.5) == 5
    assert theoretical_time_to_kill(shooter, thrower, 4.0) == pytest.approx(
        0.4 + 4.0 / 12.0
    )
    assert math.isinf(theoretical_time_to_kill(shooter, thrower, 6.01))
    with pytest.raises(ValueError, match="too small"):
        shots_to_defeat(shooter, thrower, hit_fraction=1e-310)
    slow_attack = replace(
        shooter,
        attack=replace(shooter.attack, reload_seconds=600.0),
    )
    with pytest.raises(ValueError, match="finite TTK"):
        theoretical_time_to_kill(slow_attack, thrower, 4.0, hit_fraction=1e-306)


def test_positioned_duel_reads_geometry_and_swaps_exactly(world):
    left = PositionedBrawler(SHOOTER, Vec2(1.5, 3.5))
    right = PositionedBrawler(THROWER, Vec2(5.5, 3.5))
    forward = evaluate_duel(world, BARRIER_MAP, "Heist", left, right)
    reverse = evaluate_duel(world, BARRIER_MAP, "Heist", right, left)
    assert forward.first_can_hit is False and forward.second_can_hit is True
    assert forward.edge == -1.0
    assert reverse.edge == 1.0
    assert reverse.edge == -forward.edge

    opened = evaluate_duel(
        world, BARRIER_MAP, "Heist", left, right, destroyed_walls=[Vec2(3, 3)]
    )
    assert opened.first_can_hit and opened.second_can_hit
    assert opened.edge > 0.0


def test_team_duel_edge_is_permutation_invariant_and_structurally_antisymmetric(world):
    a = [
        PositionedBrawler(SHOOTER, Vec2(1.5, 3.5)),
        PositionedBrawler(TANK, Vec2(1.5, 2.5)),
    ]
    b = [
        PositionedBrawler(THROWER, Vec2(5.5, 3.5)),
        PositionedBrawler(SHOOTER, Vec2(5.5, 2.5)),
    ]
    edge = team_duel_edge(world, BARRIER_MAP, "Heist", a, b)
    assert team_duel_edge(world, BARRIER_MAP, "Heist", list(reversed(a)), b) == edge
    assert team_duel_edge(world, BARRIER_MAP, "Heist", b, a) == -edge
    assert team_duel_edge(world, BARRIER_MAP, "Heist", a, a) == 0.0


def test_missing_coverage_is_unavailable_not_neutral(world):
    known = PositionedBrawler(SHOOTER, Vec2(1.5, 3.5))
    unknown = PositionedBrawler(123456789, Vec2(5.5, 3.5))
    assert evaluate_duel(world, BARRIER_MAP, "Heist", known, unknown) is None
    assert team_duel_edge(world, BARRIER_MAP, "Heist", [known], [unknown]) is None
    assert team_duel_edge(world, BARRIER_MAP, "Knockout", [known], [known]) is None


def test_unknown_terrain_never_becomes_numeric_evaluator_evidence(world):
    geometry = world.map_for(BARRIER_MAP, "Heist")
    rows = [row[:3] + "?" + row[4:] for row in geometry.terrain]
    incomplete = replace(geometry, terrain=tuple(rows))
    incomplete_world = replace(
        world,
        maps=tuple(incomplete if item.map_id == BARRIER_MAP else item for item in world.maps),
    )
    shooter = PositionedBrawler(SHOOTER, Vec2(1.5, 3.5))
    tank = PositionedBrawler(TANK, Vec2(5.5, 3.5))

    assert evaluate_duel(incomplete_world, BARRIER_MAP, "Heist", shooter, tank) is None
    assert team_duel_edge(
        incomplete_world, BARRIER_MAP, "Heist", [shooter], [tank]
    ) is None
    assert objective_coverage(
        incomplete_world, BARRIER_MAP, "Heist", SHOOTER, shooter.position
    ) is None
    assert objective_travel_time(
        incomplete_world, BARRIER_MAP, "Heist", TANK, shooter.position
    ) is None


def test_objective_measurements_are_physical_not_win_probabilities(world):
    origin = Vec2(1.5, 3.5)
    assert objective_coverage(world, BARRIER_MAP, "Heist", SHOOTER, origin) == 0.0
    assert objective_coverage(
        world,
        BARRIER_MAP,
        "Heist",
        SHOOTER,
        origin,
        destroyed_walls=[Vec2(3, 3)],
    ) == 1.0
    assert objective_coverage(world, BARRIER_MAP, "Heist", THROWER, origin) == 1.0
    opened_time = objective_travel_time(
        world,
        BARRIER_MAP,
        "Heist",
        TANK,
        origin,
        destroyed_walls=[Vec2(3, 3)],
    )
    assert opened_time == pytest.approx(4.0 / 3.5)


def test_objective_travel_skips_a_blocked_first_goal_and_uses_a_later_one(world):
    geometry = world.map_for(BARRIER_MAP, "Heist")
    # Both targets are open cells.  At radius 0.5 the first touches the adjacent wall and is not a
    # valid occupancy; the second remains reachable one cardinal step from the start.
    geometry = replace(
        geometry,
        objective_cells=(Vec2(2, 2), Vec2(1, 4)),
    )
    tank = replace(world.brawler_for(TANK), collision_radius_world=0.5)
    altered = replace(
        world,
        maps=tuple(geometry if item.map_id == BARRIER_MAP else item for item in world.maps),
        brawlers=tuple(tank if item.brawler_id == TANK else item for item in world.brawlers),
    )

    travel = objective_travel_time(
        altered, BARRIER_MAP, "Heist", TANK, Vec2(1.5, 3.5)
    )
    assert travel == pytest.approx(1.0 / tank.move_speed_world_per_second)


def test_world_model_remains_outside_the_tuned_recommendation_blend():
    assert "world" not in DEFAULT_WEIGHTS
    assert set(DEFAULT_WEIGHTS) == {"map", "model", "counter", "synergy", "role"}


def test_disconnected_world_objects_cannot_change_default_recommendations(monkeypatch, world):
    """The dark first build cannot affect rankings merely by attaching an arbitrary object."""
    engine = DraftEngine(stats=DraftStats(matches=[]), model=None)
    candidates = [SHOOTER, THROWER, TANK]
    monkeypatch.setattr(engine, "candidates", lambda state, roster=None: list(candidates))
    state = DraftState(map_id=BARRIER_MAP, mode="Heist")

    baseline = [asdict(score) for score in engine.recommend_picks(state, top=3)]
    engine.world_model = world
    present = [asdict(score) for score in engine.recommend_picks(state, top=3)]

    class _FailingWorld:
        def __getattribute__(self, name):
            raise RuntimeError("dark world-model code must not be read by the default scorer")

    engine.world_model = _FailingWorld()
    failing = [asdict(score) for score in engine.recommend_picks(state, top=3)]
    assert present == baseline == failing


def test_png_bootstrap_records_exact_pixels_and_emits_only_unknown_cells(tmp_path):
    image = tmp_path / "map.png"
    ihdr = struct.pack(">IIBBBBB", 150, 90, 8, 0, 0, 0, 0)
    pixels = b"".join(b"\x00" + (b"\x00" * 150) for _ in range(90))
    image.write_bytes(
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(pixels))
        + _png_chunk(b"IEND", b"")
    )
    entry = build_map_entry(
        image,
        map_id=123,
        name="Synthetic Source",
        mode="Heist",
        revision="test",
        source_uri="synthetic://source",
        pixels_per_cell=30,
        cell_size_world=1.0,
    )
    assert (entry["width"], entry["height"]) == (5, 3)
    assert entry["source"]["width_px"] == 150
    assert entry["source"]["height_px"] == 90
    assert entry["source"]["sha256"]
    assert entry["terrain"] == ["?????", "?????", "?????"]
    assert entry["annotation_method"] == "unverified"
    assert entry["verified_at"] is None


def test_png_bootstrap_output_cannot_be_loaded_as_a_verified_world(tmp_path):
    image = tmp_path / "map.png"
    ihdr = struct.pack(">IIBBBBB", 150, 90, 8, 0, 0, 0, 0)
    pixels = b"".join(b"\x00" + (b"\x00" * 150) for _ in range(90))
    image.write_bytes(
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(pixels))
        + _png_chunk(b"IEND", b"")
    )
    entry = build_map_entry(
        image,
        map_id=123,
        name="Unverified Skeleton",
        mode="Heist",
        revision="test",
        source_uri="synthetic://source",
        pixels_per_cell=30,
        cell_size_world=1.0,
    )
    payload = _payload()
    payload["maps"] = [entry]
    with pytest.raises(ValueError, match="annotation_method|unverified"):
        WorldModel.from_payload(payload)


def test_png_bootstrap_rejects_a_truncated_header_only_source(tmp_path):
    image = tmp_path / "truncated.png"
    image.write_bytes(
        PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 150, 90)
    )
    with pytest.raises(ValueError, match="truncated"):
        build_map_entry(
            image,
            map_id=123,
            name="Synthetic Source",
            mode="Heist",
            revision="test",
            source_uri="synthetic://source",
            pixels_per_cell=30,
            cell_size_world=1.0,
        )


@pytest.mark.parametrize("idat", [b"", b"not-a-zlib-stream"])
def test_png_bootstrap_rejects_empty_or_invalid_image_data(tmp_path, idat):
    image = tmp_path / "invalid-data.png"
    ihdr = struct.pack(">IIBBBBB", 150, 90, 8, 0, 0, 0, 0)
    image.write_bytes(
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )
    with pytest.raises(ValueError, match="non-empty IDAT|invalid compressed"):
        build_map_entry(
            image,
            map_id=123,
            name="Invalid Source",
            mode="Heist",
            revision="test",
            source_uri="synthetic://source",
            pixels_per_cell=30,
            cell_size_world=1.0,
        )


def test_png_bootstrap_rejects_bad_crc_and_trailing_bytes(tmp_path):
    ihdr = struct.pack(">IIBBBBB", 30, 30, 8, 0, 0, 0, 0)
    pixels = b"".join(b"\x00" + (b"\x00" * 30) for _ in range(30))
    idat = bytearray(_png_chunk(b"IDAT", zlib.compress(pixels)))
    idat[-1] ^= 0x01
    bad_crc = tmp_path / "bad-crc.png"
    bad_crc.write_bytes(
        PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + idat + _png_chunk(b"IEND", b"")
    )
    with pytest.raises(ValueError, match="invalid CRC"):
        build_map_entry(
            bad_crc,
            map_id=123,
            name="Bad CRC",
            mode="Heist",
            revision="test",
            source_uri="synthetic://source",
            pixels_per_cell=30,
            cell_size_world=1.0,
        )

    trailing = tmp_path / "trailing.png"
    trailing.write_bytes(
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(pixels))
        + _png_chunk(b"IEND", b"")
        + b"unexpected"
    )
    with pytest.raises(ValueError, match="trailing bytes"):
        build_map_entry(
            trailing,
            map_id=123,
            name="Trailing Bytes",
            mode="Heist",
            revision="test",
            source_uri="synthetic://source",
            pixels_per_cell=30,
            cell_size_world=1.0,
        )


def test_png_bootstrap_enforces_the_same_grid_cap_as_the_runtime_schema(tmp_path):
    width = (schema.MAX_GRID_DIMENSION + 1) * 30
    height = 30
    image = tmp_path / "too-wide.png"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    pixels = b"".join(b"\x00" + (b"\x00" * width) for _ in range(height))
    image.write_bytes(
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(pixels))
        + _png_chunk(b"IEND", b"")
    )
    with pytest.raises(ValueError, match="semantic-grid limits"):
        build_map_entry(
            image,
            map_id=123,
            name="Oversized Grid",
            mode="Heist",
            revision="test",
            source_uri="synthetic://source",
            pixels_per_cell=30,
            cell_size_world=1.0,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
