# Spatial World Model — decision, scope, and rollout

This project will add a mechanics-driven, spatial reasoning layer alongside the existing
match-outcome model. The first build is deliberately a **static spatial world-model foundation**:
it defines how sourced literal brawler mechanics and verified map geometry become deterministic
properties such as line of sight, path distance, and attack reach. The checked-in executable sample
is synthetic because the current catalogs do not provide trustworthy physics data. It is not yet a
battle simulator, and it does not change live recommendation scores.

## Three different model layers

These layers answer different questions and should not be conflated:

1. **Current empirical win-probability model.** The existing embedding net learns which brawlers,
   teams, maps, and modes have tended to win in real Ranked matches. It is compact, calibrated
   against observed outcomes, and captures player behavior indirectly, but it does not know why a
   wall, projectile width, or range matchup matters.
2. **Static spatial world model.** The new layer represents the battlefield and literal mechanics.
   It can derive sightlines, routes, reachable areas, attack coverage, range pressure, and eventually
   interpretable team advantages without replaying a match one frame at a time.
3. **Dynamic simulator.** A possible later layer would evolve positions, health, ammo, supers,
   terrain, and objectives through time under player policies. That is a much larger research
   project because aim, dodging, path choice, coordination, and target selection must also be
   modeled. It is not part of the first build.

The long-term target is a hybrid: mechanics supplies causal structure and immediate post-patch or
cold-start knowledge, while observed matches correct its assumptions and calibrate how much those
advantages matter in real play.

## Coordinates and map representation

Each source map image is identified immutably by its URI, SHA-256, original pixel dimensions,
grid origin, and pixels-per-cell scale. The bytes are not copied into this repository: the URI
locates the source and the hash identifies the exact bytes that were inspected. This preserves
provenance and a reversible mapping between image and world coordinates. Pixels
themselves are not treated as collision truth: antialiasing, decorations, shadows, and skins make
raw image colors unsuitable for physics.

Runtime geometry uses a compact semantic collision grid. Cell coordinates are integer `[x, y]`
indices, with the coordinate transform declared rather than inferred. The initial terrain alphabet
is:

| Code | Meaning |
| --- | --- |
| `.` | open ground |
| `#` | solid wall |
| `+` | destructible wall |
| `~` | water |
| `b` | bush |
| `?` | unknown / not yet verified |

Unknown terrain is allowed so partially traced maps can be represented honestly, but low-level
geometry must fail closed around it. Higher-level evaluator reads on an incomplete map are
unavailable rather than numeric: a blocked path caused only by `?` is not evidence that a matchup
or objective is bad. Objectives and both teams' spawn cells are stored separately from collision
terrain. Objective regions may be empty for modes without fixed objective geometry; spawn regions
remain required.

The first schema records:

- `Vec2(x, y)`.
- `SourceImage(uri, sha256, width_px, height_px, grid_origin_x_px, grid_origin_y_px,
  pixels_per_cell)`.
- `DataSource(uri, sha256, verification_method, verified_at)` for the immutable,
  independently classified mechanics-extraction input.
- `MapGeometry(map_id, name, mode, revision, annotation_method, verified_at, cell_size_world,
  width, height, terrain, objective_cells, ally_spawn_cells, enemy_spawn_cells, source)`.
- `AttackProfile(kind, damage, projectiles, ammo, reload_seconds, unload_seconds, range_world,
  min_range_world, projectile_speed_world_per_second, projectile_radius_world,
  splash_radius_world)`.
- `BrawlerMechanics(brawler_id, name, max_health, move_speed_world_per_second,
  collision_radius_world, attack)`.
- `WorldModel(schema_version, artifact_id, game_build, effective_at, power_level,
  mechanics_source, maps, brawlers)`.

Maps and brawlers are arrays in the serialized bundle so duplicate IDs can be detected instead of
being silently overwritten by JSON object-key semantics; the loader also rejects duplicate JSON
field names before schema validation. Schema validation rejects ragged or oversized grids, invalid
coordinates, unsupported versions, non-finite mechanics, and values outside their physical domains.
Map annotation provenance must be explicitly `human_verified`, `validated_decoder`, or `synthetic`;
mechanics provenance must likewise be `human_verified`, `validated_extractor`, or `synthetic`. The
production loader rejects either kind of `synthetic` input unless a development/test caller opts
in, and also rejects the reserved `synthetic:` source-URI scheme as defense in depth. These labels
are publisher attestations, not cryptographic proof of interpretation: the future offline publisher
must verify the referenced bytes and review status. The bootstrapper emits `unverified` with no
verification time, which intentionally cannot pass strict artifact validation until reviewed.

In version 1, `objective_cells` means a **walkable reference or control region**. It can describe a
Hot Zone, gem/ball reference area, or a verified approach region, but it cannot pretend a Heist safe
or another solid structure is a walkable cell. Attackable structures need a later typed-object layer
with their own collision shape and health. Until then those objective fields remain empty or use an
explicitly documented walkable region.

This first profile is only the base attack and movement foundation. The eventual mechanics model
should cover health, speed, collision radius, damage, projectiles or pellets, ammo, reload and unload
timing, range, minimum range, projectile speed and width, spread, splash, piercing, bouncing,
healing, shields, crowd control, super charge and geometry, summons, dashes or teleports, and terrain
interactions such as wall breaking. Missing mechanics remain explicitly uncovered; they are never
filled with invented values.

## Factorized reasoning and symmetry

The world model must calculate from compact brawler, map, and pairwise factors on demand. One useful
factorization defines a directed team read $F$, then antisymmetrizes it:

\[
F(A,B,m)
= U(A,m)
+ \sum_{a \in A}\sum_{b \in B} C(a,b,m)
+ \operatorname{Syn}(A,m),
\]

\[
V(A,B,m) = F(A,B,m) - F(B,A,m).
\]

Here $U$ is map-conditioned team utility, $C$ is a directed cross-team interaction, and
$\operatorname{Syn}$ is within-team capability coverage or synergy. This construction guarantees

\[
V(B,A,m)=-V(A,B,m).
\]

If the signed value is later mapped to a probability with a sigmoid,

\[
P(A \text{ wins}\mid B,m)=\sigma\!\left(V(A,B,m)\right),
\]

then team swapping also guarantees

\[
P(A \text{ wins}\mid B,m)+P(B \text{ wins}\mid A,m)=1.
\]

Team order must not affect a result, identical teams must have zero advantage, and the empty board
must be neutral. These are structural tests, not properties left for data fitting to discover.

Initial geometric outputs are facts or measurements, not confident win claims: whether a line is
blocked, the shortest traversable distance for a given body radius, and whether an attack's
projectile centre can reach a point. The first kernel is intentionally conservative: target-body
intersection and splash landing envelopes are not yet promoted to duel evidence even though their
literal dimensions are retained in the schema. Higher-level features can later include effective
damage by distance, burst survivability, time to kill, lane access, choke pressure, objective
coverage, protected firing positions, wall-break value, and team capability gaps.

Version 1 path distance is cell-centre quantized. A continuous start point must be physically
occupiable, but travel begins at the centre of its containing semantic cell; sub-cell movement is
left for the later continuous/dynamic layer rather than presented as exact here.

## Compute and memory envelope

The static model does not require better local hardware or an additional cloud service. Geometry can
be verified and compiled locally, and the deployed API can load a compact bundle and run bounded
NumPy or standard-library calculations. Expensive derived fields should be prepared offline or
computed lazily for the selected map.

The scale is small when it stays factorized:

- The current `winprob.npz` is about **59 KB** for **107 brawlers**.
- A generous profile of 256 `float32` values per brawler is about **100 KB** in total.
- Thirty-two `float32` values for every directed brawler pair are about **1.4 MB**.
- A semantic terrain grid can use one compact cell code rather than eagerly materializing many
  one-hot channels, and only the selected map's derived fields need to be resident.

The prohibited shape is an all-composition lookup. There are

\[
\binom{107}{3}=198{,}485
\]

three-brawler teams. A dense team-versus-team table contains about $39.4$ billion values; even one
`float32` per matchup is about **158 GB per map**, before multiplying by maps or patches. The design
therefore never precomputes every composition, every position pair, or every simulated draft.

Every artifact and cache gets explicit size limits. Version 1 accepts semantic grids up to the
verified-safe bounds in `world/schema.py`, not source-pixel-sized grids; larger or finer grids need
an optimized kernel and a schema revision. Loading rejects over-limit dimensions before constructing
derived structures, canonical world cells cannot be smaller than `0.001` units, per-query caches are
bounded by the accepted grid, and public artifacts never enable NumPy pickle loading. Multi-objective
travel uses one single-source search rather than repeating pathfinding for each cell. Tests assert
structural operation bounds and serialized size rather than fragile wall-clock or process-RSS
thresholds.

Cloud or GPU investment becomes relevant only for a later tick-level simulator, learned spatial
network, or self-play system. Even then, training can be temporary and offline; the deployed result
should remain compressed and cheap to serve.

## Data reality and versioning

The current sources are not sufficient for production spatial claims:

- `data/reference/brawlers.json` is a catalog and does not contain trustworthy literal combat
  mechanics.
- `data/reference/maps.json` contains catalog metadata and image links, not verified wall, water,
  bush, spawn, or objective geometry.
- Battle logs contain final teams and outcomes, not positions, shots, ammo state, destroyed terrain,
  aim, dodging, or lane assignments.

The first implementation is therefore synthetic-first. Tests use tiny, explicitly fictional maps
and brawler profiles with known answers. The bootstrap tool may preserve source-image metadata and
help prepare a trace, but it must not turn image guesses or incomplete scraping into production
truth. There is deliberately **no `data/reference/world_model.json` in the first build**. A production
bundle will be added only after its mechanics and geometry have verifiable sources and review.

### First source-image probe

The first bootstrap probe used the catalog image for Bridge Too Far (`map_id=15000072`). The exact
indexed PNG inspected on 2026-09-09 is 690×1050 pixels and hashes to
`f28011e1ea840fca1a1b5c9d2517fa4c2bf41fb9a36ee11d387c37ffd0c70cec`. Declaring 30 pixels per cell
produces an exact 23×35 grid with no remainder. That establishes a plausible image-to-grid transform
for this revision, not collision semantics: the bootstrap output intentionally leaves all 805 cells
as `?` until walls, water, bushes, objectives, and spawns are independently verified. The same scale
must be checked rather than assumed on every additional source image. The reproducible URI, catalog
snapshot hash, retrieval timestamp, byte count, transform candidate, and verification status are in
`docs/world-model-source-probes.json`.

Production data must ship as one patch-versioned, internally consistent bundle. The offline
publisher must retrieve or open every declared mechanics/map source and reproduce its SHA-256 before
it can label the bundle verified; the runtime loader validates the manifest, not remote source bytes.
The bundle's schema version, artifact ID, game build, effective time, power level, per-map revision,
and source hashes make the assumptions auditable. A new bundle is written to a temporary path,
completely decoded and validated, then atomically replaces the prior bundle. An absent, corrupt,
oversized, incompatible, or partially covered bundle disables the world read without breaking the
API; it must not mix map geometry from one patch with mechanics from another.

Historical training and evaluation must join a match to the mechanics snapshot active at that time.
Applying today's damage or geometry to an old match would create silent label leakage and corrupt
the measurement. When a matching historical snapshot is unavailable, that match is excluded from
mechanics evaluation rather than guessed.

## First build

The initial implementation surface is intentionally small:

- `backend/bsdraft/world/__init__.py` — the package's public imports.
- `backend/bsdraft/world/schema.py` — frozen value objects, strict schema validation, bounded bundle
  loading, and coverage lookups.
- `backend/bsdraft/world/geometry.py` — deterministic line-of-sight, clearance-aware traversal, and
  attack-reach primitives.
- `backend/bsdraft/world/evaluator.py` — factorized, team-order-invariant and explicitly
  antisymmetric mechanics evaluation.
- `backend/scripts/bootstrap_map_geometry.py` — source-image/bootstrap support that produces data
  for human verification, not automatic production truth.
- `docs/world-model-source-probes.json` — reproducible metadata for real source-image transform
  probes; it is evidence about source bytes, not verified collision geometry.
- A test-only synthetic bundle under `backend/tests/fixtures/` and
  `backend/tests/test_world_model.py` — schema, geometry, symmetry, resource-bound, and degradation
  contracts.
- This document and the one-line `CLAUDE.md` router entry.

The loader is fail-soft at the application boundary. Strict payload validation raises `ValueError`
internally; the public loader returns no model for a missing, unreadable, malformed, unsupported, or
over-limit bundle. Unknown maps, wrong map/mode pairs, and uncovered brawlers produce an unavailable
world read rather than a fabricated neutral number.

## Rollout and validation gate

The first build stays **dark and disconnected from serving**. Its evaluator can run offline against
the synthetic contract fixture, but the API and recommendation engine neither load nor call it.
World-model code stays separate from `score_candidate`, `DEFAULT_WEIGHTS`, and
`PickScore.breakdown`; attaching an object to an engine cannot alter pick IDs, ordering, base scores,
final scores, or breakdowns. It cannot remove a candidate or turn missing coverage into a weighted
`0.5`. A later phase will add an explicit diagnostic seam and run verified data in true shadow mode.

Before any scoring weight is enabled, evaluate three systems on the same data:

1. mechanics-only;
2. the current empirical system;
3. a hybrid that adds mechanics as a residual or explicitly weighted feature.

Evaluation must use temporal holdouts across balance and map changes, plus cold-start slices for new
or reworked brawlers. Report log loss, AUC, calibration error, ranking quality where measurable, and
coverage. The useful question is not whether the mechanics formula can explain one showcase matchup;
it is whether it improves repeated unseen patches without damaging calibration. Any live scoring
weight requires evidence from that gate and the normal held-out signal-weight evaluation documented
in `docs/model-evaluation.md`.

## Build phases

1. **Foundation:** strict patch-aware schema, bounded fail-soft loader, synthetic fixtures.
2. **Spatial kernel:** line of sight, clearance-aware pathing, attack reach, deterministic tests.
3. **Verified data:** source and review real map geometry and literal mechanics; publish an atomic
   bundle only when coverage is honest.
4. **Derived reasoning:** range bands, influence fields, burst and survivability, objective pressure,
   pair interactions, and human-readable explanations.
5. **Shadow evaluation:** temporal and cold-start comparison against the empirical model.
6. **Hybrid rollout:** add scoring influence only if the evaluation gate passes; preserve
   antisymmetry, calibration, explainability, and graceful degradation.
7. **Dynamic research, optional:** coarse engagement transitions first; tick-level simulation or
   self-play only if evidence shows the static model's ceiling warrants the compute and complexity.

## First-build non-goals

- No live recommendation-score or ranking changes.
- No guessed production mechanics or automatically inferred collision map presented as exact.
- No tick-by-tick six-player simulation, aim/dodge policy, or self-play training.
- No raw-image CNN/GNN inference in the deployed API.
- No exhaustive composition, position-pair, or draft-tree tensor.
- No promise that literal stats alone reproduce player behavior or calibrated win probability.
- No new always-on cloud service, GPU requirement, or serve-time dependency outside the existing
  lightweight deployment tier.
