"""Unit tests for the post-draft game plan (bsdraft.engine.gameplan).

Covers the data-backed half added on top of the original heuristic plan: that thin cells are
dropped rather than shown at low confidence, that the map read only names an anchor / weak link
when the spread is bigger than noise, that the model's read is withheld on an unfinished draft,
and that the whole data half degrades to the legacy heuristic plan with no stats or model.
Stub stats keep the assertions off the live dataset.

    PYTHONPATH=backend python -m pytest backend/tests/test_gameplan.py
"""
from __future__ import annotations

from itertools import permutations

from bsdraft.api.schemas import GamePlan as GamePlanSchema
from bsdraft.engine.gameplan import MIN_CELL_GAMES, MIN_MAP_GAMES, game_plan
from bsdraft.engine.state import DraftState

SHELLY, COLT, BULL = 16000000, 16000001, 16000002        # Damage Dealer, Damage Dealer, Tank
JESSIE, DYNAMIKE, MORTIS = 16000007, 16000009, 16000011  # Controller, Artillery, Assassin
EL_PRIMO = 16000010                                      # Tank
MODE, MAP = "Knockout", 15000001

FULL = dict(map_id=MAP, mode=MODE, our_team=[SHELLY, COLT, BULL], their_team=[JESSIE, DYNAMIKE, MORTIS])

SPROUT, RT = 16000037, 16000066                          # Artillery, Damage Dealer
GRIFF, BIBI, ASH = 16000050, 16000026, 16000051          # Controller, Tank, Tank
PARALLEL_PLAYS = 15000293
PARALLEL_FULL = dict(
    map_id=PARALLEL_PLAYS,
    mode="Hot Zone",
    our_team=[SPROUT, RT, BULL],
    their_team=[GRIFF, BIBI, ASH],
)


class _Rate:
    def __init__(self, winrate, games=100.0):
        self.winrate = winrate
        self.games = games
        self.confidence = games / (games + 20.0)


class _StubStats:
    """Per-cell rates with an explicit sample, so a test can push one cell under a threshold."""

    def __init__(self, map_rates=None, counters=None, synergies=None, games=100.0):
        self._map = map_rates or {}
        self._counters = counters or {}
        self._syn = synergies or {}
        self._games = games

    def brawler_rate(self, bid, map_id=None):
        wr, g = self._map.get(bid, (0.5, self._games))
        return _Rate(wr, g)

    def counter(self, attacker, defender):
        wr, g = self._counters.get((attacker, defender), (0.5, self._games))
        return _Rate(wr, g)

    def synergy(self, a, b):
        wr, g = self._syn.get(frozenset((a, b)), (0.5, self._games))
        return _Rate(wr, g)


class _Model:
    available = True

    def __init__(self, prob=0.6):
        self._prob = prob
        self.calls = 0

    def prob(self, our, their, map_id, mode):
        self.calls += 1
        return self._prob


def _assignments_by_name(plan):
    return {assignment["name"]: assignment for assignment in plan["assignments"]}


# --- the structured opening-assignment contract ---

def test_parallel_plays_assigns_exact_opening_jobs_and_rt_tracks_bibi_without_stats():
    """The motivating draft must be actionable even without the measured data layer."""
    gp = game_plan(DraftState(**PARALLEL_FULL))
    assignments = _assignments_by_name(gp)
    trackers = [a for a in gp["assignments"] if a["tracks"] is not None]

    assert gp["formation"]
    assert set(assignments) == {"Sprout", "R-T", "Bull"}
    assert [a["name"] for a in trackers] == ["R-T"]
    assert assignments["R-T"]["start"] == "home_zone"
    assert assignments["R-T"]["tracks"] == "Bibi"
    assert assignments["Bull"]["start"] == "away_zone"
    assert assignments["Bull"]["tracks"] is None
    assert "R-T" in assignments["Bull"]["adjust"]
    assert "Bibi" in assignments["Bull"]["adjust"]
    assert assignments["Sprout"]["start"] == "flex"
    assert assignments["Sprout"]["tracks"] is None
    assert assignments["Sprout"]["adjust"] == ""
    assert all(a["position"] and a["job"] for a in assignments.values())
    assert "bottom-left" in assignments["R-T"]["job"]
    assert "full reset or respawn" in assignments["R-T"]["adjust"]
    assert "hold far until R-T starts crossing" in assignments["Bull"]["adjust"]
    assert "center fence" in assignments["Sprout"]["job"]
    assert "unless the enemy can finish near" in gp["formation"]


def test_parallel_plays_curated_tracker_beats_adverse_full_head_to_head_data():
    counters = {
        (ours, theirs): (0.90, 500.0)
        for ours in PARALLEL_FULL["our_team"]
        for theirs in PARALLEL_FULL["their_team"]
    }
    counters[(RT, BIBI)] = (0.10, 500.0)

    gp = game_plan(DraftState(**PARALLEL_FULL), _StubStats(counters=counters))
    trackers = [a for a in gp["assignments"] if a["tracks"] is not None]

    assert gp["head_to_head"]["best"]["ours"] != "R-T"
    assert [(a["name"], a["tracks"]) for a in trackers] == [("R-T", "Bibi")]


def test_head_to_head_results_never_become_positional_instructions():
    counters = {
        (ours, theirs): (0.46, 500.0)
        for ours in FULL["our_team"]
        for theirs in FULL["their_team"]
    }
    counters[(SHELLY, JESSIE)] = (0.65, 5000.0)

    gp = game_plan(DraftState(**FULL), _StubStats(counters=counters))

    assert gp["head_to_head"]["best"]["games"] == 5000.0
    assert gp["head_to_head"]["best"]["edge"] == "strong"
    assert all(a["tracks"] is None for a in gp["assignments"])
    assert all(a["adjust"] == "" for a in gp["assignments"])


def test_measured_map_form_does_not_move_rule_based_opening_jobs():
    state = DraftState(**FULL)
    expected = {a["name"]: a["start"] for a in game_plan(state)["assignments"]}
    stats = _StubStats(map_rates={
        SHELLY: (0.90, 5000.0),
        COLT: (0.10, 5000.0),
        BULL: (0.50, 5000.0),
    })

    actual = {a["name"]: a["start"] for a in game_plan(state, stats)["assignments"]}

    assert actual == expected


def test_parallel_plays_assignments_do_not_depend_on_request_order():
    """Draft-slot order is UI state, not a tactical signal; the same six names get the same jobs."""
    expected = {
        name: (assignment["start"], assignment["tracks"])
        for name, assignment in _assignments_by_name(
            game_plan(DraftState(**PARALLEL_FULL))
        ).items()
    }

    for our_team in permutations(PARALLEL_FULL["our_team"]):
        for their_team in permutations(PARALLEL_FULL["their_team"]):
            state = DraftState(
                map_id=PARALLEL_PLAYS,
                mode="Hot Zone",
                our_team=list(our_team),
                their_team=list(their_team),
            )
            actual = {
                name: (assignment["start"], assignment["tracks"])
                for name, assignment in _assignments_by_name(game_plan(state)).items()
            }
            assert actual == expected


def test_parallel_plays_blind_pick_keeps_positions_but_drops_matchup_targets():
    gp = game_plan(DraftState(
        map_id=PARALLEL_PLAYS,
        mode="Hot Zone",
        our_team=list(PARALLEL_FULL["our_team"]),
    ))
    assignments = _assignments_by_name(gp)

    assert {name: a["start"] for name, a in assignments.items()} == {
        "R-T": "home_zone",
        "Bull": "away_zone",
        "Sprout": "flex",
    }
    assert all(a["tracks"] is None and a["adjust"] == "" for a in assignments.values())


def test_unknown_hot_zone_map_uses_mode_jobs_without_fabricating_coordinates():
    gp = game_plan(DraftState(
        map_id=99999999,
        mode="Hot Zone",
        our_team=list(PARALLEL_FULL["our_team"]),
        their_team=list(PARALLEL_FULL["their_team"]),
    ))

    starts = {a["start"] for a in gp["assignments"]}
    assert starts == {"zone_anchor", "off_angle", "flex"}
    assert not ({"home_zone", "away_zone"} & starts)
    rendered = " ".join([
        gp["formation"],
        *(a["position"] for a in gp["assignments"]),
        *(a["job"] for a in gp["assignments"]),
    ]).lower()
    assert "bottom-left" not in rendered and "top-right" not in rendered


def test_partial_team_has_no_formation_or_assignments():
    for our_team in ([], [SPROUT], [SPROUT, RT]):
        gp = game_plan(DraftState(
            map_id=PARALLEL_PLAYS,
            mode="Hot Zone",
            our_team=our_team,
            their_team=list(PARALLEL_FULL["their_team"]),
        ))
        assert gp["formation"] == ""
        assert gp["assignments"] == []


def test_parallel_plays_geometry_is_guarded_by_mode():
    gp = game_plan(DraftState(
        map_id=PARALLEL_PLAYS,
        mode="Knockout",
        our_team=list(PARALLEL_FULL["our_team"]),
        their_team=list(PARALLEL_FULL["their_team"]),
    ))

    starts = {a["start"] for a in gp["assignments"]}
    assert starts == {"safe_sightline", "crossfire", "trade"}
    assert not ({"home_zone", "away_zone"} & starts)
    assert "bottom-left" not in gp["formation"].lower()
    assert "top-right" not in gp["formation"].lower()


def test_game_plan_schema_serialization_retains_structured_assignments():
    wire = GamePlanSchema(**game_plan(DraftState(**PARALLEL_FULL))).model_dump()
    assignments = {assignment["name"]: assignment for assignment in wire["assignments"]}

    assert wire["formation"]
    assert assignments["R-T"]["start"] == "home_zone"
    assert assignments["R-T"]["tracks"] == "Bibi"
    assert assignments["Bull"]["start"] == "away_zone"
    assert assignments["Sprout"]["start"] == "flex"


def test_no_stats_or_model_is_the_legacy_heuristic_plan():
    """The data half must be optional: every caller that passes only a state still gets the
    plan it always got, with the new keys present but empty."""
    gp = game_plan(DraftState(**FULL))
    assert gp["win_condition"] and gp["roles"] and gp["tips"]        # heuristic half intact
    assert gp["map_read"] == [] and gp["pairs"] == []
    assert gp["head_to_head"] is None and gp["model_read"] is None


def test_head_to_head_grid_is_ours_by_theirs():
    stats = _StubStats(counters={(SHELLY, JESSIE): (0.62, 500.0), (BULL, MORTIS): (0.41, 500.0)})
    h2h = game_plan(DraftState(**FULL), stats)["head_to_head"]
    assert [r["enemy"] for r in h2h["grid"]] == ["Jessie", "Dynamike", "Mortis"]
    assert [c["name"] for c in h2h["grid"][0]["vs"]] == ["Shelly", "Colt", "Bull"]
    assert h2h["best"]["ours"] == "Shelly" and h2h["best"]["theirs"] == "Jessie"
    assert h2h["danger"]["ours"] == "Bull" and h2h["danger"]["theirs"] == "Mortis"
    assert h2h["focus"]["enemy"] == "Mortis"        # lowest row mean, and below even


def test_thin_head_to_head_cell_is_blank_not_low_confidence():
    """A cell under the sample floor is a hole in the grid — never a number the reader would
    take at face value."""
    stats = _StubStats(counters={(SHELLY, JESSIE): (0.75, MIN_CELL_GAMES - 1)})
    row = game_plan(DraftState(**FULL), stats)["head_to_head"]["grid"][0]
    assert row["vs"][0] == {"name": "Shelly", "winrate": None, "games": 0.0, "edge": "unknown"}
    assert row["mean"] is not None and 0.49 < row["mean"] < 0.51   # mean skips the dropped cell


def test_focus_withheld_when_no_enemy_beats_us():
    stats = _StubStats(counters={(o, e): (0.58, 500.0) for o in (SHELLY, COLT, BULL)
                                 for e in (JESSIE, DYNAMIKE, MORTIS)})
    assert game_plan(DraftState(**FULL), stats)["head_to_head"]["focus"] is None


def test_map_read_names_an_anchor_only_on_a_real_spread():
    tight = _StubStats(map_rates={SHELLY: (0.545, 500.0), COLT: (0.54, 500.0), BULL: (0.535, 500.0)})
    assert {r["tag"] for r in game_plan(DraftState(**FULL), tight)["map_read"]} == {"solid"}

    wide = _StubStats(map_rates={SHELLY: (0.57, 500.0), COLT: (0.52, 500.0), BULL: (0.46, 500.0)})
    tags = {r["name"]: r["tag"] for r in game_plan(DraftState(**FULL), wide)["map_read"]}
    assert tags == {"Shelly": "anchor", "Colt": "solid", "Bull": "weak"}


def test_map_read_drops_a_thin_brawler():
    stats = _StubStats(map_rates={BULL: (0.9, MIN_MAP_GAMES - 1)})
    assert [r["name"] for r in game_plan(DraftState(**FULL), stats)["map_read"]] == ["Shelly", "Colt"]


def test_pairs_are_sorted_strongest_first():
    stats = _StubStats(synergies={frozenset((SHELLY, COLT)): (0.55, 500.0),
                                  frozenset((SHELLY, BULL)): (0.61, 500.0)})
    pairs = game_plan(DraftState(**FULL), stats)["pairs"]
    assert [(p["a"], p["b"]) for p in pairs][0] == ("Shelly", "Bull")
    assert [p["winrate"] for p in pairs] == sorted((p["winrate"] for p in pairs), reverse=True)


def test_model_read_only_on_a_finished_draft():
    """A partial-board number is a marginal over how drafts usually continue — right for ranking
    picks, wrong to hand a player as "your odds", so the model isn't even called."""
    stats, model = _StubStats(), _Model(0.61)
    partial = game_plan(DraftState(map_id=MAP, mode=MODE, our_team=[SHELLY, COLT],
                                   their_team=[JESSIE]), stats, model)
    assert partial["model_read"] is None and model.calls == 0

    done = game_plan(DraftState(**FULL), stats, model)
    assert done["model_read"]["win_prob"] == 0.61 and model.calls == 1
    assert done["model_read"]["verdict"] == "The draft is on your side"


def test_blind_pick_drops_every_enemy_section():
    stats = _StubStats()
    gp = game_plan(DraftState(map_id=MAP, mode=MODE, our_team=[SHELLY, COLT, BULL]), stats, _Model())
    assert gp["head_to_head"] is None and gp["enemy"] is None and gp["model_read"] is None
    assert len(gp["map_read"]) == 3 and len(gp["pairs"]) == 3   # our own half still reads


def test_enemy_shape_needs_two_picks():
    stats = _StubStats()
    one = game_plan(DraftState(map_id=MAP, mode=MODE, our_team=[SHELLY], their_team=[EL_PRIMO]), stats)
    assert one["enemy"] is None
    two = game_plan(DraftState(map_id=MAP, mode=MODE, our_team=[SHELLY, COLT],
                               their_team=[EL_PRIMO, MORTIS]), stats)
    assert two["enemy"]["archetype"] == "Aggressive" and two["enemy"]["clash"]


def test_callouts_discount_a_thin_cell():
    """The headline cells are an argmax over the grid, so a thin cell must not win on variance:
    a 62% off ~130 matches should lose the "lean on" slot to a 58% off ~13,000."""
    stats = _StubStats(counters={(SHELLY, JESSIE): (0.62, 130.0), (COLT, JESSIE): (0.58, 13000.0),
                                 (BULL, MORTIS): (0.38, 130.0), (BULL, DYNAMIKE): (0.42, 13000.0)})
    h2h = game_plan(DraftState(**FULL), stats)["head_to_head"]
    assert h2h["best"]["ours"] == "Colt" and h2h["best"]["winrate"] == 0.58
    assert h2h["danger"]["theirs"] == "Dynamike" and h2h["danger"]["winrate"] == 0.42


def test_a_thin_cell_still_wins_when_it_is_genuinely_extreme():
    """The discount is a bar to clear, not a sample-size sort — a big enough gap still promotes
    the smaller sample."""
    stats = _StubStats(counters={(SHELLY, JESSIE): (0.74, 130.0), (COLT, JESSIE): (0.55, 13000.0)})
    assert game_plan(DraftState(**FULL), stats)["head_to_head"]["best"]["ours"] == "Shelly"


# --- the callout contract: split at even, so the two headlines can never be the same cell ---

def test_best_and_danger_can_never_be_the_same_cell():
    """Regression: `best` and `danger` used to be independent argmaxes over the whole grid, so a
    single deep-sampled cell won both and the panel printed "lean on X" beside "risk X" with
    identical numbers. Splitting the pool at even makes the two sets disjoint by construction."""
    # One cell, sampled far deeper than the floor — the exact shape that used to collide.
    stats = _StubStats(counters={(SHELLY, JESSIE): (0.571, 5000.0)}, games=MIN_CELL_GAMES - 1)
    h2h = game_plan(DraftState(map_id=MAP, mode=MODE, our_team=[SHELLY],
                               their_team=[JESSIE]), stats)["head_to_head"]
    assert h2h["best"]["ours"] == "Shelly"      # it wins, so it can only be the "lean on"
    assert h2h["danger"] is None                # ...and there is nothing to call a risk


def test_lean_on_is_never_a_losing_matchup():
    """A green "lean on" on a sub-even cell contradicted the grid's own colour for that same
    cell. When every cell is a loss there is simply no `best` to show."""
    losing = {(o, e): (0.46, 500.0) for o in (SHELLY, COLT, BULL) for e in (JESSIE, DYNAMIKE, MORTIS)}
    h2h = game_plan(DraftState(**FULL), _StubStats(counters=losing))["head_to_head"]
    assert h2h["best"] is None
    assert h2h["danger"] is not None and h2h["danger"]["winrate"] < 0.50


def test_risk_is_never_a_winning_matchup():
    winning = {(o, e): (0.56, 500.0) for o in (SHELLY, COLT, BULL) for e in (JESSIE, DYNAMIKE, MORTIS)}
    h2h = game_plan(DraftState(**FULL), _StubStats(counters=winning))["head_to_head"]
    assert h2h["danger"] is None
    assert h2h["best"] is not None and h2h["best"]["winrate"] > 0.50


def test_focus_needs_more_than_one_surviving_cell():
    """A row averaging a single cell is one matchup wearing the word "overall", so it must not
    carry a callout that claims something about the whole comp."""
    counters = {(SHELLY, JESSIE): (0.30, 500.0)}          # only this cell clears the floor
    h2h = game_plan(DraftState(**FULL), _StubStats(counters=counters, games=MIN_CELL_GAMES - 1))["head_to_head"]
    jessie = next(r for r in h2h["grid"] if r["enemy"] == "Jessie")
    assert jessie["mean_cells"] == 1
    assert h2h["focus"] is None

    # Two surviving cells in the row -> it may speak, and carries its combined sample.
    counters[(COLT, JESSIE)] = (0.34, 500.0)
    h2h2 = game_plan(DraftState(**FULL), _StubStats(counters=counters, games=MIN_CELL_GAMES - 1))["head_to_head"]
    assert h2h2["focus"]["enemy"] == "Jessie"
    assert h2h2["focus"]["cells"] == 2 and h2h2["focus"]["games"] == 1000.0


def test_row_average_reports_how_many_cells_it_covers():
    counters = {(SHELLY, JESSIE): (0.40, 200.0), (COLT, JESSIE): (0.60, 300.0)}
    row = game_plan(DraftState(**FULL), _StubStats(counters=counters, games=MIN_CELL_GAMES - 1))["head_to_head"]["grid"][0]
    assert row["mean_cells"] == 2 and row["mean_games"] == 500.0
    assert row["mean"] == 0.50


def test_oversized_teams_are_truncated_not_walked():
    """`/api/recommend` takes bare int lists and `game_plan` runs on every call, so an oversized
    body must not turn into a quadratic grid. A real board is never more than three a side."""
    big = list(range(16000000, 16000000 + 400))
    gp = game_plan(DraftState(map_id=MAP, mode=MODE, our_team=big, their_team=big), _StubStats(), _Model())
    assert len(gp["roles"]) == 3 and len(gp["threats"]) <= 3
    assert len(gp["head_to_head"]["grid"]) == 3
    assert all(len(r["vs"]) == 3 for r in gp["head_to_head"]["grid"])
    assert len(gp["map_read"]) == 3 and len(gp["pairs"]) == 3
    assert len(gp["assignments"]) == 3
    assert {a["name"] for a in gp["assignments"]} == {"Shelly", "Colt", "Bull"}
