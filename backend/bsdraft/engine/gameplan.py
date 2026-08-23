"""Post-draft game plan.

Two layers, kept distinguishable in the output so the UI can label them:

* **Heuristic** — win condition, per-brawler roles, mode do's/don'ts, how to cover a
  composition hole. Rule-based, because the match data is draft->outcome only: it has no
  positions, rotations or timings, so nothing in it could teach a model how to *play* the
  mode. This is standard mode/role strategy and is presented as such.
* **Data-backed** — the head-to-head grid, per-brawler map form, ally pair rates and the
  model's read on the finished draft. These come from the same collected matches and the
  same win-prob model that drive the pick board, and every number ships with its effective
  (recency-weighted) sample so a thin cell can be discounted rather than trusted.

Both degrade independently: with no ``stats``/``model`` the plan is exactly the heuristic
plan it has always been, and each data section drops out on its own when its cells are too
thin to say anything.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from bsdraft.constants import TEAM_SIZE
from bsdraft.engine.scoring import _class_of, _name_map
from bsdraft.engine.state import DraftState

MODE_PLAN: Dict[str, dict] = {
    "Gem Grab": {
        "objective": "Hold 10 gems through the 15-second countdown.",
        "tips": [
            "Control the center mine and assign a durable gem carrier to protect.",
            "When you hit 10, back off and play defensively for the timer.",
            "Bait out enemy Supers before committing to a fight.",
        ],
        "avoid": [
            "Dying with gems — especially near the countdown.",
            "Letting one player hoard all the gems out front.",
        ],
    },
    "Brawl Ball": {
        "objective": "Score 2 goals (or lead when time runs out).",
        "tips": [
            "Win the ball with your tank/assassin and use walls to open lanes.",
            "Save Supers for scoring or last-ditch defense.",
            "Keep one player back to defend your net.",
        ],
        "avoid": [
            "Throwing the ball blindly into a crowd.",
            "Everyone pushing up and leaving an open net.",
        ],
    },
    "Knockout": {
        "objective": "Win 2 of 3 rounds — no respawns, so every life is precious.",
        "tips": [
            "Play for picks: catch enemies out of position and win the numbers game.",
            "Use bushes and cover; poke from range before committing.",
            "Rotate together — don't get isolated.",
        ],
        "avoid": [
            "Face-checking bushes alone.",
            "Over-aggression — one death cripples the whole round.",
        ],
    },
    "Heist": {
        "objective": "Break the enemy safe before they break yours.",
        "tips": [
            "Bring burst to the safe when the enemy Supers are down.",
            "Split pressure to force them to choose what to defend.",
            "Track enemy Supers and push on the gap.",
        ],
        "avoid": [
            "Leaving your safe undefended.",
            "Feeding Supers by trading badly.",
        ],
    },
    "Hot Zone": {
        "objective": "Control the zone(s) to fill your meter faster.",
        "tips": [
            "Use area-control brawlers to hold the zone and deny theirs.",
            "Rotate to contest, but only fight where you can win.",
            "Stack the zone when ahead to close it out.",
        ],
        "avoid": [
            "Chasing kills off the zone.",
            "Clumping into throwers / AoE.",
        ],
    },
    "Bounty": {
        "objective": "Lead on stars when time runs out (7-star cap).",
        "tips": [
            "Poke from range and pick off high-bounty targets.",
            "When ahead, play safe and protect your star lead.",
            "Group up early to avoid giving first blood.",
        ],
        "avoid": [
            "Feeding stars by over-extending.",
            "Fighting at a star disadvantage.",
        ],
    },
}

ROLE_BY_CLASS = {
    "Tank": "Frontline — soak damage, create space, dive their backline.",
    "Assassin": "Flanker — catch isolated targets and delete their carry.",
    "Marksman": "Backline — hold range and deal damage from safety.",
    "Controller": "Zoner — lock down chokes and control space.",
    "Artillery": "Poke — chip from range and deny areas with throws.",
    "Support": "Enabler — peel for and pocket your carry.",
    "Damage Dealer": "Damage — trade from cover and burst priority targets.",
    "Unclassified": "Flex — play to this brawler's strengths.",
}

THREAT_BY_CLASS = {
    "Tank": "kite {name} and keep distance; focus it when its Super is down.",
    "Assassin": "group up and watch bushes for {name}; don't get caught alone or low.",
    "Marksman": "use cover against {name}; don't walk open lanes — dive it with frontline.",
    "Controller": "respect {name}'s zoning/CC and don't clump up; flank when you can.",
    "Artillery": "close the distance on {name} or break its walls — don't sit in its throw zone.",
    "Support": "pressure or dive {name}, or focus the brawler it's pocketing.",
    "Damage Dealer": "respect {name}'s burst; trade from cover, don't face-tank it.",
}

_TONE = {
    "Aggressive": "Be proactive",
    "Control / poke": "Be patient",
    "Balanced": "Stay flexible",
}
_MODE_VERB = {
    "Gem Grab": "control the mine and protect your carry",
    "Brawl Ball": "win the ball and convert with your frontline",
    "Knockout": "play for picks and win the numbers game",
    "Heist": "burst the safe when their Supers are down",
    "Hot Zone": "hold the zone and out-rotate them",
    "Bounty": "out-poke them and protect your star lead",
}

# How our comp's shape plays against theirs. Keyed (ours, theirs); the shapes come from
# `_archetype`, so this is a 3x3 over {Aggressive, Control / poke, Balanced}.
_CLASH = {
    ("Aggressive", "Aggressive"):
        "Both comps want the fight. Whoever engages on their terms wins — bait a Super out before you commit.",
    ("Aggressive", "Control / poke"):
        "They want to poke you out from range. Every second you spend at their range is theirs — close it, use cover, and go on their reload.",
    ("Aggressive", "Balanced"):
        "You're the aggressor. Force the tempo before their backline gets set, and don't let the game settle into a poke war.",
    ("Control / poke", "Aggressive"):
        "They're built to dive you. Never be alone, hold cover, and make them walk through your damage to reach you.",
    ("Control / poke", "Control / poke"):
        "A poke war. Win trades from cover, track reloads, and take the objective while they're backing off to heal.",
    ("Control / poke", "Balanced"):
        "Out-range them at rest and punish anyone who steps out — but don't get dragged into a close fight with their frontline.",
    ("Balanced", "Aggressive"):
        "Expect early aggression. Absorb the first engage without giving a free kill, then punish the over-extension.",
    ("Balanced", "Control / poke"):
        "They out-range you standing still. Break the stalemate with your frontline and cover — don't trade at their range.",
    ("Balanced", "Balanced"):
        "Even shapes. This comes down to positioning and Super timing on the objective, not the draft.",
}

# Effective (recency-weighted) sample a cell needs before the plan will speak from it. Below
# these the number is noise dressed up as a read, so the section simply drops the cell.
MIN_CELL_GAMES = 12.0    # one (ours, theirs) head-to-head cell
MIN_PAIR_GAMES = 12.0    # one ally pair
MIN_MAP_GAMES = 15.0     # one (brawler, map) cell
MIN_MAP_SPREAD = 0.02    # best-worst gap before the map read names an anchor / weak link
CALLOUT_Z = 2.0          # sigmas a thin cell must clear to take a head-to-head headline


def _archetype(classes: List[str]):
    aggro = sum(c in ("Tank", "Assassin") for c in classes)
    rangey = sum(c in ("Marksman", "Controller", "Artillery") for c in classes)
    if aggro >= 2 and rangey == 0:
        return "Aggressive", "Force fights and close distance — snowball early picks and pressure the objective."
    if rangey >= 2 and aggro == 0:
        return "Control / poke", "Play your range advantage: poke, zone, and punish over-extension. Avoid melee brawls."
    return "Balanced", "Let your frontline engage and your backline follow up — win the range war, then commit."


def _edge(winrate: float) -> str:
    """Bucket a win rate into a verdict the UI can colour by. The bands are deliberately wide:
    a smoothed rate inside 48-52% is not distinguishable from even at these sample sizes."""
    if winrate >= 0.55:
        return "strong"
    if winrate >= 0.52:
        return "favored"
    if winrate > 0.48:
        return "even"
    if winrate > 0.45:
        return "unfavored"
    return "losing"


def _bound(cell: dict, sign: int) -> float:
    """A pessimistic view of a cell's rate, used *only* to pick the `best`/`danger` callouts —
    each cell still displays its own smoothed rate.

    Taking the plain argmax over the grid is a max-selection: a cell only has to clear
    `MIN_CELL_GAMES`, so a thin one with wide error bars can win the headline on variance
    alone — 62% off ~130 matches outranking a better-established 58% off ~13,000. Ranking on
    ``winrate -/+ CALLOUT_Z * se`` makes a big claim from a small sample clear a proportionally
    bigger bar. At two sigma a near-tie like that one resolves to the deep sample, while a cell
    that is genuinely extreme (say 74% off the same ~130) still takes the slot. ``se`` is the
    worst-case binomial standard error, ``sqrt(0.25 / n)``.
    """
    games = cell.get("games") or 0.0
    se = (0.25 / games) ** 0.5 if games > 0 else 0.5
    return cell["winrate"] + sign * CALLOUT_Z * se


def _map_read(state: DraftState, stats, names) -> List[dict]:
    """Per-brawler form on *this map*, so the plan can name who to play through and who is
    along for the ride. Same `(brawler, map)` cell the pick board's `map` component scores."""
    rows = []
    for b in state.our_team[:TEAM_SIZE]:
        rate = stats.brawler_rate(b, state.map_id)
        if rate.games < MIN_MAP_GAMES:
            continue
        rows.append({"name": names.get(b, str(b)), "cls": _class_of(b),
                     "winrate": round(rate.winrate, 4), "games": round(rate.games, 1), "tag": "solid"})
    if len(rows) < 2:
        return rows
    rows.sort(key=lambda r: r["winrate"], reverse=True)
    best, worst = rows[0], rows[-1]
    # Only call one of them out when the spread is bigger than the noise: three brawlers within
    # a point of each other are the same brawler as far as this map is concerned, and naming a
    # "weak link" there would invent a hierarchy the data doesn't have.
    if best["winrate"] - worst["winrate"] >= MIN_MAP_SPREAD:
        if best["winrate"] >= 0.51:
            best["tag"] = "anchor"
        if worst["winrate"] <= 0.49:
            worst["tag"] = "weak"
    return rows


def _pairs(state: DraftState, stats, names) -> List[dict]:
    """Every ally pair we drafted, best first — which two actually want to play together."""
    rows = []
    team = list(state.our_team[:TEAM_SIZE])
    for i in range(len(team)):
        for j in range(i + 1, len(team)):
            rate = stats.synergy(team[i], team[j])
            if rate.games < MIN_PAIR_GAMES:
                continue
            rows.append({"a": names.get(team[i], str(team[i])), "b": names.get(team[j], str(team[j])),
                         "winrate": round(rate.winrate, 4), "games": round(rate.games, 1),
                         "edge": _edge(rate.winrate)})
    rows.sort(key=lambda r: r["winrate"], reverse=True)
    return rows


def _head_to_head(state: DraftState, stats, names) -> Optional[dict]:
    """The full grid of our brawlers against theirs, plus the two cells worth acting on.

    ``stats.counter(ours, theirs)`` is our *side's* win rate across matches where those two were
    on opposite teams — a team outcome attributed to the pairing, not a duel result. It still
    reads as "which of ours is comfortable into which of theirs", which is what the plan needs,
    which is why the copy downstream says "with both on the board" rather than "wins the 1v1".

    Returns the grid, the enemy our comp does worst against overall (`focus`), our single worst
    cell (`danger`) and our single best (`best`). Cells thinner than `MIN_CELL_GAMES` are dropped
    rather than shown at low confidence, so a sparse grid simply has holes in it.

    `best` and `danger` are each side of even, never merely the ends of the range: a callout that
    says "lean on this" has to name a matchup we actually win, and one that says "this is the
    risk" has to name one we actually lose. Ranking the extremes without that floor put a green
    "lean on" on a 48% cell the grid itself had already coloured as unfavourable, and — because
    the two argmaxes are independent — let a single deep-sampled cell win *both* slots and print
    itself as the thing to lean on and the thing to fear side by side. Splitting the pool at 0.50
    fixes both: the sets are disjoint by construction, so the collision cannot recur.
    """
    our = state.our_team[:TEAM_SIZE]
    their = state.their_team[:TEAM_SIZE]
    if not our or not their:
        return None
    grid, cells = [], []
    for e in their:
        row = {"enemy": names.get(e, str(e)), "enemy_cls": _class_of(e), "vs": []}
        rates, sample = [], 0.0
        for o in our:
            rate = stats.counter(o, e)
            if rate.games < MIN_CELL_GAMES:
                row["vs"].append({"name": names.get(o, str(o)), "winrate": None, "games": 0.0, "edge": "unknown"})
                continue
            cell = {"name": names.get(o, str(o)), "winrate": round(rate.winrate, 4),
                    "games": round(rate.games, 1), "edge": _edge(rate.winrate)}
            row["vs"].append(cell)
            rates.append(rate.winrate)
            sample += rate.games
            cells.append({**cell, "ours": names.get(o, str(o)), "theirs": names.get(e, str(e))})
        # The average is over the cells that survived the floor, so it carries how many of them
        # there were: a row averaging one surviving cell is not the same claim as one averaging
        # three, and the UI has to be able to say so rather than printing a bare "AVG".
        row["mean"] = round(sum(rates) / len(rates), 4) if rates else None
        row["mean_cells"] = len(rates)
        row["mean_games"] = round(sample, 1)
        grid.append(row)
    if not cells:
        return None
    # A row that averages a single surviving cell is one matchup wearing the word "overall", so it
    # can't carry the focus callout — that headline claims something about the whole comp.
    scored = [r for r in grid if r["mean"] is not None and r["mean_cells"] >= 2]
    focus = min(scored, key=lambda r: r["mean"]) if scored else None
    wins = [c for c in cells if c["winrate"] > 0.50]
    losses = [c for c in cells if c["winrate"] < 0.50]
    return {
        "grid": grid,
        "focus": {"enemy": focus["enemy"], "enemy_cls": focus["enemy_cls"], "winrate": focus["mean"],
                  "games": focus["mean_games"], "cells": focus["mean_cells"]}
        if focus and focus["mean"] < 0.50 else None,
        "danger": min(losses, key=lambda c: _bound(c, +1)) if losses else None,
        "best": max(wins, key=lambda c: _bound(c, -1)) if wins else None,
    }


_MODEL_VERDICT = [
    (0.57, "The draft is on your side"),
    (0.53, "Slight draft edge to you"),
    (0.47, "Coin flip on the draft alone"),
    (0.43, "Slight draft deficit"),
    (0.00, "Drafted into an uphill match"),
]


def _model_read(state: DraftState, model) -> Optional[dict]:
    """The win-prob model's read on the finished 3v3 — the same net the pick board's `model`
    component comes from, run once on the final board instead of per candidate.

    Only for a complete draft: on a partial board the number is a marginal over how drafts
    usually continue, which is the right thing to *rank picks* with and the wrong thing to
    hand a player as "your odds"."""
    if model is None or not getattr(model, "available", False):
        return None
    if len(state.our_team) < TEAM_SIZE or len(state.their_team) < TEAM_SIZE:
        return None
    prob = model.prob(state.our_team[:TEAM_SIZE], state.their_team[:TEAM_SIZE], state.map_id, state.mode)
    if prob is None:
        return None
    verdict = next(text for floor, text in _MODEL_VERDICT if prob >= floor)
    return {
        "win_prob": round(float(prob), 4),
        "verdict": verdict,
        "note": "Draft only — the model sees six brawlers and the map, nothing about how either team plays.",
    }


def game_plan(state: DraftState, stats=None, model=None) -> dict:
    """Every read below is bounded to `TEAM_SIZE` per side.

    `/api/recommend` accepts `our_team`/`their_team` as bare int lists, and this runs on every
    call in both phases before anything else. The head-to-head grid is |theirs| x |ours|, so an
    oversized body used to turn straight into quadratic work and resident memory on a 512 MB
    instance — a regression this layer introduced, since the rule-based plan never looked at more
    than the class list. Truncating here keeps a malformed or hostile body costing the same as a
    real draft; a real board can never exceed three a side anyway.
    """
    names = _name_map()
    our_cls = [_class_of(b) for b in state.our_team[:TEAM_SIZE]]
    archetype, playstyle = _archetype(our_cls)
    plan = MODE_PLAN.get(state.mode, {"objective": "", "tips": [], "avoid": []})

    roles = [
        {"name": names.get(b, str(b)), "cls": _class_of(b),
         "role": ROLE_BY_CLASS.get(_class_of(b), ROLE_BY_CLASS["Unclassified"])}
        for b in state.our_team[:TEAM_SIZE]
    ]
    threats = []
    for e in state.their_team[:TEAM_SIZE]:
        cls = _class_of(e)
        tip = THREAT_BY_CLASS.get(cls)
        if tip:
            threats.append({"name": names.get(e, str(e)), "cls": cls,
                            "tip": tip.format(name=names.get(e, str(e)))})

    compensate = []
    if our_cls and not any(c in ("Tank", "Assassin") for c in our_cls):
        compensate.append("No frontline — you can't contest space head-on; poke and kite, don't get dived.")
    if our_cls and not any(c in ("Marksman", "Controller", "Artillery") for c in our_cls):
        compensate.append("No long range — close distance fast and avoid poke wars you'll lose.")
    if sum(1 for b in state.their_team[:TEAM_SIZE] if _class_of(b) == "Tank") >= 2 and "Marksman" not in our_cls:
        compensate.append("Enemy is tank-heavy — kite relentlessly, chip them down, never get cornered.")

    tone = _TONE.get(archetype, "Stay flexible")
    verb = _MODE_VERB.get(state.mode, "play to your comp's strengths")
    win_condition = f"{tone}: {verb}."

    # Their shape only reads as a shape once two of them are on the board; one pick is a brawler,
    # not a comp. Blind pick (no enemy revealed) skips this entirely.
    enemy = None
    if len(state.their_team[:TEAM_SIZE]) >= 2:
        their_arch, their_style = _archetype([_class_of(b) for b in state.their_team[:TEAM_SIZE]])
        enemy = {"archetype": their_arch, "playstyle": their_style,
                 "clash": _CLASH.get((archetype, their_arch), "")}

    return {
        "objective": plan["objective"],
        "win_condition": win_condition,
        "archetype": archetype,
        "playstyle": playstyle,
        "roles": roles,
        "threats": threats,
        "tips": plan["tips"],
        "avoid": plan["avoid"],
        "compensate": compensate,
        "enemy": enemy,
        "map_read": _map_read(state, stats, names) if stats is not None else [],
        "pairs": _pairs(state, stats, names) if stats is not None else [],
        "head_to_head": _head_to_head(state, stats, names) if stats is not None else None,
        "model_read": _model_read(state, model),
    }
