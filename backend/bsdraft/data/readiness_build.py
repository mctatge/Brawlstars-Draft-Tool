"""Measure what an under-leveled brawler actually costs, from the match log alone.

The served win rates are, empirically, Power 11 win rates: 97.3% of collected player-slots are
Power 11 (Ranked hard-blocks selecting below Power 9 through Diamond, Power 11 from Mythic up, so
the sub-11 tail is thin and structural). Scoring a Power 9 brawler off that table is unmeasured
extrapolation — :mod:`bsdraft.engine.mastery` currently assumes "the dataset's win rates already
fold real power in", which this module exists to test.

The naive contrast is a trap: players who bring a Power 9 brawler into a Diamond lobby are simply
worse players, and the cross-player comparison reads about -16 points, most of it skill. So the
player is used as their own control — their under-11 appearances are contrasted against *their own*
Power 11 appearances, restricted to lobbies where all five other players are Power 11 (so the
comparison isn't polluted by whoever else is under-leveled). Strata are (tag x Ranked bracket);
outcomes are residualized against a population table built only from all-Power-11 lobbies, so
"plays Power 9 on off-meta brawlers" doesn't masquerade as a power effect. Strata are combined
Mantel-Haenszel style, and a single player is capped so a grinder can't dominate a pool.

The estimate ships only if a **placebo** passes: relabel a random half of all-Power-11 appearances
as "treated" and re-run the identical pipeline. A design that manufactures an effect shows it here.
:mod:`backend.scripts.export_readiness` refuses to write the artifact when the placebo fails.

Home-only (needs the full match log) and numpy-backed — deliberately NOT on the serve import path.
The serve side reads the small stdlib-loadable ``data/reference/readiness.json`` this produces.
:func:`mh_combine` is pure so it can be unit-tested on synthetic strata with a known effect.

See docs/readiness.md for the three-estimator progression and the shipping-constant rule.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from bsdraft.data.dataset import iter_matches
from bsdraft.engine.tiers import match_bracket

# Estimator knobs (written into the artifact meta, so retuning needs no code change).
DEFAULTS = dict(
    pop_prior=50.0,       # pseudo-games at 0.5 smoothing the all-P11 population table
    cap_per_arm=30,       # max appearances one player contributes to one (stratum, arm)
    min_strata=200,       # refuse to publish a level's estimate below this many paired strata
    min_abs_z=3.0,        # ...or below this |z|
    placebo_tol=0.005,    # |placebo RD| above this fails the gate outright
    placebo_seed=7,       # fixed so a rebuild is reproducible
    se_haircut=2.0,       # SEs shaved off the point estimate before shipping (see _ship_value)
    methodology_margin=0.015,  # ...or this much, whichever is larger (see _ship_value)
    ship_grid=0.005,      # shipping constants are floored to this grid (half a point)
)

MAX_POWER = 11
_TREATED_LEVELS = (9, 10)   # the window that survives the Ranked floor below Mythic
_LE8 = "le8"                # everything below 9 — reported, never shipped as its own constant


# --------------------------------------------------------------------------------------
# pass 1 — the population table (all-Power-11 lobbies only)
# --------------------------------------------------------------------------------------

def _appearances(match: dict) -> Iterator[Tuple[dict, bool, List[dict]]]:
    """Yield (player, won, others) for all six slots of a decided 3v3."""
    a, b = match.get("team_a") or [], match.get("team_b") or []
    if len(a) != 3 or len(b) != 3:
        return
    aw = match.get("a_won")
    if aw is None:
        return
    both = a + b
    for side, won in ((a, bool(aw)), (b, not bool(aw))):
        for p in side:
            others = [q for q in both if q is not p]
            yield p, won, others


@dataclass(frozen=True)
class Population:
    """The all-Power-11 baseline an under-leveled appearance is measured against."""

    cell: Dict[Tuple[int, str], float]   # (brawler_id, bracket) -> smoothed win rate
    brawler: Dict[int, float]            # brawler_id -> all-bracket rate (the back-off shelf)
    n_matches: int

    def rate(self, bid: Optional[int], bracket: str) -> float:
        if bid is None:
            return 0.5
        v = self.cell.get((bid, bracket))
        return v if v is not None else self.brawler.get(bid, 0.5)


def build_population(matches: Optional[Path] = None, prior: float = DEFAULTS["pop_prior"],
                     progress_every: int = 250_000) -> Population:
    """Smoothed win rates from all-Power-11 lobbies only.

    Built at full power on purpose: this is the baseline an under-leveled appearance is measured
    *against*, so letting sub-11 games into it would drag the reference toward the effect being
    estimated. Backs off to the brawler's all-bracket rate, then to 0.5.
    """
    cell: Dict[Tuple[int, str], List[float]] = defaultdict(lambda: [0.0, 0.0])
    allb: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0])
    n = 0
    for m in iter_matches(matches):
        bracket = match_bracket(m)
        if bracket is None:
            continue
        rows = list(_appearances(m))
        if not rows:
            continue
        # all six at full power, else this match teaches us nothing about the baseline
        if any(p.get("power") != MAX_POWER for p, _, _ in rows):
            continue
        n += 1
        if progress_every and n % progress_every == 0:
            print(f"  population: {n:,} all-P11 matches", flush=True)
        for p, won, _ in rows:
            bid = p.get("brawler_id")
            if bid is None:
                continue
            cell[(bid, bracket)][0] += won
            cell[(bid, bracket)][1] += 1
            allb[bid][0] += won
            allb[bid][1] += 1

    brawler = {b: (w + prior * 0.5) / (g + prior) for b, (w, g) in allb.items()}
    return Population(
        cell={k: (w + prior * brawler.get(k[0], 0.5)) / (g + prior)
              for k, (w, g) in cell.items()},
        brawler=brawler,
        n_matches=n,
    )


# --------------------------------------------------------------------------------------
# pass 2 — accumulate paired strata
# --------------------------------------------------------------------------------------

class _Arms:
    """Sparse per-arm accumulators keyed by stratum. Each entry is [residual_sum, n]."""

    def __init__(self) -> None:
        self.control: Dict[tuple, List[float]] = {}
        self.treated: Dict[object, Dict[tuple, List[float]]] = {
            9: {}, 10: {}, _LE8: {},
        }
        self.placebo_a: Dict[tuple, List[float]] = {}
        self.placebo_b: Dict[tuple, List[float]] = {}

    @staticmethod
    def _add(pool: Dict[tuple, List[float]], key: tuple, y: float, cap: int) -> None:
        e = pool.get(key)
        if e is None:
            pool[key] = [y, 1.0]
        elif e[1] < cap:
            e[0] += y
            e[1] += 1.0


def accumulate(matches: Optional[Path] = None, pop: Optional[Population] = None,
               cap: int = DEFAULTS["cap_per_arm"], seed: int = DEFAULTS["placebo_seed"],
               progress_every: int = 250_000) -> Tuple[_Arms, dict]:
    """Second pass: bucket every *clean* appearance into its player's control or treated arm.

    Clean = all five other players in the lobby are Power 11, so the only power anomaly on the
    board is this slot. Outcomes are residualized against the population table, which strips the
    brawler-choice and bracket confounds before any contrast is taken.
    """
    pop = pop if pop is not None else Population(cell={}, brawler={}, n_matches=0)
    arms = _Arms()
    rng = random.Random(seed)
    seen = {"matches": 0, "clean_appearances": 0, "newest_ts": 0, "power_hist": defaultdict(int)}

    for m in iter_matches(matches):
        bracket = match_bracket(m)
        if bracket is None:
            continue
        rows = list(_appearances(m))
        if not rows:
            continue
        seen["matches"] += 1
        ts = m.get("ts")
        if isinstance(ts, int) and ts > seen["newest_ts"]:
            seen["newest_ts"] = ts
        if progress_every and seen["matches"] % progress_every == 0:
            print(f"  strata: {seen['matches']:,} matches", flush=True)

        for p, won, others in rows:
            pw = p.get("power")
            seen["power_hist"][pw] += 1
            if not isinstance(pw, int) or pw <= 0:
                continue                                   # unknown power is not evidence
            if any(q.get("power") != MAX_POWER for q in others):
                continue                                   # not a clean lobby
            tag = p.get("tag")
            if not tag:
                continue
            seen["clean_appearances"] += 1
            y = float(won) - pop.rate(p.get("brawler_id"), bracket)
            key = (tag, bracket)
            if pw >= MAX_POWER:
                arms._add(arms.control, key, y, cap)
                # placebo: split the control arm in half and contrast it against itself
                pool = arms.placebo_a if rng.random() < 0.5 else arms.placebo_b
                arms._add(pool, key, y, cap)
            elif pw in _TREATED_LEVELS:
                arms._add(arms.treated[pw], key, y, cap)
            else:
                arms._add(arms.treated[_LE8], key, y, cap)

    seen["power_hist"] = dict(seen["power_hist"])
    return arms, seen


# --------------------------------------------------------------------------------------
# the estimator — pure, unit-testable
# --------------------------------------------------------------------------------------

def mh_combine(treated: Dict[tuple, List[float]],
               control: Dict[tuple, List[float]]) -> dict:
    """Mantel-Haenszel-style combine of per-stratum risk differences.

    Only strata present in BOTH arms contribute — that pairing is what makes the player their own
    control. Stratum weight is the harmonic term ``n_t*n_c/(n_t+n_c)``, which is the usual
    precision weight for a difference of means.

    The SE is a *robust* weighted-mean standard error taken across strata rather than a binomial
    formula: the outcomes here are residuals in [-1, 1], not 0/1, and the residualization plus the
    per-player cap both break the binomial assumption. The across-stratum spread absorbs all of it.
    """
    rds: List[float] = []
    ws: List[float] = []
    n_t = n_c = 0.0
    for key, t in treated.items():
        c = control.get(key)
        if c is None or t[1] <= 0 or c[1] <= 0:
            continue
        rds.append(t[0] / t[1] - c[0] / c[1])
        ws.append((t[1] * c[1]) / (t[1] + c[1]))
        n_t += t[1]
        n_c += c[1]
    if not rds:
        return dict(rd=float("nan"), se=float("nan"), z=float("nan"), n_strata=0,
                    n_treated=0.0, n_control=0.0)

    r = np.asarray(rds, dtype=float)
    w = np.asarray(ws, dtype=float)
    wsum = float(w.sum())
    rd = float((w * r).sum() / wsum)
    # robust (sandwich) SE of a weighted mean: sqrt(Σ w²(r-rd)²) / Σ w
    se = float(np.sqrt((w ** 2 * (r - rd) ** 2).sum()) / wsum)
    z = rd / se if se > 0 else float("nan")
    return dict(rd=rd, se=se, z=z, n_strata=len(rds), n_treated=n_t, n_control=n_c)


def _ship_value(rd: float, se: float, haircut: float, grid: float, margin: float) -> float:
    """The constant actually shipped: the *magnitude* of the deficit, shaved by a safety margin
    and floored to ``grid``.

    The margin is ``max(haircut * se, margin)`` — whichever uncertainty dominates governs, and on
    this corpus that is emphatically not the sampling error. Independent specifications of this
    estimator have returned Power 9 at 7.5, 8.1, 9.4 and 9.9 points, all with clean placebos: a
    spread of ~2.4 points driven by stratum, cap and residualization choices. A single run's SE is
    ~0.45 points, so keying the haircut to the SE alone would absorb a fifth of the uncertainty it
    claims to cover. ``methodology_margin`` names the real quantity in the units it lives in.

    Direction of error is deliberate. Under-claiming a real effect is recoverable — the number
    rises the next time this runs — while over-claiming re-creates the over-personalization the
    2026-08-17 de-weighting was fighting. Never returns a negative (a "deficit" that helps).
    """
    if not np.isfinite(rd) or not np.isfinite(se):
        return 0.0
    conservative = abs(rd) - max(haircut * se, margin)
    if conservative <= 0:
        return 0.0
    return float(np.floor(conservative / grid) * grid)


def estimate_readiness(matches: Optional[Path] = None, params: Optional[dict] = None) -> dict:
    """Full two-pass build. Returns the artifact payload (see docs/readiness.md for the schema)."""
    p = dict(DEFAULTS)
    p.update(params or {})
    t0 = time.time()

    print("pass 1/2 — population table (all-P11 lobbies)", flush=True)
    pop = build_population(matches, prior=p["pop_prior"])
    print(f"  {pop.n_matches:,} all-P11 matches -> {len(pop.cell):,} (brawler, bracket) cells",
          flush=True)

    print("pass 2/2 — paired strata", flush=True)
    arms, seen = accumulate(matches, pop, cap=p["cap_per_arm"], seed=p["placebo_seed"])
    print(f"  {seen['matches']:,} matches, {seen['clean_appearances']:,} clean appearances",
          flush=True)

    placebo = mh_combine(arms.placebo_a, arms.placebo_b)
    levels: Dict[str, dict] = {}
    for lvl in (*_TREATED_LEVELS, _LE8):
        r = mh_combine(arms.treated[lvl], arms.control)
        r["ship"] = _ship_value(r["rd"], r["se"], p["se_haircut"],
                                p["ship_grid"], p["methodology_margin"])
        r["shippable"] = bool(
            r["n_strata"] >= p["min_strata"]
            and np.isfinite(r["z"]) and abs(r["z"]) >= p["min_abs_z"]
            and r["rd"] < 0                     # a deficit must actually be a deficit
        )
        levels[str(lvl)] = r

    placebo_ok = bool(np.isfinite(placebo["rd"]) and abs(placebo["rd"]) <= p["placebo_tol"])
    monotone = _monotone(levels)

    return {
        "schema": 1,
        "meta": {
            "built_ts": int(time.time()),
            "build_seconds": round(time.time() - t0, 1),
            "newest_match_ts": seen["newest_ts"],
            "matches": seen["matches"],
            "clean_appearances": seen["clean_appearances"],
            "power_hist": {str(k): v for k, v in sorted(
                seen["power_hist"].items(), key=lambda kv: (kv[0] is None, kv[0]), reverse=True)},
            "params": p,
        },
        "placebo": {**placebo, "tol": p["placebo_tol"], "ok": placebo_ok},
        "levels": levels,
        "monotone": monotone,
        # The serve-side constants, in win-rate points, magnitude (a deficit is subtracted).
        # Power <= 8 rides Power 9's constant: it is 0.005% of slots and below the Ranked floor
        # in every bracket, so it is unfieldable rather than merely weak.
        "power_deficit": _power_deficit(levels),
        "ok": bool(placebo_ok and monotone and levels["9"]["shippable"]),
    }


def _monotone(levels: Dict[str, dict]) -> bool:
    """A real power effect must not get *smaller* as power drops."""
    try:
        return levels["9"]["rd"] <= levels["10"]["rd"]
    except (KeyError, TypeError):
        return False


def _power_deficit(levels: Dict[str, dict]) -> Dict[str, float]:
    out = {"11": 0.0}
    for lvl in _TREATED_LEVELS:
        r = levels.get(str(lvl)) or {}
        out[str(lvl)] = float(r.get("ship", 0.0)) if r.get("shippable") else 0.0
    for lvl in range(1, 9):                     # below the floor — inherit Power 9's constant
        out[str(lvl)] = out["9"]
    return out
