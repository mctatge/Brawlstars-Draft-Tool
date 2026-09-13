"""Personalized "what to buy/upgrade next" advisor.

Given a player's ownership snapshot (power level + owned gadgets / star powers / gears /
hypercharge per brawler, from ``/players/{tag}``) and their Ranked bracket, this enumerates the
purchases they *haven't* made yet and ranks them by **value per resource**: the expected win-rate
lift a purchase — together with every prerequisite it drags in — buys per coin-equivalent spent.
It is the inverse of :mod:`bsdraft.engine.loadout`: the loadout advisor tells you which owned item
to *equip*; this tells you which unowned item is most worth *acquiring*.

Design (product decisions; revised 2026-08-19 — the original "rank by raw value, cost is context
only" rule surfaced recs like "buy Damien's star power" for a Damien parked far below Power 11
with no items, which the owner rightly called nonsense):

* **Rank by value per cost.** ``value_score = value_lift / cost_equiv × 1000`` — win-rate lift
  per 1,000 coin-equivalents, where ``cost_equiv`` folds coins, power points and credits together
  with tunable exchange weights. The API exposes ownership, never balances, so *affordability*
  is still unknowable — but *efficiency* isn't, and it is what a careful player optimizes.
* **Prerequisites are part of the purchase.** A rec is the *package* needed to realize its value
  from the player's current state: the power climb to the bracket's floor, a first gadget + star
  power for an unbuilt brawler, the Starr Road unlock for an unowned one. Its cost is the package
  cost, never the sticker price alone — and its value is **additive over the package**: the climb
  carries the value of *access* plus every item it realizes (owned or bought), so reaching the
  same end state is worth the same whichever path you take.
* **The Ranked power floor is a hard gate.** Ranked does NOT normalize power: brawlers play at
  their real level and each bracket blocks selecting one below a floor (Power 9 through Diamond,
  Power 11 from Mythic up — ``tiers.min_power_for_bracket``). A brawler below the floor is
  unfieldable, so nothing bought on it has realizable value until it is climbed: it gets exactly
  ONE rec — the "ranked-ready" package — and no item recs. An unknown bracket assumes the
  stricter Power-11 floor (the safer failure for an advisor: it hides a few P9–10 item recs from a
  Diamond player instead of recommending unrealizable buys to a Mythic one); the client can pin
  the floor explicitly.
* **Meta-weighted; roster-aware for draft likelihood.** Every rec scales with the brawler's meta
  strength (exponentially in win rate, so investment favors meta brawlers) and with a
  *depth factor*: how many brawlers the player can already field that are stronger — overall, or
  in the brawler's best ranked mode — with diminishing returns past a free top tier. Mains and
  mastery never enter: ownership decides what's buyable, the meta decides what's worth it.
* **Account-wide across the ranked map pool** — strength is the games-weighted average of the
  brawler's win rate over the *current* ranked maps (the bracket's table when it exists), falling
  back to the global rate on thin data.

Everything tunable lives in ``economy.json`` (priors in relative win-rate-lift units — only their
ratios matter — costs, and the ``scoring`` knobs) with complete code fallbacks, so a stale or
partial file still ranks coherently and never errors. Pure stdlib so it stays importable on the
serve path (no torch/sklearn/pandas), like :mod:`bsdraft.engine.loadout`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional

from bsdraft.data import reference as R
from bsdraft.engine import itemstats as IS
from bsdraft.engine.stats import DraftStats
from bsdraft.engine.tiers import BRACKETS, min_power_for_bracket

MAX_POWER = 11
FLOORS = (9, 11)
# Power levels at which a gear slot opens (first at 8, second at 10).
_GEAR_SLOT_POWERS = (8, 10)

# ---- code fallbacks for every tunable (mirrors economy.json; see its comments for the why) ----
_DEFAULT_PRIORS = {
    "gadget_first": 3.0, "gadget_second": 0.6,
    "star_power_first": 3.5, "star_power_second": 0.6,
    "gear_core": 0.7, "gear_extra": 0.25,
    "hypercharge": 6.0,
    "power_level": 1.5,       # per power level of real stat gain above the floor, up to 11
    "access": 3.0,            # having a bare brawler available to field at all
    "readiness": 0.5,         # share of the ranked-ready value a ≤Diamond climb to 11 earns now
}
_DEFAULT_SCORING = {
    "meta_center": 0.50, "meta_scale": 0.125,
    "resource_weights": {"coins": 1.0, "power_points": 0.5, "credits": 1.0},
    "depth_free": 8.0, "depth_k": 12.0,
    "mode_shrink_games": 300.0,   # pseudo-games pulling a mode rate toward the brawler's overall rate
    "delta_scale": 0.10,      # a ±10pp significant measured delta doubles / halves a second item
    "boosted_discount": 0.5,
    "unknown_floor": 11,
}
_DEFAULT_COSTS = {
    "power_cost_cumulative": {
        "power_points": [0, 0, 20, 50, 100, 180, 310, 520, 860, 1410, 2300, 3740],
        "coins":        [0, 0, 20, 55, 130, 270, 560, 1040, 1840, 3090, 4965, 7765],
    },
    "item_costs": {
        "gadget": {"coins": 1000}, "star_power": {"coins": 2000}, "gear": {"coins": 1000},
        "hypercharge": {"coins": 5000},
    },
    # Starr Road credit prices. Rare / Super Rare / Common aren't on the Starr Road (Trophy Road
    # rewards since 2025-06) so they're deliberately absent: an unowned one isn't *buyable*.
    "new_brawler_credits": {"Epic": 925, "Mythic": 1900, "Legendary": 3800, "Ultra Legendary": 5500},
    "power_gates": {"gadget": 7, "gear": 8, "star_power": 9, "hypercharge": 11},
}
# The Starr Road is walked rarity by rarity (choices appear within the current tier), so only the
# lowest tier with an unowned brawler is purchasable right now.
_STARR_ROAD_ORDER = ("Epic", "Mythic", "Legendary", "Ultra Legendary")

# Below this effective sample across the ranked maps, fall back to the global rate.
_MIN_MAP_GAMES = 8.0
# A mode-level strength only counts as a "best mode" view for depth with this much real data (and
# it is shrunk toward the brawler's overall rate by ``scoring.mode_shrink_games`` pseudo-games, so
# a 30-game blip can't make a 46% brawler "depth-free" while a 400-game specialist keeps its edge).
_MIN_MODE_GAMES = 100.0
# A new-brawler rec needs real games behind its win rate — below this it's skipped outright (the
# data never saw it: unreleased, or an unobtainable retired collab) rather than vouched for.
_MIN_MEASURED_GAMES = 40.0

_norm_re = re.compile(r"[^a-z0-9]")


def _norm(name: str) -> str:
    return _norm_re.sub("", (name or "").lower())


@dataclass(frozen=True)
class OwnedState:
    """One brawler's ownership, distilled from a roster entry. Ids match the catalog
    (``R.load_brawlers``); gears are keyed by normalized name (no catalog id exists).
    ``power`` 0 means "unknown" (an older client that omits it) and is treated as maxed."""
    power: int = 0
    star_powers: FrozenSet[int] = frozenset()
    gadgets: FrozenSet[int] = frozenset()
    gears: FrozenSet[str] = frozenset()          # normalized owned gear names
    has_hypercharge: bool = False


@dataclass
class _Rec:
    brawler_id: int
    brawler_name: str
    kind: str                    # power_upgrade|gadget|star_power|gear|hypercharge|new_brawler
    value_lift: float            # relative win-rate lift the whole package realizes
    cost: Dict[str, int]         # package cost, summed across steps
    cost_equiv: Optional[float]  # coin-equivalents (None ⇒ nothing in the package is priced)
    value_score: float           # the sort key: lift per 1k coin-equivalents
    meta_winrate: float
    confidence: str              # "measured" | "heuristic" | "eligibility_only"
    rationale: str
    steps: List[dict] = field(default_factory=list)   # [{kind, label, cost}] — the package
    cost_estimated: bool = False                       # a step had no price and got a nominal one
    item_id: Optional[int] = None
    item_name: Optional[str] = None
    target_power: Optional[int] = None
    item_delta: Optional[float] = None
    gate: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "brawler_id": self.brawler_id, "brawler_name": self.brawler_name, "kind": self.kind,
            "value_score": round(self.value_score, 4), "value_lift": round(self.value_lift, 4),
            "cost": self.cost,
            "cost_equiv": None if self.cost_equiv is None else round(self.cost_equiv, 1),
            "cost_estimated": self.cost_estimated,
            "meta_winrate": round(self.meta_winrate, 4),
            "confidence": self.confidence, "rationale": self.rationale, "steps": self.steps,
            "item_id": self.item_id, "item_name": self.item_name, "target_power": self.target_power,
            "item_delta": None if self.item_delta is None else round(self.item_delta, 4),
            "gate": self.gate,
        }


# --- economy helpers (fail-safe: code defaults fill every section; the file overlays per key) ----

def _priors(economy: dict) -> dict:
    """Lift-unit priors. The file's section is used only when it is v2-shaped (carries the keys
    only v2 defines) — an old 0..1-unit section would otherwise get overlaid key-by-key onto
    lift-unit defaults and produce a unit-mixed ranking."""
    out = dict(_DEFAULT_PRIORS)
    sec = (economy or {}).get("impact_priors") or {}
    if isinstance(sec, dict) and {"access", "power_level", "gear_core"} <= set(sec):
        for k, v in sec.items():
            if k in out and isinstance(v, (int, float)):
                out[k] = float(v)
    return out


def _scoring(economy: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULT_SCORING.items()}
    for k, v in ((economy or {}).get("scoring") or {}).items():
        if k == "resource_weights" and isinstance(v, dict):
            out[k].update({rk: float(rv) for rk, rv in v.items() if isinstance(rv, (int, float))})
        elif k in out and isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _costs(economy: dict) -> dict:
    """Cost tables: defaults overlaid per section by the file, so a missing section can't silently
    make a package free (cost is the ranking denominator now)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULT_COSTS.items()}
    econ = economy or {}
    for sec in ("item_costs", "new_brawler_credits", "power_gates"):
        v = econ.get(sec)
        if isinstance(v, dict):
            out[sec].update({k: x for k, x in v.items() if not str(k).startswith("_")})
    pc = econ.get("power_cost_cumulative")
    if isinstance(pc, dict):
        for k in ("coins", "power_points"):
            arr = pc.get(k)
            if isinstance(arr, list) and len(arr) > MAX_POWER:
                out["power_cost_cumulative"][k] = arr
    return out


def _merge_cost(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + int(v)
    return out


def _hypercharge_eligible(economy: dict, brawler_name: str) -> bool:
    """Does this brawler have a hypercharge to buy? The API can't tell (see economy.json), so this
    reads the curated availability policy. Returns False when the section is absent (fail-safe:
    don't recommend a hypercharge we can't vouch for)."""
    cfg = economy.get("hypercharge_availability") if economy else None
    if not isinstance(cfg, dict):
        return False
    names = {_norm(n) for n in (cfg.get("list") or [])}
    mode = cfg.get("mode", "all_except")
    if mode == "only":
        return _norm(brawler_name) in names
    return _norm(brawler_name) not in names   # "all_except"


def resolve_floor(rank_bracket: Optional[str], power_floor: Optional[int],
                  economy: Optional[dict] = None) -> int:
    """The Ranked power floor to gate on: an explicit ``power_floor`` (9/11) wins; else the
    bracket's floor; an unknown bracket assumes the stricter floor (``scoring.unknown_floor``)."""
    if power_floor in FLOORS:
        return int(power_floor)
    if rank_bracket in BRACKETS:
        return min_power_for_bracket(rank_bracket)
    return int(_scoring(economy or {})["unknown_floor"])


# --- scoring context ---------------------------------------------------------------------------

@dataclass
class _Ctx:
    """Everything a rec builder needs beyond the brawler: the knobs, the floor, the strength
    tables (overall + per ranked mode) and the player's fieldable pool for the depth factor."""
    economy: dict
    priors: dict
    scoring: dict
    costs: dict
    floor: int
    strength: Dict[int, float]                      # brawler id → overall ranked-pool strength
    mode_strength: Dict[str, Dict[int, float]]      # mode → brawler id → strength (thin ⇒ absent)
    pool: Dict[int, float]                          # fieldable brawler id → build weight (0.5..1)
    itemstats: Optional[dict]
    boosted: FrozenSet[int]

    # -- value --------------------------------------------------------------------------------
    def meta_factor(self, strength: float) -> float:
        """exp((s − 0.5) / scale): a 50% brawler → 1.0, 55% → 1.49, 45% → 0.67 at scale .125.
        Equal win-rate gaps ⇒ equal ratios, no zero-crossing. Live ranked-pool strengths run
        ~.44–.59 (p10 .466, median .498, p90 .535) ⇒ factors ~.64–2.0; kind structure (priors
        0.25..5, costs 1k..5k+) still dominates, meta re-orders within a kind."""
        s = self.scoring
        scale = float(s["meta_scale"]) or 0.125
        return math.exp((strength - float(s["meta_center"])) / scale)

    def depth(self, brawler_id: int):
        """``(n, mode, rate)``: how many brawlers the player already fields that are stronger than
        this one — the build-weighted count, taken over the overall ranked pool AND each ranked
        mode where this brawler has real data (rates shrunk toward its overall rate), keeping the
        smallest: a mode specialist is an option in its mode even if it's the 20th-best brawler
        overall. ``mode``/``rate`` name the ranked mode that view came from and the brawler's
        rate there, or None when the overall pool is already the kindest view."""
        views = [(None, self.strength)] + [(mode, m) for mode, m in self.mode_strength.items()
                                            if brawler_id in m]
        best, best_mode, best_rate = None, None, None
        for mode, view in views:
            s = view.get(brawler_id)
            if s is None:
                s = self.strength.get(brawler_id, 0.5)
            n = 0.0
            for x, w in self.pool.items():
                if x == brawler_id:
                    continue
                sx = view.get(x)
                if sx is None:
                    sx = self.strength.get(x, 0.5)
                if sx > s:
                    n += w
            if best is None or n < best:
                best, best_mode, best_rate = n, mode, (s if mode else None)
        return (best or 0.0), best_mode, best_rate

    def depth_n(self, brawler_id: int) -> float:
        return self.depth(brawler_id)[0]

    def depth_note(self, brawler_id: int) -> str:
        """Human form of the depth read. A build-weighted count below one (a lone bare P11) reads
        as "nothing stronger"; when a specialist mode view is what counts, name the mode and the
        brawler's rate there so the claim doesn't look contradicted by the overall win rate."""
        n, mode, rate = self.depth(brawler_id)
        where = f" in {mode} ({_pct(rate)} there)" if mode and rate is not None else ""
        if n < 1.0:
            return f"nothing you field is stronger{where}"
        k = int(round(n))
        return f"you already field {k} stronger brawler{'s' if k != 1 else ''}{where}"

    def depth_factor(self, brawler_id: int) -> float:
        """Diminishing returns past the player's free top tier: 1 while fewer than ``depth_free``
        stronger brawlers are fieldable, then 1/(1 + (n − free)/k)."""
        n = self.depth_n(brawler_id)
        free = float(self.scoring["depth_free"])
        k = float(self.scoring["depth_k"]) or 12.0
        return 1.0 / (1.0 + max(0.0, n - free) / k)

    def weight(self, brawler_id: int, strength: float) -> float:
        """Common multiplier on every rec for this brawler: meta × depth × boosted discount."""
        w = self.meta_factor(strength) * self.depth_factor(brawler_id)
        if brawler_id in self.boosted:
            w *= float(self.scoring["boosted_discount"])
        return w

    def delta_multiplier(self, delta: Optional[float]) -> float:
        """Signed, capped multiplier for a significant measured delta on a *second* item (or a
        gear): a positive delta says you'd actually switch to it (close to a real own-vs-not-own
        gain); a negative one says the item you own already measures better. Never beyond
        halving/doubling."""
        if delta is None:
            return 1.0
        scale = float(self.scoring["delta_scale"]) or 0.10
        return max(0.5, min(2.0, 1.0 + delta / scale))

    # -- cost ---------------------------------------------------------------------------------
    def item_cost(self, kind: str) -> Dict[str, int]:
        return dict(self.costs["item_costs"].get(kind) or {})

    def power_cost(self, cur: int, target: int) -> Dict[str, int]:
        """Coin + power-point cost to go from ``cur`` to ``target`` power (cumulative table)."""
        table = self.costs["power_cost_cumulative"]
        cur = max(1, min(MAX_POWER, int(cur or 1)))
        target = max(1, min(MAX_POWER, int(target)))
        out: Dict[str, int] = {}
        if target <= cur:
            return out
        for key in ("coins", "power_points"):
            arr = table.get(key)
            if isinstance(arr, list) and len(arr) > target:
                delta = int(arr[target]) - int(arr[cur])
                if delta > 0:
                    out[key] = delta
        return out

    def credits(self, rarity: str):
        """(credit cost or None, estimated?) — an unknown rarity gets the dearest known price as a
        nominal so it never ranks as a free unlock."""
        table = self.costs["new_brawler_credits"]
        c = table.get(rarity)
        if isinstance(c, (int, float)) and c > 0:
            return int(c), False
        known = [int(v) for v in table.values() if isinstance(v, (int, float)) and v > 0]
        return (max(known) if known else None), True

    def cost_equiv(self, cost: Dict[str, int]) -> Optional[float]:
        w = self.scoring["resource_weights"]
        known = [(k, v) for k, v in cost.items() if v and k in w]
        if not known:
            return None
        return float(sum(float(w[k]) * float(v) for k, v in known))

    def fieldable(self, st: OwnedState) -> bool:
        return st.power == 0 or st.power >= self.floor

    def effective_power(self, st: OwnedState) -> int:
        return MAX_POWER if st.power == 0 else int(st.power)

    def gear_slots(self, st: OwnedState) -> int:
        p = self.effective_power(st)
        return sum(1 for g in _GEAR_SLOT_POWERS if p >= g)

    def owned_item_lift(self, b, st: OwnedState) -> float:
        """Prior-units of everything already on the brawler — what a climb to the floor realizes."""
        pr = self.priors
        lift = 0.0
        if st.gadgets:
            lift += pr["gadget_first"] + pr["gadget_second"] * (len(st.gadgets) - 1)
        if st.star_powers:
            lift += pr["star_power_first"] + pr["star_power_second"] * (len(st.star_powers) - 1)
        if st.has_hypercharge:
            lift += pr["hypercharge"]
        if st.gears:
            core = min(len(st.gears), len(_GEAR_SLOT_POWERS))
            lift += pr["gear_core"] * core + pr["gear_extra"] * (len(st.gears) - core)
        return lift

    def make(self, **kw) -> _Rec:
        """Finish a rec: derive cost_equiv and the value-per-cost sort key. A package with nothing
        priced (impossible with the default tables, but the contract is never-error) scores
        against a nominal 1,000 so the list still ranks by value instead of dividing by zero."""
        ce = self.cost_equiv(kw["cost"])
        denom = max(ce if ce else 1000.0, 1.0)
        kw["cost_equiv"] = ce
        kw["value_score"] = kw["value_lift"] / denom * 1000.0
        if not math.isfinite(kw["value_score"]):
            kw["value_score"] = 0.0
        return _Rec(**kw)


# --- meta / measurement helpers ----------------------------------------------------------------

def _pool_strength(stats: DraftStats, brawler_id: int, map_ids: List[int], min_games: float):
    """``(games-weighted win rate over map_ids, effective games)`` — None when below min_games."""
    num = den = 0.0
    for mid in map_ids:
        r = stats.brawler_rate(brawler_id, mid)
        num += r.winrate * r.games
        den += r.games
    if den >= min_games:
        return num / den, den
    return None


def _meta_strength(stats: DraftStats, brawler_id: int, map_ids: List[int]) -> float:
    """Games-weighted average win rate across the current ranked maps, falling back to the global
    rate when the per-map sample is thin. A smoothed rate (~0.5 baseline) — higher ⇒ stronger."""
    s = _pool_strength(stats, brawler_id, map_ids, _MIN_MAP_GAMES)
    return s[0] if s is not None else stats.brawler_rate(brawler_id, None).winrate


def _global_table(stats: DraftStats) -> DraftStats:
    """The global table behind a bracket table (bracket tables shrink toward it). Existence checks
    — "has the data ever seen this brawler?" — must run here, not on a thin bracket's counts."""
    fb = getattr(stats, "fallback", None)
    return fb if fb is not None else stats


def _measured_delta(itemstats: Optional[dict], brawler_id: int, item_id: Optional[int],
                    gear_name: Optional[str] = None) -> Optional[float]:
    """The significant measured win-rate delta for an item, or None. Positive ⇒ measured better
    than the brawler's other single-owned items of that type."""
    if not itemstats:
        return None
    if gear_name is not None:
        cell = IS.gear_cell(itemstats, brawler_id, gear_name)
    else:
        cell = IS.accessory_cell(itemstats, brawler_id, item_id)
    if cell and cell.get("significant"):
        return float(cell.get("delta", 0.0))
    return None


def _best_missing_accessory(missing, itemstats, brawler_id):
    """Pick which of the missing gadgets/star powers to recommend first: the one with the best
    significant measured delta — a measured *negative* delta ranks below "unmeasured" (the data
    says that one is the worse buy) — else the catalog-first (stable)."""
    def key(acc):
        d = _measured_delta(itemstats, brawler_id, acc.id)
        return (d if d is not None and d > 0 else (-1.0 if d is None else -2.0 + d),)
    best = max(missing, key=key)   # ties keep the first (catalog order) — max is stable that way
    return best, _measured_delta(itemstats, brawler_id, best.id)


def _gear_names(economy: dict) -> list:
    """The curated universal-gear list (reuses loadout's hand-maintained gears.json guide), each as
    ``(display_name, normalized_name, base_prior, roles)``."""
    from bsdraft.engine.loadout import _gear_guide  # reuse the fail-safe loader; both stdlib-only
    out = []
    for g in _gear_guide().get("gears", []):
        name = g.get("name", "")
        out.append((name, _norm(name), float(g.get("base", 0.4)), g.get("roles", {}) or {}))
    return out


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _rel_stats(cur: int, target: int) -> int:
    """Relative HP/damage gain from ``cur`` to ``target`` power, in %, from the +10%-of-base per
    level rule (P1 = 100% … P11 = 200%): 9→11 ≈ +11%, 10→11 ≈ +5%."""
    return int(round(((1 + 0.1 * (target - 1)) / (1 + 0.1 * (cur - 1)) - 1) * 100))


# --- entry point -------------------------------------------------------------------------------

def recommend_purchases(owned: Dict[int, OwnedState], stats: DraftStats,
                        itemstats: Optional[dict] = None, top: int = 20,
                        ranked_maps=None, economy: Optional[dict] = None,
                        rank_bracket: Optional[str] = None, power_floor: Optional[int] = None,
                        boosted: Optional[Iterable[int]] = None,
                        min_per_kind: int = 0) -> List[dict]:
    """Rank a player's most *efficient* next purchases across the account. ``owned`` maps brawler
    id → :class:`OwnedState` (a brawler absent from the map is unowned ⇒ a new-brawler candidate).
    The Ranked power floor comes from ``power_floor`` (explicit 9/11), else ``rank_bracket``, else
    the stricter default — see :func:`resolve_floor`. ``stats`` should be the bracket's table when
    one exists (the caller decides; the engine passes ``bracket_stats.get(bracket, stats)``).

    ``min_per_kind`` reserves up to that many slots per purchase kind (best-first) so a kind that
    is legitimately low-efficiency — new-brawler unlocks, say, which are priced in a separate
    currency and drag in a full climb — stays discoverable below the overall top; the result is
    still one list sorted by ``value_score``."""
    economy = economy if economy is not None else R.load_economy()
    ranked_maps = list(ranked_maps if ranked_maps is not None else R.load_ranked_maps())
    boosted_ids = frozenset(boosted if boosted is not None else R.load_ranked_boosted())
    floor = resolve_floor(rank_bracket, power_floor, economy)
    brawlers = list(R.load_brawlers())
    map_ids = [m.id for m in ranked_maps]
    by_mode: Dict[str, List[int]] = {}
    for m in ranked_maps:
        by_mode.setdefault(getattr(m, "mode", "") or "", []).append(m.id)

    strength = {b.id: _meta_strength(stats, b.id, map_ids) for b in brawlers}
    scoring = _scoring(economy)
    shrink = float(scoring["mode_shrink_games"])
    mode_strength: Dict[str, Dict[int, float]] = {}
    for mode, ids in by_mode.items():
        tbl = {}
        for b in brawlers:
            ps = _pool_strength(stats, b.id, ids, _MIN_MODE_GAMES)
            if ps is not None:
                rate, games = ps
                tbl[b.id] = (rate * games + strength[b.id] * shrink) / (games + shrink)
        if tbl:
            mode_strength[mode] = tbl
    # Brawlers the data has actually seen — judged on the GLOBAL table (a thin bracket's counts
    # say nothing about existence). An unreleased or retired-collab catalog entry is never vouched
    # for as an unlock, and never blocks the Starr Road tier walk either.
    base = _global_table(stats)
    vouched = {b.id for b in brawlers
               if base.brawler_rate(b.id, None).games >= _MIN_MEASURED_GAMES}

    # The pool the depth factor counts against: fieldable owned brawlers, weighted by how built
    # they are (a bare P11 is half an option), plus this season's boosted brawlers at full build.
    pool: Dict[int, float] = {}
    for bid, st in owned.items():
        if bid in strength and (st.power == 0 or st.power >= floor):
            pool[bid] = 0.5 + (0.25 if st.gadgets else 0.0) + (0.25 if st.star_powers else 0.0)
    for bid in boosted_ids:
        if bid in strength:
            pool[bid] = 1.0

    ctx = _Ctx(economy=economy or {}, priors=_priors(economy), scoring=scoring,
               costs=_costs(economy), floor=floor, strength=strength, mode_strength=mode_strength,
               pool=pool, itemstats=itemstats, boosted=boosted_ids)
    gears_guide = _gear_names(economy)
    starr_tier = _starr_road_tier(brawlers, owned, vouched)
    recs: List[_Rec] = []

    for b in brawlers:
        s = strength[b.id]
        st = owned.get(b.id)
        if st is None:
            if b.rarity == starr_tier and b.id in vouched:
                recs.append(_access_rec(ctx, b, s, st=None))
            continue
        if not ctx.fieldable(st):
            # Below the bracket's floor nothing on this brawler has realizable value — the only
            # sensible next purchase is the climb (plus a core build if it has none).
            recs.append(_access_rec(ctx, b, s, st=st))
            continue

        # --- gadgets / star powers: recommend the best still-missing one of each type ---
        for accessories, id_field, first_key, second_key, label in (
            (b.gadgets, "gadgets", "gadget_first", "gadget_second", "gadget"),
            (b.star_powers, "star_powers", "star_power_first", "star_power_second", "star power"),
        ):
            owned_ids = getattr(st, id_field)
            missing = [a for a in accessories if a.id not in owned_ids]
            if not missing:
                continue
            acc, delta = _best_missing_accessory(missing, itemstats, b.id)
            is_first = len(owned_ids) == 0
            recs.append(_accessory_rec(ctx, b, s, acc, label,
                                       ctx.priors[first_key if is_first else second_key],
                                       delta, is_first))

        rec = _gear_rec(ctx, b, st, s, gears_guide)
        if rec is not None:
            recs.append(rec)

        # One power-related rec per brawler: the climb to 11 while short of it (the Hypercharge
        # surfaces as a straight buy once there — no double-listing the same climb).
        if ctx.effective_power(st) < MAX_POWER:
            recs.append(_power_rec(ctx, b, st, s))
        elif not st.has_hypercharge and _hypercharge_eligible(economy, b.name):
            recs.append(_hypercharge_rec(ctx, b, s))

        # Buffies are intentionally not advised here. The curated availability policy now lets
        # mastery/readiness distinguish "owns none" from "none released", but a purchase row also
        # needs a reviewed cost/package model and an impact prior in this advisor's relative-lift
        # units. Do not reuse the scorer's win-rate-point readiness prior as though the units match.

    recs.sort(key=_sort_key)
    return [r.as_dict() for r in _select(recs, top, min_per_kind)]


def _sort_key(r: _Rec):
    return (-r.value_score, -r.value_lift, r.brawler_name, r.kind)


def _starr_road_tier(brawlers, owned, vouched) -> Optional[str]:
    """The rarity whose brawlers the player can currently pick on the Starr Road: the lowest
    purchasable tier that still has an unowned, *vouched-for* brawler (the road is walked tier by
    tier; a catalog entry the data has never seen doesn't hold the tier open)."""
    for rarity in _STARR_ROAD_ORDER:
        if any(b.rarity == rarity and b.id not in owned and b.id in vouched for b in brawlers):
            return rarity
    return None


def _select(recs: List[_Rec], top: int, min_per_kind: int) -> List[_Rec]:
    """Top ``top`` by score, but first reserve the best ``min_per_kind`` of every kind (when
    there are that many) so no purchase kind is silently starved out of the list. Output keeps
    the global sort order; it only exceeds ``top`` when the reserved slots alone do."""
    if min_per_kind <= 0 or top <= 0:
        return recs[:top]
    chosen: List[_Rec] = []
    seen = set()
    per_kind: Dict[str, int] = {}
    for r in recs:
        if per_kind.get(r.kind, 0) < min_per_kind:
            per_kind[r.kind] = per_kind.get(r.kind, 0) + 1
            chosen.append(r)
            seen.add(id(r))
    for r in recs:
        if len(chosen) >= top:
            break
        if id(r) not in seen:
            chosen.append(r)
            seen.add(id(r))
    chosen.sort(key=_sort_key)
    return chosen


# --- per-kind rec builders ----------------------------------------------------------------------

def _step(kind: str, label: str, cost: Dict[str, int]) -> dict:
    return {"kind": kind, "label": label, "cost": cost}


def _boost_note(ctx: _Ctx, b) -> str:
    return (" Free at Power 11 with a full loadout this season, so this only matters once the boost"
            " rotates out." if b.id in ctx.boosted else "")


def _access_rec(ctx: _Ctx, b, s: float, st: Optional[OwnedState]) -> _Rec:
    """The "ranked-ready" package for a brawler the player can't field yet: the Starr Road unlock
    (if unowned), the power climb to the bracket floor, and a first gadget + star power if it has
    none — everything needed before the brawler is a real option in their Ranked bracket. Valued
    additively: access (a bare fieldable brawler) + every item the climb realizes, owned or bought."""
    pr = ctx.priors
    steps: List[dict] = []
    cost: Dict[str, int] = {}
    estimated = False
    lift = pr["access"]
    cur = 1 if st is None else max(1, ctx.effective_power(st))
    if st is None:
        credits, est = ctx.credits(b.rarity)
        estimated = estimated or est
        c = {"credits": int(credits)} if credits else {}
        steps.append(_step("new_brawler", f"Unlock ({b.rarity})", c))
        cost = _merge_cost(cost, c)
    if cur < ctx.floor:
        c = ctx.power_cost(cur, ctx.floor)
        steps.append(_step("power_upgrade", f"Power {cur}→{ctx.floor}", c))
        cost = _merge_cost(cost, c)
    # Core build: the first gadget / star power (catalog permitting). Their gates (7 / 9) sit at or
    # below every floor, so they're buyable the moment the climb is done.
    need = []
    if b.gadgets and (st is None or not st.gadgets):
        c = ctx.item_cost("gadget")
        steps.append(_step("gadget", "First gadget", c)); cost = _merge_cost(cost, c)
        lift += pr["gadget_first"]; need.append("gadget")
    if b.star_powers and (st is None or not st.star_powers):
        c = ctx.item_cost("star_power")
        steps.append(_step("star_power", "First star power", c)); cost = _merge_cost(cost, c)
        lift += pr["star_power_first"]; need.append("star power")
    if st is not None:
        lift += ctx.owned_item_lift(b, st)      # the climb realizes what's already on the brawler

    value = lift * ctx.weight(b.id, s)
    depth_note = ctx.depth_note(b.id)
    build_note = f" plus a {' and '.join(need)} to build" if need else ""
    if st is None:
        why = (f"Unlock {b.name} ({b.rarity}, on your Starr Road tier) and climb to Power "
               f"{ctx.floor}{build_note} — a {_pct(s)}-win-rate brawler you don't own; "
               f"{depth_note}.{_boost_note(ctx, b)}")
        return ctx.make(brawler_id=b.id, brawler_name=b.name, kind="new_brawler", value_lift=value,
                        cost=cost, meta_winrate=s, confidence="heuristic", rationale=why,
                        steps=steps, cost_estimated=estimated, target_power=ctx.floor,
                        gate=f"requires Power {ctx.floor}")
    if b.id in ctx.boosted:
        why = (f"{b.name} is free at Power 11 with a full loadout this season, so this climb"
               f"{build_note} only pays off once the boost rotates out — then it makes a "
               f"{_pct(s)}-win-rate brawler ranked-ready; {depth_note}.")
    else:
        why = (f"{b.name} can't be fielded in your bracket below Power {ctx.floor} — the climb"
               f"{build_note} makes a {_pct(s)}-win-rate brawler ranked-ready; {depth_note}.")
    return ctx.make(brawler_id=b.id, brawler_name=b.name, kind="power_upgrade", value_lift=value,
                    cost=cost, meta_winrate=s, confidence="heuristic", rationale=why, steps=steps,
                    cost_estimated=estimated, item_name=f"Power {cur}→{ctx.floor}",
                    target_power=ctx.floor, gate=f"requires Power {ctx.floor}")


def _accessory_rec(ctx: _Ctx, b, s: float, acc, label: str, prior: float,
                   delta: Optional[float], is_first: bool) -> _Rec:
    # A first item: the measured delta (item-vs-the-other-item) only says WHICH to buy, not how
    # much a first item is worth — it picks, it doesn't scale. A second item: the delta says how
    # much you'd actually gain by switching, so it scales (signed, capped).
    mult = 1.0 if is_first else ctx.delta_multiplier(delta)
    value = prior * mult * ctx.weight(b.id, s)
    cost = ctx.item_cost(acc.kind)
    ordinal = "first" if is_first else "second"
    wr = f"{_pct(s)}-win-rate brawler you field"
    if delta is not None and is_first:
        conf, why = "measured", (f"{b.name}'s first {label} — {acc.name} measures "
                                 f"{delta * 100:+.1f}% vs the other {label}; a core slot still empty "
                                 f"on a {wr}.")
    elif delta is not None and delta > 0:
        conf, why = "measured", (f"{b.name}'s second {label} — {acc.name} measures +{delta * 100:.1f}% "
                                 f"vs the one you own, so you'd actually switch; on a {wr}.")
    elif delta is not None:
        conf, why = "measured", (f"{b.name}'s second {label} — {acc.name} measures {delta * 100:.1f}% "
                                 f"vs the one you own: flexibility only.")
    elif is_first:
        conf, why = "heuristic", f"{b.name}'s first {label} — a core loadout slot still empty on a {wr}."
    else:
        conf, why = "heuristic", (f"{b.name}'s second {label} — loadout flexibility on a {wr}; "
                                  f"lower value than a first {label}.")
    return ctx.make(brawler_id=b.id, brawler_name=b.name, kind=acc.kind, value_lift=value,
                    cost=cost, meta_winrate=s, confidence=conf, rationale=why + _boost_note(ctx, b),
                    steps=[_step(acc.kind, acc.name, cost)],
                    item_id=acc.id, item_name=acc.name, item_delta=delta)


def _gear_rec(ctx: _Ctx, b, st: OwnedState, s: float, gears_guide) -> Optional[_Rec]:
    missing = [g for g in gears_guide if g[1] not in st.gears]
    if not missing:
        return None
    # Which gear: best significant measured gear (a measured-negative one sinks below unmeasured),
    # else the best editorial fit (base + class role).
    def gear_key(g):
        d = _measured_delta(ctx.itemstats, b.id, None, gear_name=g[0])
        return (d if d is not None and d > 0 else (-1.0 if d is None else -2.0 + d),
                g[2] + float(g[3].get(b.cls, 0.0)))
    best = max(missing, key=gear_key)
    delta = _measured_delta(ctx.itemstats, b.id, None, gear_name=best[0])
    slots = ctx.gear_slots(st)
    core = len(st.gears) < slots
    if not core and not (delta is not None and delta > 0):
        # A spare gear with no measured reason to swap isn't a recommendation — it's filler that
        # the per-kind reservation would otherwise force into every list.
        return None
    prior = ctx.priors["gear_core"] if core else ctx.priors["gear_extra"]
    value = prior * ctx.delta_multiplier(delta) * ctx.weight(b.id, s)
    cost = ctx.item_cost("gear")
    if core:
        slot = "fills an empty gear slot"
    elif slots < len(_GEAR_SLOT_POWERS):
        slot = f"a spare gear (the second slot opens at Power {_GEAR_SLOT_POWERS[-1]})"
    else:
        slot = "a spare gear for flexibility (both slots already filled)"
    if delta is not None:
        conf, why = "measured", (f"{best[0]} gear on {b.name} — measures {delta * 100:+.1f}% win rate; "
                                 f"{slot}.")
    else:
        conf, why = "heuristic", f"{best[0]} gear on {b.name} — {slot} on a {_pct(s)}-win-rate brawler."
    return ctx.make(brawler_id=b.id, brawler_name=b.name, kind="gear", value_lift=value,
                    cost=cost, meta_winrate=s, confidence=conf, rationale=why + _boost_note(ctx, b),
                    steps=[_step("gear", f"{best[0]} gear", cost)], item_name=best[0], item_delta=delta)


def _hypercharge_rec(ctx: _Ctx, b, s: float) -> _Rec:
    """A straight buy — only emitted at Power 11 (below it the climb is the next purchase)."""
    cost = ctx.item_cost("hypercharge")
    value = ctx.priors["hypercharge"] * ctx.weight(b.id, s)
    why = f"{b.name}'s Hypercharge — the biggest single upgrade on a {_pct(s)}-win-rate brawler you field."
    return ctx.make(brawler_id=b.id, brawler_name=b.name, kind="hypercharge", value_lift=value,
                    cost=cost, meta_winrate=s, confidence="eligibility_only",
                    rationale=why + _boost_note(ctx, b),
                    steps=[_step("hypercharge", "Hypercharge", cost)], item_name="Hypercharge")


def _power_rec(ctx: _Ctx, b, st: OwnedState, s: float) -> _Rec:
    """A fieldable brawler still short of Power 11 (only under the Power-9 floor): the remaining
    levels are real stats (+10% of base each), open the second gear slot / Hypercharge, and buy
    Mythic readiness — you'll need Power 11 to keep fielding it once you promote, so the climb
    earns a share (``readiness``) of the ranked-ready value now."""
    pr = ctx.priors
    cur = ctx.effective_power(st)
    cost = ctx.power_cost(cur, MAX_POWER)
    levels = MAX_POWER - cur
    ready = pr["access"] + ctx.owned_item_lift(b, st)
    lift = pr["power_level"] * levels + pr["readiness"] * ready
    value = lift * ctx.weight(b.id, s)
    unlocks = []
    if cur < _GEAR_SLOT_POWERS[-1]:
        unlocks.append("the second gear slot")
    unlocks.append("the Hypercharge slot")
    why = (f"{b.name} Power {cur}→{MAX_POWER} — about +{_rel_stats(cur, MAX_POWER)}% HP and damage on a "
           f"{_pct(s)}-win-rate brawler you field, opens {' and '.join(unlocks)}, and keeps it "
           f"fieldable once you reach Mythic (Power 11 required there).")
    return ctx.make(brawler_id=b.id, brawler_name=b.name, kind="power_upgrade", value_lift=value,
                    cost=cost, meta_winrate=s, confidence="heuristic",
                    rationale=why + _boost_note(ctx, b),
                    steps=[_step("power_upgrade", f"Power {cur}→{MAX_POWER}", cost)],
                    item_name=f"Power {cur}→{MAX_POWER}", target_power=MAX_POWER)
