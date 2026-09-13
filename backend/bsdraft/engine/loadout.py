"""Post-draft loadout advice: which gadget / star power / gear to equip on a drafted brawler.

Rule-based, in the same spirit as :mod:`bsdraft.engine.gameplan`. It classifies each of a
brawler's gadgets and star powers by *effect* (parsed from the catalog description) and scores
that effect against the game mode, rather than asserting a tier-list read the match data can't
support (battle logs record the brawler, never the equipped item). Gears come from a small
curated guide (``data/reference/gears.json``) since no catalog exposes them. When the caller
passes the opposing picks, a bounded enemy-comp overlay (class-count reads -> per-effect deltas,
clamped) nudges the same fit scale — see the "enemy-comp overlay" block below.

The response carries a ``fit`` score (0..1) and a ``source`` tag per item so the UI can label it
and a later data-driven pass can swap the heuristic for a measured win rate without changing the
shape. That upgrade — *single-item-owner inference*: attribute a player's recent brawler matches
to their one owned gadget/star power/gear and diff against zero-owners — is the planned Phase 2,
and it reuses the owned-item ids now retained in :mod:`bsdraft.engine.mastery`.

Pure stdlib so it stays importable on the serve path (no torch/sklearn/pandas).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import List, Optional

from bsdraft.constants import REFERENCE_DIR
from bsdraft.data import reference as R
from bsdraft.engine import itemstats as IS
from bsdraft.engine.composition import FRONTLINE, RANGE

# Maps a measured win-rate delta (points) onto the 0..1 `fit` scale the UI already sorts by, so a
# measured pick slots in beside the heuristic fits: +5% -> 0.65, +15% -> 0.95, -5% -> 0.35.
_FIT_PER_DELTA = 3.0

# ---- effect taxonomy -------------------------------------------------------------------------
# Each gadget/star power is bucketed into one effect by scanning its description for keywords.
# Order matters: the first matching bucket wins, so the list is tuned "headline effect first"
# (a dash-that-also-reloads reads as mobility) and DAMAGE sits late because many descriptions
# mention "damage" only incidentally ("deals x damage"). Matching is word-aware: raw substring
# matching turned "removes" into mobility via the word "moves", and catalog value tokens such as
# ``speedReducePercent`` into movement effects.
_EFFECT_KEYWORDS = [
    ("mobility", ("dash", "dashes", "jump", "jumps", "leap", "teleport", "sprint", "roll",
                  "charges forward", "moves to", "moves faster", "movement speed", "hop")),
    ("reload",   ("reload", "reloads", "ammo", "ammunition", "attack speed")),
    ("sustain",  ("heal", "heals", "healing", "restore", "restores", "shield", "regenerat",
                  "regain", "invulnerab", "immune", "immunity")),
    ("control",  ("slow", "slows", "slowing", "stun", "stuns", "freeze", "frozen", "knock", "push",
                  "pushes", "pull", "pulls", "root", "snare", "silence")),
    ("vision",   ("vision", "reveal", "reveals", "bush", "bushes", "detect")),
    ("damage",   ("damage", "deals", "pierc", "poison", "burn", "explo", "destroy")),
    # `range` sits last (before the utility fallback): a lot of descriptions mention "range"
    # incidentally ("+damage at max range"), so a real damage/effect keyword should win first.
    ("range",    ("range", "longer", "farther", "further", "reach")),
]
# A few kits alter Super availability without using any of the ordinary effect words above.
# Keep this as phrase matching rather than adding bare "charge" to the keyword buckets: phrases
# such as "Super charging area" describe where an ability works, not an item that charges Super.
_SUPER_CHARGE_RE = re.compile(
    r"\b(?:re)?charg(?:e|es|ed|ing)\b[^.!?]{0,40}\bsuper\b"
    r"|\bcharg(?:e|es|ed|ing)\s+up\b[^.!?]{0,40}\bsuper\b"
    r"|\bsuper\s+(?:charge|charging)\s+rate\b"
    r"|\bsuper\s+(?:fully\s+)?charged\b"
    r"|\bsuper\b[^.!?]{0,24}\brecharged\b",
    re.IGNORECASE,
)
_EFFECT_META = {
    "mobility": ("Mobility", "reposition, engage, or escape"),
    "reload":   ("Reload / ammo", "more attack uptime and burst"),
    "sustain":  ("Sustain", "heal or shield to survive trades"),
    "control":  ("Control / CC", "slow, push, or lock down enemies"),
    "vision":   ("Vision", "reveal enemies in bushes"),
    "range":    ("Range", "poke from a safer distance"),
    "damage":   ("Damage", "raw damage output"),
    "super_charge": ("Super charge", "get your Super online sooner"),
    "projectile_defense": ("Projectile defense", "deny incoming shots and cross firing lanes"),
    "utility":  ("Utility", "situational value"),
}
# How valuable each effect is per mode (0..1). Absent effects fall back to _NEUTRAL. Kept in sync
# with the mode priorities encoded in engine.gameplan (survive Knockout, mobility for Brawl Ball,
# burst for Heist, range for Bounty, control/sustain for the zone modes).
_NEUTRAL = 0.35
_MODE_EFFECT = {
    "Gem Grab":   {"sustain": 0.80, "control": 0.75, "reload": 0.60, "vision": 0.60,
                   "mobility": 0.50, "damage": 0.50, "range": 0.50,
                   "projectile_defense": 0.65},
    "Brawl Ball": {"mobility": 0.85, "control": 0.70, "damage": 0.60, "sustain": 0.55,
                   "reload": 0.50, "range": 0.45, "vision": 0.45,
                   "projectile_defense": 0.75},
    "Knockout":   {"sustain": 0.80, "control": 0.70, "vision": 0.65, "range": 0.62,
                   "damage": 0.58, "reload": 0.55, "mobility": 0.55,
                   "projectile_defense": 0.70},
    "Heist":      {"damage": 0.85, "reload": 0.75, "mobility": 0.60, "range": 0.52,
                   "control": 0.50, "sustain": 0.45, "vision": 0.35,
                   "projectile_defense": 0.45},
    "Hot Zone":   {"control": 0.80, "sustain": 0.75, "damage": 0.60, "reload": 0.55,
                   "mobility": 0.50, "vision": 0.50, "range": 0.50,
                   "projectile_defense": 0.70},
    "Bounty":     {"range": 0.80, "damage": 0.70, "vision": 0.65, "mobility": 0.55,
                   "sustain": 0.55, "control": 0.55, "reload": 0.55,
                   "projectile_defense": 0.65},
}
# A faster Super is broad tempo: especially useful for objective pressure and Brawl Ball, less
# decisive than raw damage in Heist and Bounty. This is a prior, not a measured item read.
_SUPER_CHARGE_MODE = {
    "Gem Grab": 0.62, "Brawl Ball": 0.70, "Knockout": 0.64,
    "Heist": 0.55, "Hot Zone": 0.65, "Bounty": 0.60,
}
_PROJECTILE_DEFENSE_RE = re.compile(
    r"\b(?:destroy|destroys|destroying)\b[^.!?]{0,30}\bincoming\s+projectiles?\b",
    re.IGNORECASE,
)
_DIRECT_DAMAGE_RE = re.compile(
    r"\b(?:damage|damages|deals|poison|poisons|burn|burns|explode|explodes|explosion)\b",
    re.IGNORECASE,
)
_PREFIX_KEYWORDS = frozenset(("explo", "pierc", "regenerat", "invulnerab", "destroy"))

# ---- enemy-comp overlay ----------------------------------------------------------------------
# Comp-aware adjustment on top of the mode fit: enemy class counts fire coarse "reads" (all
# thresholded at >=2 so a lone enemy pick fires nothing), each read contributing small per-effect
# deltas shaped exactly like _MODE_EFFECT. Reads co-fire and sum; the total per effect is clamped
# at ±_COMP_CLAMP — calibrated so the heuristic can never claim more than a strong measured signal
# is worth (+5% measured win rate maps to +0.15 fit via _FIT_PER_DELTA). Class counts drive the
# aggro/tanky/poke reads (class is the only playstyle attribute the reference carries); the Phase-2
# cc_heavy read additionally inspects each enemy's gadget/SP KIT for control (via _cc_kit_ids). It
# never models the enemy's *specific equipped* item — logs don't record that — so the reads stay
# coarse (class counts + a has-CC-in-kit flag) by design, not a per-item enemy-loadout read.
_COMP_CLAMP = 0.15
_COMP_CHIP_MIN = 0.04   # below this applied delta, don't clutter the item with reason chips
_COMP_EFFECT = {
    # 2+ Tank/Assassin: divers reach you — control/peel and sustain gain value, poke range less so.
    "aggro": {"control": 0.10, "sustain": 0.06, "mobility": 0.04, "range": -0.03},
    # 2+ Tanks specifically: raw damage to melt HP pools, control to kite (gameplan's "kite the Tank").
    "tanky": {"damage": 0.10, "control": 0.04},
    # 2+ Marksman/Controller/Artillery: close the gap or dodge — mobility and sustain to survive poke.
    "poke": {"mobility": 0.08, "sustain": 0.06},
    # ALL enemies carry CC in their gadget/SP kits: escape and sustain to survive the lock-down.
    "cc_heavy": {"mobility": 0.06, "sustain": 0.04},
}
_COMP_CHIP = {"aggro": "vs dive", "tanky": "vs tanks", "poke": "vs poke", "cc_heavy": "vs CC"}
_COMP_READ_LABEL = {"aggro": "dive-heavy ({n} Tank/Assassin)", "tanky": "{n} Tanks",
                    "poke": "poke-heavy ({n} ranged)", "cc_heavy": "CC-heavy kits ({n})"}
# Fire thresholds per read. Class reads fire at 2-of-3. cc_heavy needs the FULL enemy team:
# the 2026-08-18 kit audit put the corrected CC-kit base rate at ~54% of the roster, so a >=2
# threshold would fire in over half of all drafts — a near-constant, not a signal.
_COMP_READ_MIN = {"aggro": 2, "tanky": 2, "poke": 2, "cc_heavy": 3}

# Corrections to the keyword scan behind _cc_kit_ids, from a full-roster audit of every gadget/SP
# description (2026-08-18; item-level false-positive rate 8%). False: the control keyword refers to
# the brawler moving ITSELF. True: real enemy CC the first-match-wins scan bucketed elsewhere
# (mobility/damage/sustain won first) — pulls, knock-ups, silences, sleeps, hypnosis, zone slows.
# Name-keyed: an unknown name silently falls back to the scan, so catalog renames degrade safely.
_CC_KIT_OVERRIDES = {
    "Carl": False, "Janet": False,
    "Berry": True, "Bibi": True, "Bolt": True, "Bonnie": True, "Cordelius": True, "Draco": True,
    "El Primo": True, "Finx": True, "Frank": True, "Gene": True, "Moe": True, "Ollie": True,
    "Otis": True, "Pierce": True, "Sandy": True, "Sirius": True, "Wendy": True,
}

_TOKEN_RE = re.compile(r"<!.*?>")
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Strip the catalog's unfilled ``<!card.value…>`` tokens and collapse whitespace."""
    return _WS_RE.sub(" ", _TOKEN_RE.sub("", text or "")).strip()


def _num(x, default: float = 0.0) -> float:
    """Defensive float for hand-curated guide values: junk (null, strings, wrong types) degrades to
    the default rather than 500ing the endpoint — the gear guide's fail-safe contract."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _classify(description: str) -> str:
    # Remove the catalog's unfilled value tokens before matching. Besides making the text shown to
    # users cleaner, this prevents fields such as ``speedReducePercent`` from looking like a
    # movement effect and lets Super-charge phrases span a removed numeric token.
    d = _clean(description).lower()
    # A few gadgets trade damage for a faster Super. Only let an explicit Super-charge phrase
    # win when the description has no direct damage payload; Darryl's spin, for example, deals
    # damage and recharges his Super, so it should remain a damage gadget.
    if _SUPER_CHARGE_RE.search(d) and not _DIRECT_DAMAGE_RE.search(d):
        return "super_charge"
    if _PROJECTILE_DEFENSE_RE.search(d):
        return "projectile_defense"
    for effect, keywords in _EFFECT_KEYWORDS:
        if any(_keyword_match(d, k) for k in keywords):
            return effect
    return "utility"


def _keyword_match(text: str, keyword: str) -> bool:
    """Match a catalog phrase without letting it fire inside another word or value token."""
    token = keyword.strip().lower()
    if not token:
        return False
    if token in _PREFIX_KEYWORDS:
        return bool(re.search(rf"\b{re.escape(token)}", text))
    return bool(re.search(rf"\b{re.escape(token)}\b", text))


def _mode_fit(effect: str, mode: str) -> float:
    if effect == "super_charge":
        return _SUPER_CHARGE_MODE.get(mode, _NEUTRAL)
    return _MODE_EFFECT.get(mode, {}).get(effect, _NEUTRAL)


@lru_cache(maxsize=1)
def _by_id() -> dict:
    return {b.id: b for b in R.load_brawlers()}


@lru_cache(maxsize=1)
def _cc_kit_ids() -> frozenset:
    """Ids of brawlers whose gadget/SP kit offers real enemy-targeted CC: the control-bucket
    keyword scan, corrected by the audited ``_CC_KIT_OVERRIDES``. Kits are static per process."""
    out = set()
    for b in R.load_brawlers():
        has = any(_classify(a.description) == "control" for a in b.gadgets + b.star_powers)
        if _CC_KIT_OVERRIDES.get(b.name, has):
            out.add(b.id)
    return frozenset(out)


@lru_cache(maxsize=1)
def _gear_guide() -> dict:
    """Curated gear guide (hand-maintained; gears aren't in any catalog). Fail-safe to empty so a
    missing/broken file degrades to 'no gear tips' rather than erroring the whole endpoint."""
    path = REFERENCE_DIR / "gears.json"
    if not path.exists():
        return {"note": "", "gears": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"note": "", "gears": []}
    return {"note": doc.get("note", ""), "gears": doc.get("gears", []) or []}


def _comp_overlay(enemies: Optional[List[int]], self_id: Optional[int] = None) -> Optional[dict]:
    """Characterize the enemy comp (class counts only) and derive the per-effect fit deltas.

    ``None`` when no *known* enemies were given — the comp layer is absent, not defaulted, mirroring
    how scoring leaves ``counter`` unset until ``their_team`` is non-empty. Unknown ids and
    Unclassified brawlers contribute nothing. Ids are deduped and the queried brawler is filtered
    out: the frontend can't send duplicates/self, but the endpoint is public and never-4xx, so a
    hand-crafted ``enemies=<tank>,<tank>`` must not fire the >=2-threshold reads off one opponent.
    """
    byid = _by_id()
    ids = [e for e in dict.fromkeys(enemies or []) if e != self_id and e in byid]
    if not ids:
        return None
    classes = [byid[e].cls for e in ids]
    counts = {
        "aggro": sum(c in FRONTLINE for c in classes),
        "tanky": sum(c == "Tank" for c in classes),
        "poke": sum(c in RANGE for c in classes),
        "cc_heavy": sum(e in _cc_kit_ids() for e in ids),
    }
    # cc_heavy additionally requires the WHOLE list to be CC-kit brawlers: its semantics are "the
    # full enemy team", and the public endpoint accepts up to 5 hand-crafted ids — 3-of-5 would
    # recreate the near-constant-fire problem the full-team threshold exists to avoid.
    fired = [k for k in _COMP_EFFECT
             if counts[k] >= _COMP_READ_MIN[k] and (k != "cc_heavy" or counts[k] == len(ids))]
    bonus: dict = {}
    chips: dict = {}
    for read in fired:
        for eff, d in _COMP_EFFECT[read].items():
            bonus[eff] = bonus.get(eff, 0.0) + d
            chips.setdefault(eff, []).append(("+ " if d > 0 else "− ") + _COMP_CHIP[read])
    bonus = {e: max(-_COMP_CLAMP, min(_COMP_CLAMP, v)) for e, v in bonus.items()}
    reads = [_COMP_READ_LABEL[k].format(n=counts[k]) for k in fired]
    return {"bonus": bonus, "chips": chips, "reads": reads, "fired": fired}


def _fold_comp(item: dict, adj: float, chips: List[str]) -> None:
    """Fold a comp adjustment into ``fit`` AFTER the measured overlay, recording the applied
    (post-clamp01) delta so ``fit - comp_delta`` reconstructs the comp-blind fit exactly. ``why`` is
    never rewritten — the measured claim stays verbatim; the signed chips are the separate channel."""
    if not adj:
        return
    base = item["fit"]
    new = max(0.0, min(1.0, base + adj))
    item["comp_delta"] = round(new - base, 3)
    item["fit"] = round(new, 3)
    if abs(item["comp_delta"]) >= _COMP_CHIP_MIN:
        item["comp_why"] = list(chips)


def _apply_comp(item: dict, effect: str, overlay: Optional[dict]) -> None:
    """Accessory path: the comp adjustment for the item's effect bucket, from the overlay tables."""
    if not overlay:
        return
    _fold_comp(item, overlay["bonus"].get(effect, 0.0), overlay["chips"].get(effect, []))


def _apply_measured(item: dict, cell: dict) -> None:
    """Overlay a *significant* measured win-rate cell onto a heuristic item: flip source to
    'winrate', set fit from the delta, and replace the reasoning with the measured number. The delta
    is the item's edge over the brawler's OTHER single-owned items of that type (already shrunk +
    BH-FDR-gated at build time), and is honestly signed — a measured negative is shown, not hidden;
    it just won't win :func:`_mark_best` (lower fit)."""
    delta = float(cell.get("delta", 0.0))
    item["source"] = "winrate"
    item["fit"] = round(min(1.0, max(0.0, 0.5 + _FIT_PER_DELTA * delta)), 3)
    sign = "+" if delta >= 0 else "−"
    kind = item["kind"].replace("_", " ")
    item["why"] = (f"Measured {sign}{abs(delta) * 100:.1f}% win rate vs this brawler's other "
                   f"{kind}s (single-item owners, n≈{cell.get('n_eff', 0):.0f})")


def _advise_accessory(acc: R.Accessory, mode: str, brawler_id: int, itemstats: Optional[dict],
                      comp: Optional[dict] = None) -> dict:
    effect = _classify(acc.description)
    label, blurb = _EFFECT_META[effect]
    item = {
        "id": acc.id,
        "name": acc.name,
        "kind": acc.kind,
        "image_url": acc.image_url,
        "effect": label,
        "description": _clean(acc.description),
        "fit": round(_mode_fit(effect, mode), 3),
        "recommended": False,  # set by _mark_best once the whole kind is scored
        "why": f"{label}: {blurb}.",
        "source": "heuristic",
        "comp_delta": 0.0,     # applied enemy-comp adjustment; fit - comp_delta = comp-blind fit
        "comp_why": [],        # signed reason chips, e.g. "+ vs dive"
        "comp_flipped": False, # set by _mark_best when the pick only wins because of the comp
    }
    cell = IS.accessory_cell(itemstats, brawler_id, acc.id)
    if cell and cell.get("significant"):
        _apply_measured(item, cell)
    _apply_comp(item, effect, comp)   # after measured: folds into fit, never touches why/source
    return item


def _mark_best(items: List[dict], mode: str) -> None:
    """Flag the single best item of a kind. Prefers a MEASURED item that is measured *better* than
    the brawler's other items of the type (positive delta, i.e. fit > 0.5) — a measured edge beats an
    effect guess. A lone measured item that's measured *worse* must NOT be recommended, so when no
    positive-measured item exists we fall back to the best fit over all items (which lets a stronger
    heuristic sibling win over a negative-measured one).

    The enemy-comp overlay folds into ``fit`` with the applied delta recorded on ``comp_delta``, so
    measured-vs-heuristic arbitration runs on the comp-blind base fit: the measured-better test uses
    it, and a measured-WORSE item competes at it (a comp bump must never promote an item whose own
    measurement says it loses to its siblings). Heuristic items rank by the final comp-adjusted fit.
    When the comp adjustment alone changes the winner, the pick is flagged ``comp_flipped``."""
    if not items:
        return

    def base(it: dict) -> float:
        # fit and comp_delta are each 3-dp rounded; round the reconstruction too, or ~1e-16 float
        # noise mis-resolves TRUE base-fit ties in the exact max comparisons below.
        return round(it["fit"] - it.get("comp_delta", 0.0), 3)

    def rank(it: dict) -> float:
        if it["source"] == "winrate" and base(it) <= 0.5:
            return base(it)   # measured-worse: the comp bump can't lift it over siblings
        return it["fit"]

    measured_better = [it for it in items if it["source"] == "winrate" and base(it) > 0.5]
    pool = measured_better or items
    best = max(pool, key=rank)
    if best is not max(pool, key=base):
        best["comp_flipped"] = True
    best["recommended"] = True
    if best["source"] == "winrate":
        best["why"] = "Top measured pick — " + best["why"][0].lower() + best["why"][1:]
    else:
        best["why"] = f"Best {best['kind'].replace('_', ' ')} fit for {mode} — {best['why'][0].lower()}{best['why'][1:]}"


def _gear_items(cls: str, mode: str, brawler_id: int, itemstats: Optional[dict],
                comp: Optional[dict] = None, top: int = 2) -> List[dict]:
    guide = _gear_guide()
    out: List[dict] = []
    for g in guide["gears"]:
        modes = g.get("modes") if isinstance(g.get("modes"), dict) else {}
        roles = g.get("roles") if isinstance(g.get("roles"), dict) else {}
        fit = max(0.0, min(1.0, _num(g.get("base", 0.4), 0.4)
                           + _num(modes.get(mode)) + _num(roles.get(cls))))
        name = g.get("name", "")
        item = {
            "id": None,
            "name": name,
            "kind": "gear",
            "image_url": "",
            "effect": g.get("effect", ""),
            "description": g.get("description", ""),
            "fit": round(fit, 3),
            "recommended": False,
            "why": (f"Situational — {g.get('effect','').lower()}." if g.get("effect") else "Situational."),
            "source": "curated",
            "comp_delta": 0.0,
            "comp_why": [],
            "comp_flipped": False,
        }
        cell = IS.gear_cell(itemstats, brawler_id, name)
        if cell and cell.get("significant"):
            _apply_measured(item, cell)
        # Comp offsets extend the curated additive idiom (base + modes + roles + vs): each gear may
        # carry a sparse "vs" dict keyed by read keys; fired reads sum, clamped like the accessories.
        if comp and comp["fired"]:
            vs = g.get("vs") if isinstance(g.get("vs"), dict) else {}
            adj = sum(_num(vs.get(k)) for k in comp["fired"])
            adj = max(-_COMP_CLAMP, min(_COMP_CLAMP, adj))
            chips = [("+ " if _num(vs.get(k)) > 0 else "− ") + _COMP_CHIP[k]
                     for k in comp["fired"] if _num(vs.get(k))]
            _fold_comp(item, adj, chips)
        out.append(item)

    # Display order stays final-fit descending, but the top-two SELECTION uses the same measured
    # arbitration as _mark_best: a measured-WORSE gear (comp-blind base fit <= 0.5) competes at its
    # base fit, so a comp bump can never promote a gear its own measurement says loses to its
    # siblings. A gear in the picks only because of the comp is flagged comp_flipped, mirroring the
    # accessory contract.
    def base_fit(it: dict) -> float:
        return round(it["fit"] - it.get("comp_delta", 0.0), 3)

    def rank(it: dict) -> float:
        if it["source"] == "winrate" and base_fit(it) <= 0.5:
            return base_fit(it)
        return it["fit"]

    out.sort(key=lambda it: it["fit"], reverse=True)
    aware = sorted(out, key=rank, reverse=True)[:top]
    blind = sorted(out, key=base_fit, reverse=True)[:top]
    for item in aware:
        item["recommended"] = True
        if not any(item is b for b in blind):
            item["comp_flipped"] = True
        if item["source"] != "winrate":
            item["why"] = (f"Core pick for {mode}" +
                           (f" — {item['effect'].lower()}." if item["effect"] else "."))
    return out


def loadout_advice(brawler_id: int, mode: str, map_id: Optional[int] = None,
                   enemies: Optional[List[int]] = None) -> Optional[dict]:
    """Loadout advice for a drafted brawler, or ``None`` if the brawler id is unknown.

    ``enemies`` (ids of the queried brawler's opponents, seat-flip resolved by the caller) turns on
    the comp-aware overlay; absent/empty keeps today's comp-blind behavior byte-identical.
    ``map_id`` is accepted for a future per-map slice; the heuristic is mode-level today.
    """
    b = _by_id().get(brawler_id)
    if b is None:
        return None
    itemstats = IS.get_itemstats()   # None until the table is built/synced -> pure heuristic
    comp = _comp_overlay(enemies, self_id=b.id)
    gadgets = [_advise_accessory(a, mode, b.id, itemstats, comp) for a in b.gadgets]
    star_powers = [_advise_accessory(a, mode, b.id, itemstats, comp) for a in b.star_powers]
    _mark_best(gadgets, mode)
    _mark_best(star_powers, mode)
    gears = _gear_items(b.cls, mode, b.id, itemstats, comp)
    measured = any(it["source"] == "winrate" for it in gadgets + star_powers + gears)
    note = ("Measured win rates where the sample is sufficient; effect-based fit otherwise."
            if measured else f"Effect-based fit for {mode} — not a live tier read yet.")
    comp_reads = comp["reads"] if comp else []
    if comp_reads:
        note += " Adjusted for the enemy comp."
    return {
        "brawler_id": b.id,
        "brawler_name": b.name,
        "cls": b.cls,
        "mode": mode,
        "gadgets": gadgets,
        "star_powers": star_powers,
        "gears": gears,
        "note": note,
        "comp_reads": comp_reads,
    }
