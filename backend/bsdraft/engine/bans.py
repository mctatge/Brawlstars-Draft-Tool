"""Ban valuation: rank bans by what they actually deny, given everything already banned.

A ban is worth the difference between the draft that happens with the brawler available and the
draft that happens without it — not the brawler's raw strength. That difference is *conditional
on the rest of the ban set*, which is why this list re-ranks as your teammates ban and a static
threat table does not:

  * **Substitutes absorb a ban.** When a comparable brawler covers the same job, banning either
    barely moves the draft — the enemy just builds the same team around the other one.
  * **The survivor's value jumps.** Once the substitute is gone, banning the last brawler that
    can do that job costs the enemy a real drop-off, so it climbs the list.
  * **First pick makes a ban self-harm.** Brawl Stars bans are global — a banned brawler is gone
    for *both* teams. With first pick you would simply take the top threat, so banning it denies
    you, not them; it drops to a lower tier of the list and is flagged, because it is advice about
    your pick rather than your ban. The 1-2-2-1 snake hands picks 1/4/5 to the first-pick team and
    2/3/6 to the other, so the right targets are the brawlers the *enemy* is in line to take.
  * **You can't ban what you can't use.** With a roster loaded, a brawler you don't own can never
    become one of your picks, so banning it costs you nothing.

Method — score every comp once, then price the ban from each side of it:

  **Who holds what.** Draft order is modelled as picks drawn in turn from a softmax over
  draftability, down the remaining 1-2-2-1 slots. A dominant brawler is spent almost entirely at
  the first slot, so whoever owns that slot is likely to hold it; an ordinary one spreads evenly
  and lands on either side. Our slots skip anything the roster says we can't field, and every
  brawler is scaled by the odds it survives the enemy's three unseen bans — nobody drafts from
  the pool as it stands, so nobody's comps should be built from it.

  **What a ban costs a side.** Score every 3-brawler comp the pool can still field — the
  partial-draft net reads a comp against an unknown board directly, which is exactly "how strong
  is this comp here" — and weight each by how likely that side is to end up holding all three.
  The aggregate is a *soft maximum*: how good is the best thing that side can still build,
  softened by how uncertain that is. Removing an option can only lower a soft max, and the drop
  reduces to a closed form in the share of the good comps that run through the brawler. That
  share is where substitutability lives: brawlers covering the same job rarely appear in a strong
  comp together, so each holds only part of the mass and neither is expensive to lose — until one
  is banned and the survivor inherits the lot.

  **The ban.** ``ban_value = deny(them) - SELF_COST_W * deny(us)``. What a ban takes from them is
  the case for it; what it takes from us is the case against, at a discount — we choose the ban
  knowing our own plan and can draft around the hole it leaves, and the roster tells us what we
  can field while theirs stays hidden. Weighted equally the two sides very nearly cancel, since
  both draft the same pool, and the list would rank on the residue.

  Note what the survival factor buys here: a brawler they were always going to ban carries almost
  no weight in *either* side's comps, so its denial collapses on both sides and spending our ban
  on it comes out worthless — the wasted-ban effect falls out of the arithmetic instead of being
  asserted by a discount. What it promotes is the ban they *aren't* making: on a map whose two
  loudest threats are certain enemy bans, the strongest thing they'll still be holding is the
  brawler nobody else is banning, and that is where our ban belongs.

**What you can actually see.** You never see the enemy's three bans before choosing yours, and
the only bans that can reach the board first are your two teammates' — so the ban list can only
re-rank twice in a draft, and all three enemy bans stay priced as unseen throughout it.

Deliberately smooth: no argmax anywhere. A projected draft line — pick the best, then the next
best — is a step function of its inputs, where one flipped pick cascades into a wholly different
board; it reads as a sophisticated model and behaves like a coin toss. Aggregating over comps
keeps every brawler's value continuous in the ban set.

**No ban data exists.** Supercell's battle log has no ban field (`collect/match.py`), so unlike
`scoring.DEFAULT_WEIGHTS` nothing here is fit to held-out matches. This is a model-derived
projection, and the constants below are deliberate priors, not tuned parameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from bsdraft.constants import NUM_BANS_PER_TEAM, PICK_ORDER, TEAM_SIZE
from bsdraft.data import reference as R
from bsdraft.engine.scoring import DEFAULT_WEIGHTS, _name_map
from bsdraft.engine.state import DraftState

POOL_N = 20          # brawlers valued at all; the rest can't realistically reach a pick
TEAM_N = 12          # brawlers the enemy's expected team is built from (C(12,3) = 220 comps)
# Softmax temperatures, in standard deviations of each sample's own spread. Draft phases revolve
# around a handful of brawlers, so that distribution is peaked; many comps are viable, so that one
# is broad — and a broad comp distribution is what keeps the denial estimate off an argmax.
DRAFT_TEMP_SD = 0.5
TEAM_TEMP_SD = 1.0
SELF_COST_W = 0.5    # what losing a brawler we'd have taken costs, against what it costs them
SELF_DENY_RATIO = 1.10   # how far our side's loss must exceed theirs to call it a self-ban
MIN_SURVIVAL = 0.25  # floor on the odds a brawler survives the enemy's unseen bans

# Draftability weighs the model against the empirical read in the same ratio the pick scorer
# gives them, so a brawler's worth reads consistently across the ban and pick phases.
_MODEL_SHARE = DEFAULT_WEIGHTS["model"] / (DEFAULT_WEIGHTS["model"] + DEFAULT_WEIGHTS["map"])


@dataclass
class BanScore:
    brawler_id: int
    name: str
    cls: str
    threat: float                       # standalone map threat (win-rate + how contested), 0..1
    map_winrate: float
    use_rate: float
    confidence: float
    ban_value: Optional[float] = None   # projected swing in our win prob; None = not projected
    replacement: Optional[str] = None   # who the enemy builds around instead
    self_deny: bool = False             # the draft projects this brawler onto *our* side


def _threat(winrate: float, use: float) -> float:
    """Standalone threat: how strong the brawler is here, nudged by how contested it is."""
    return 0.85 * winrate + 0.15 * min(1.0, use * 3.0)


def _prob_many(model, teams: Sequence[Sequence[int]], map_id: int, mode: str) -> List[float]:
    """Each team's strength against an unknown board, batched — one matmul on the real model."""
    if not teams:
        return []
    a, b = [list(t) for t in teams], [[] for _ in teams]
    if hasattr(model, "prob_batch"):
        return list(model.prob_batch(a, b, map_id, mode))
    return [model.prob(t, [], map_id, mode) for t in a]


def _temperature(scores: Sequence[float], temp_sd: float) -> float:
    """``temp_sd`` standard deviations of the scores' own spread.

    A fixed temperature can't serve every input here: comp strengths spread far wider than
    per-brawler draftability, and both spread differently on a settled map than a volatile one.
    Scaling to the sample keeps every distribution equally opinionated — a fixed value would
    collapse to an argmax on one input and to a flat average on another."""
    if len(scores) < 2:
        return 1.0
    mean = sum(scores) / len(scores)
    sd = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    return max(sd * temp_sd, 1e-9)


def _softmax(scores: Sequence[float], temp_sd: float) -> List[float]:
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    temp = _temperature(scores, temp_sd)
    top = max(scores)
    w = [math.exp((s - top) / temp) for s in scores]
    total = sum(w) or 1.0
    return [x / total for x in w]


def _draftability(state: DraftState, rows: Sequence[BanScore], model) -> Dict[int, float]:
    """How likely each brawler is to be drafted here, on a win-rate-like scale.

    Fuses the model's read of the brawler with standalone threat — which carries use-rate, the
    best evidence of what a real opponent actually picks. A brawler the model loves on a map
    nobody plays it on is not the pick that gets made, and a ban list built on picks nobody
    makes is worthless."""
    threat = {r.brawler_id: r.threat for r in rows}
    ids = [r.brawler_id for r in rows]
    solo = _prob_many(model, [[b] for b in ids], state.map_id, state.mode)
    return {b: _MODEL_SHARE * p + (1.0 - _MODEL_SHARE) * threat[b] for b, p in zip(ids, solo)}


def _denial(comps: Sequence[Sequence[int]], strength: Sequence[float], hold: Dict[int, float],
            pool: Sequence[int]) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """How far one side's expected comp falls if each brawler is off the board, plus who they'd
    build around instead.

    A comp counts for what it's worth *and* for the odds that side actually assembles it — you
    can't field three brawlers you were never going to hold, and without that factor the estimate
    collapses onto whichever comp the model likes best regardless of who gets it.

    The aggregator is a soft maximum over comps, not an average: "how good is the best thing they
    can still build", softened by how uncertain that is. An average is the wrong functional here —
    removing a below-average brawler *raises* a mean, so most of the pool would come out as
    negative denial and floor to a flat zero. A soft max can only fall when an option is taken
    away, which reduces to a closed form: a brawler's denial is set by the share of the good comps
    that run through it, so a brawler whose share is covered by a substitute is cheap to lose,
    and one holding unique mass is not."""
    tau = _temperature(strength, TEAM_TEMP_SD)
    top_s = max(strength)
    weight = []
    for i, c in enumerate(comps):
        odds = 1.0
        for b in c:
            odds *= hold.get(b, 0.0)
        weight.append(odds * math.exp((strength[i] - top_s) / tau))
    total = sum(weight)
    if total <= 0.0:
        return {b: 0.0 for b in pool}, {b: None for b in pool}
    best = max(range(len(comps)), key=lambda i: weight[i])

    deny: Dict[int, float] = {}
    instead: Dict[int, Optional[int]] = {}
    for x in pool:
        share = sum(w for w, c in zip(weight, comps) if x in c) / total
        deny[x] = -tau * math.log(max(1.0 - share, 1e-9))
        keep = [i for i, c in enumerate(comps) if x not in c]
        top = max(keep, key=lambda i: weight[i]) if keep else None
        instead[x] = next((b for b in comps[top] if b not in comps[best]), None) if top is not None else None
    return deny, instead


def _sides(state: DraftState, draftability: Dict[int, float], roster) -> Dict[int, Tuple[float, float]]:
    """Odds each brawler ends up on our side vs theirs, walking the remaining 1-2-2-1 slots.

    Each slot takes the brawler with probability proportional to its draftability among what's
    left, so a dominant brawler is claimed at the first slot — by whoever owns it. Our slots skip
    what the roster says we can't field, which is what makes an unowned brawler free to ban."""
    us_index = 0 if state.we_pick_first else 1
    start = min(len(state.our_team) + len(state.their_team), 2 * TEAM_SIZE)
    slots = [PICK_ORDER[i] == us_index for i in range(start, 2 * TEAM_SIZE)]

    ids = list(draftability)
    p = dict(zip(ids, _softmax([draftability[b] for b in ids], DRAFT_TEMP_SD)))
    out = {b: [0.0, 0.0] for b in ids}
    for j, slot_is_ours in enumerate(slots):
        # A slot takes exactly one brawler, so its odds are normalized across the pool. The
        # survival factor is what separates the slots: a dominant brawler is nearly all spent at
        # the first slot, while an unremarkable one is spread evenly down the snake — so who owns
        # the early slots decides the top of the pool and barely touches the rest.
        share = {b: p[b] * (1.0 - p[b]) ** j for b in ids}
        if slot_is_ours and roster is not None:
            share = {b: (w if b in roster else 0.0) for b, w in share.items()}
        total = sum(share.values()) or 1.0
        for b in ids:
            out[b][0 if slot_is_ours else 1] += share[b] / total
    return {b: (o, t) for b, (o, t) in out.items()}


def _hidden_enemy_bans(state: DraftState) -> int:
    """How many enemy bans are still unseen.

    You never see the enemy's three before choosing your own, and the only bans that can appear
    on the board before yours are your two teammates' — so through the whole ban phase this is
    3, whatever you've entered. It only falls once the reveal puts more than two bans on the
    board and the pick phase starts consuming them."""
    revealed = max(0, len(state.bans) - (NUM_BANS_PER_TEAM - 1))
    return max(0, NUM_BANS_PER_TEAM - revealed)


def _survival(threat: Dict[int, float], pool: Sequence[int], n_bans: int) -> Dict[int, float]:
    """The odds each brawler is still on the board once the unseen enemy bans land.

    Their ban logic isn't observable, so assume it tracks the same standalone threat we can see,
    softmax it, and scale so the pool's odds sum to the bans they have left. Floored rather than
    driven to zero: this is an estimate of a decision we cannot see, and a brawler we're sure
    they'll ban is exactly the brawler we'd most regret being wrong about."""
    if n_bans <= 0:
        return {b: 1.0 for b in pool}
    weights = _softmax([threat[b] for b in pool], DRAFT_TEMP_SD)
    return {b: max(1.0 - n_bans * w, MIN_SURVIVAL) for b, w in zip(pool, weights)}


def _value_pool(state: DraftState, rows: List[BanScore], model, roster) -> None:
    """Fill in ``ban_value`` / ``replacement`` / ``self_deny`` on the top slice of ``rows``
    (already sorted by standalone threat). Mutates in place."""
    pool = rows[:POOL_N]
    by_id = {r.brawler_id: r for r in pool}
    draftability = _draftability(state, pool, model)
    ladder = sorted(by_id, key=lambda b: draftability[b], reverse=True)
    sides = _sides(state, draftability, roster)

    # Three enemy bans are still coming and are never revealed in time, so nobody drafts from the
    # pool as it stands. Folding those odds into how likely a side is to hold a brawler prices
    # both halves of that at once: their comps get built from the board that will actually exist,
    # and a brawler they were always going to ban carries almost no weight for either of us — so
    # spending our ban on it comes out worthless without a separate rule saying so.
    survive = _survival({r.brawler_id: r.threat for r in rows}, ladder, _hidden_enemy_bans(state))

    # One batch of comps, priced twice: once weighted by the odds they end up holding each
    # brawler, once by the odds we do. What a ban takes from them is the case for it; what it
    # takes from us is the case against.
    build = ladder[:TEAM_N]
    comps = list(combinations(build, TEAM_SIZE))
    strength = _prob_many(model, comps, state.map_id, state.mode)
    theirs, instead = _denial(comps, strength, {b: sides[b][1] * survive[b] for b in build}, build)
    ours, _ = _denial(comps, strength, {b: sides[b][0] * survive[b] for b in build}, build)

    for i, x in enumerate(ladder):
        row, deny_them, deny_us = by_id[x], theirs.get(x, 0.0), ours.get(x, 0.0)
        # Flagged only on a clear margin. With first pick we hold the early slot, so we're
        # marginally likelier to end up with *any* strong brawler — flagging on a hair's
        # difference would stamp the whole top tier and say nothing.
        row.self_deny = deny_us > SELF_DENY_RATIO * deny_them
        # Our own loss weighs less than theirs: we choose the ban knowing our plan and can draft
        # around the hole it leaves, while they meet it cold — and the roster tells us what we
        # can field while theirs stays hidden. Weighted equally, the two sides would cancel to
        # noise, since both draft the same pool.
        row.ban_value = deny_them - SELF_COST_W * deny_us
        alt = instead.get(x) or next((b for b in ladder[i + 1:]), None)
        row.replacement = _name_map().get(alt, str(alt)) if alt is not None else None


def recommend(state: DraftState, stats, model=None, top: int = 6, roster=None) -> List[BanScore]:
    """Rank bans by projected swing in our win probability, falling back to standalone threat
    when there's no model to project with.

    The projection needs a partial-draft artifact: every quantity here is a team or a lone
    brawler read against an *unknown* board, which is exactly what the mask row scores and what
    a pre-mask export can't express at all. Legacy artifacts therefore keep the old threat
    ordering rather than a half-built version of this one."""
    used = state.picked_or_banned()
    rows: List[BanScore] = []
    for b in R.pickable_brawlers():  # released-only: never offer an unreleased entry as a ban
        if b.id in used:
            continue
        rate = stats.brawler_rate(b.id, state.map_id)
        use = stats.use_rate(b.id, state.map_id)
        rows.append(BanScore(b.id, _name_map().get(b.id, str(b.id)), b.cls,
                             _threat(rate.winrate, use), rate.winrate, use, rate.confidence))
    rows.sort(key=lambda r: r.threat, reverse=True)

    if (model is None or not getattr(model, "available", False)
            or not getattr(model, "supports_partial", False) or not rows):
        return rows[:top]

    _value_pool(state, rows, model, roster)

    def rank(r: BanScore) -> tuple:
        # Three tiers. A brawler the draft hands to *us* is advice about our pick, not our ban —
        # denying it is worth something on paper (it is the strongest thing on the map, which is
        # why it can still top the raw numbers), but "ban the brawler you are about to first-pick"
        # is not a recommendation, so it sits below every ban aimed at their side. Unprojected
        # rows sit below both: a ban we know to be weak still beats one we never examined.
        tier = 0 if r.ban_value is None else (1 if r.self_deny else 2)
        return (tier, r.ban_value if r.ban_value is not None else -1.0, r.threat)

    rows.sort(key=rank, reverse=True)
    shown = rows[:top]
    # Keep the brawler we're projected to take on the board, even though it ranks below every
    # real ban. On a map with an obvious top threat, dropping it silently reads as the list
    # having missed it; carrying it flagged answers the question it raises — you take this, you
    # don't ban it — which is the whole point of tracking who holds what.
    flagged = next((r for r in rows if r.self_deny and r.ban_value is not None), None)
    if flagged is not None and top > 1 and all(r.brawler_id != flagged.brawler_id for r in shown):
        shown[-1] = flagged
    return shown
