# Readiness: what an under-leveled brawler actually costs

The served win rates are Power 11 win rates. 97.3% of collected player-slots are Power 11, so
`map_winrate` and the model's marginal both describe a maxed brawler with a full loadout. Scoring
a Power 9 brawler off that table is unmeasured extrapolation.

`engine/mastery.py` used to assert the opposite — *"the dataset's win rates already fold real power
in"* — and excluded power from the mastery score on that basis. This document is the measurement
that retired that claim.

**Owner:** `backend/bsdraft/data/readiness_build.py` (home-only, numpy) →
`backend/scripts/export_readiness.py` → `data/reference/readiness.json` (checked in, stdlib-loadable).

```bash
PYTHONPATH=backend python backend/scripts/export_readiness.py
```

Two full passes over `data/raw/matches.jsonl`; multi-minute. Nothing on the serve path imports the
build module — the serve side reads only the small JSON artifact.

---

## Why the obvious estimator is wrong

Three estimators, on the same corpus, in increasing order of trustworthiness:

| Estimator | Power 9 result | What's wrong with it |
| --- | --- | --- |
| Naive cross-player | ≈ −16 pp | Players who bring a Power 9 brawler into a Diamond lobby are *worse players*. Most of this is skill, not power. |
| Within-player | −8.1 pp | Better — the player is their own control — but brawler choice still rides along: which brawlers you happen to leave under-leveled is not random. |
| **Within-player, residualized** | **−9.4 pp** | Ships. Outcomes are residualized against an all-Power-11 population table before contrasting, so brawler choice and bracket are removed first. |

The gap between the first row and the last is the entire reason this needed a real design. Anyone
re-deriving this number and getting −16 has measured matchmaking, not power.

Note the residualized estimate is *larger* than the un-residualized one, not smaller. Players bring
their under-leveled brawlers out on better-than-average picks, which had been masking part of the
deficit — the confound was working in the flattering direction.

## The design

**Unit of observation.** One player-slot in one decided ranked 3v3.

**Clean lobbies only.** An appearance counts only when all five *other* players are Power 11, so the
sole power anomaly on the board is the slot being measured. Two under-leveled players in one lobby
would otherwise contaminate each other's contrast.

**Strata = (player tag × Ranked bracket).** The player is their own control: their under-11
appearances are contrasted against *their own* Power 11 appearances. Skill cancels within the
stratum. Bracket enters the key because a player crosses brackets within a season.

**Residualized outcome.** Each appearance contributes `won − p_pop(brawler, bracket)`, where
`p_pop` is built in a separate first pass **from all-Power-11 lobbies only** (smoothed with 50
pseudo-games, backing off to the brawler's all-bracket rate, then 0.5). Building the baseline at
full power is deliberate: letting sub-11 games into it would drag the reference toward the effect
being estimated.

**Per-player cap.** A player contributes at most 30 appearances to any one (stratum, arm), so a
grinder cannot dominate a pool.

**Combine.** Mantel-Haenszel over strata present in *both* arms, weight `n_t·n_c/(n_t+n_c)`.

**Standard error.** A robust across-stratum weighted-mean SE, not a binomial formula — the outcomes
are residuals in [−1, 1] rather than 0/1, and both the residualization and the cap break the
binomial assumption. The across-stratum spread absorbs all of it.

## The placebo is a gate, not a footnote

Relabel a random half of all-Power-11 clean appearances as "treated" and run the identical
pipeline. Everything is held fixed except the thing being tested, so a design that manufactures an
effect shows it here.

`export_readiness.py` **refuses to write the artifact** when `|placebo RD| > 0.005`, when the
deficit is not monotone in power, or when Power 9 misses the strata / |z| bars. `--force` writes
anyway and stamps `"forced": true` into the artifact so a reader can see the gate was overridden.

The per-level `shippable` bar (≥200 paired strata **and** |z| ≥ 3) does real work, and the sub-9
bucket is the standing demonstration: it reports an absurd risk difference — around −100 points, on
*two* strata — because a residualized difference of means lives in [−2, 2] and two observations
constrain nothing. It is correctly never shipped, and Power ≤ 8 inherits Power 9's constant
instead. Treat a wild `le8` line in the report as the gate working, not as a finding.

## The shipping rule

The constant shipped is not the point estimate. It is the magnitude shaved by a safety margin and
floored to a **0.5-point grid** (`_ship_value`).

The margin is `max(2 · se, methodology_margin)` — whichever uncertainty dominates governs. On this
corpus that is emphatically **not** the sampling error. Four specifications of this estimator have
returned:

| Specification | Power 9 | Power 10 | Placebo | Placebo strata |
| --- | --- | --- | --- | --- |
| Replication A | −7.5 | −3.3 | −0.10 | — |
| Replication B | −9.9 | −6.2 | −0.01 | 8,284 |
| Within-player, no residualization | −8.1 | −4.0 | +0.07 | 897,188 |
| **Ships** (residualized) | **−9.4** | **−5.9** | **+0.07** | 993,149 |

All four agree on sign, ordering and magnitude class, and every placebo is clean. The ~2.4-point
spread is driven by stratum, cap and residualization choices — it is **methodological**, not
sampling. A single run's SE is ~0.45 points, so keying the haircut to the SE alone would absorb
about a fifth of the uncertainty it claims to cover. `methodology_margin` (1.5 points) names the
real quantity in the units it lives in; the SE term takes over only when a future rebuild is thin
enough for sampling error to dominate.

Applied to the shipping run: Power 9 is `9.4 − 1.5 = 7.9` → **0.075**, Power 10 is
`5.9 − 1.5 = 4.4` → **0.040**.

Direction of the error is deliberate: under-claiming a real effect is recoverable, because the
number rises the next time this runs. Over-claiming re-creates the over-personalization that the
2026-08-17 de-weighting was fighting.

## Artifact schema

```jsonc
{
  "schema": 1,
  "meta": {
    "built_ts": 1755749000,
    "newest_match_ts": 1755739000,   // stale rebuild detector — see below
    "matches": 1425659,
    "clean_appearances": 7626490,
    "power_hist": {"11": 8272654, "10": 123500, "9": 106250},   // 97.30 / 1.45 / 1.25 %
    "params": { /* DEFAULTS, so retuning needs no code change */ }
  },
  "placebo": {"rd": 0.0007, "se": 0.0004, "z": 1.68, "tol": 0.005, "ok": true},
  "levels": {
    "9":  {"rd": -0.0944, "se": 0.0045, "z": -21.07, "n_strata": 12526,
           "ship": 0.075, "shippable": true},
    "10": { /* rd -0.0587, ship 0.040 */ },
    "le8": { /* reported, never shipped as its own constant */ }
  },
  "monotone": true,
  "power_deficit": {"11": 0.0, "10": 0.040, "9": 0.075, "8": 0.075, /* ...1..8 */},
  "ok": true
}
```

`power_deficit` is the only key the serve path needs: **magnitude in win-rate points, subtracted**
from the base score. Power ≤ 8 inherits Power 9's constant — it is 0.005% of slots and below the
Ranked floor in every bracket, so it is unfieldable rather than merely weak.

### `newest_match_ts` is the stale-rebuild detector

The crawler has a documented silent-stall mode: an IP-lock 403 logs "+5000 matches" while
`matches.jsonl` is frozen. A rebuild against a stalled crawler would look completely healthy —
same match count, same estimates, passing placebo. Compare `newest_match_ts` against wall-clock
before trusting a re-derived constant.

## How the constant reaches a score

`engine/readiness.py` is the pure-stdlib serve-side reader. It never imports the numpy estimator
above; it loads `readiness.json` if present and falls back to the code defaults if not, so a
missing artifact degrades rather than erroring a request.

The fusion in `engine/scoring.py` is now two stages instead of one:

$$\text{base} = \frac{\sum_{k \in A} w_k v_k}{\sum_{k \in A} w_k}, \qquad
\text{score} = \operatorname{clamp}_{[0,1]}\big(\text{base} - \Delta_{\text{ready}} + \Delta_{\text{items}} + \Delta_{\text{hist}}\big)$$

with $A$ the five ablation-tuned objective signals. `base_score` therefore does not depend on
whether a roster is loaded, which is what makes the two blind-pick columns comparable — previously
the personalized read renormalized over a larger denominator and printed a higher number for the
same board.

| Adjustment | Cap | Provenance |
| --- | --- | --- |
| `readiness` | 0.12 | **measured** (power) + **estimated** (loadout) + **unpriced** (hypercharge) |
| `item_edge` | 0.05 | measured — inert, the table does not exist |
| `history_edge` | 0.02 | unvalidated product knob |

Two sizing rules hold the labels honest, both pinned by tests:

- **No declared prior outranks a measurement.** The largest loadout prior (a missing star power,
  0.021) sits below the smallest non-zero measured deficit (Power 10, 0.040).
- **The unvalidated knob stays well under the measured one.** `HISTORY_CAP` is 0.02 against a
  Power 9 deficit of 0.075. A personal win rate is confounded with when you played, who you queued
  with, and meta drift since; a power level is not.

### Two traps worth knowing about

**Gear slots are power-gated.** Ranked opens the two gear slots at Power 8 and Power 10, so a
Power 9 copy has exactly one. Charging it for an empty second slot would bill the same shortfall
twice — once as power, once as a missing gear. `Fielded.gear_slots` charges only for slots the
power level has actually unlocked.

**The player's overall win rate is a header fact, not a per-pick adjustment.** A difference-in-
differences that nets a player's global rate out of each brawler's edge looks principled and is
useless: the term is a per-player constant, so it shifts every candidate identically and cannot
reorder anything. All it does is move the level — on a 40%-overall account it silently added ~5.5
points to every row. `PersonalStats.overall_rate()` exists for display; scoring deliberately does
not call it.

## What this does not measure

**Hypercharge.** Battle logs carry no hypercharge field — an appearance is
`{tag, brawler_id, brawler_name, power, trophies}` — so no estimator exists on data in hand. It
ships displayed-and-unpriced. `collect/profiles.py` now records `hc` (and `ht`) per brawler
because ownership is only observable *live*: a hypercharge contrast can never be reconstructed
from the match log retroactively, so every un-profiled day is unrecoverable.

**Per-map or per-mode slices.** The strata key has no map component and the effect is estimated
globally. Power is a stat multiplier, not a matchup property, so a global constant is the right
shape; a per-map slice would also thin the already-thin sub-11 tail past usefulness.

**Item ownership.** Handled separately — see [item-winrate.md](item-winrate.md). Note that table
has never been built (no `profiles.jsonl`, no `ITEMSTATS_URL`), so it currently contributes nothing
anywhere.

## Related

- [MODEL_CARD.md](MODEL_CARD.md) — the model conditions on brawler ids, map and mode only; power is
  collected and then dropped in feature construction.
- [model-evaluation.md](model-evaluation.md) — the held-out ablation behind the five objective
  weights. Readiness sits *outside* that blend as an additive post-adjustment and makes no accuracy
  claim about those five.
- [item-winrate.md](item-winrate.md) — the other half of "how built is this brawler".
