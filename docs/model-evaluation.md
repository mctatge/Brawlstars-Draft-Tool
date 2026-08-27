# Model Evaluation — How the draft signals are weighted, and whether that should change

The recommender fuses six signals into one pick score: a brawler's **map** win-rate, the
learned **model** (win-prob net), pairwise **synergy** with allies, **counter** vs. revealed
enemies, mode-based **role** fit, and player-specific **mastery**/**personal** history. The
first four are combined with fixed global weights ([`DEFAULT_WEIGHTS`](../backend/bsdraft/engine/scoring.py)).

**Personalization weights are product judgment, not ablation output.** `mastery` and `personal`
apply only on a loaded roster seat; the ablation below never scores them. On **2026-08-17** they
were cut (`mastery .25 → .10`, `personal .20 → .08`): at the old values the two "how good is
*this* player on it" terms were ~31% of a personalized pick's score (~37% before any enemy is
drafted), out-driving the model and outweighing `counter` — so the board over-ranked
already-mastered brawlers and buried meta picks the player could improve into. They're now a
nudge (~15% combined), not a driver. No accuracy claim rides on them; tune freely — a per-seat
*meta / my-roster* toggle is the planned next step. Guarded by `backend/tests/test_scoring.py`.

This doc records the held-out ablation that tunes those weights. It has been run twice —
June 2026 on ~40k matches, and **August 2026 on ~995k matches** — and the headline finding
**reversed** between runs as the dataset grew 25×.

> TL;DR (2026-08-10, n = 995,135) — Draft is a small but real edge (AUC ~0.59–0.65 by mode;
> matchmaking equalizes teams). **The retrained net now out-discriminates every empirical
> signal** (AUC .625 vs .608 for the blend) — a full reversal of the June result, where the
> empirical side out-ranked a weaker net and earned ~69% of the stacker weight (it now earns
> ~22%). **Context-dependent (per-map/mode) weighting still does not help.** Applied
> rebalance: `model .20 → .40`, funded by `map .32 → .25`, `synergy .15 → .05` (its
> conditional coefficient is ~0 in every mode), `counter .23 → .20`.

## Method (leakage-free by construction)

Three scripts, run over the labeled Ranked matches:

- [`scripts/ablate_components.py`](../backend/scripts/ablate_components.py) — net vs. empirical signals, head-to-head.
- [`scripts/ablate_context.py`](../backend/scripts/ablate_context.py) — does the right weighting move by mode?
- [`scripts/sweep_blend.py`](../backend/scripts/sweep_blend.py) — concrete full-blend candidates (model + trio), scored as shippable fixed weight sets.

Design choices that keep the comparison honest:

1. **The net is re-trained on the train split inside the harness.** The shipped model was
   trained on data that has since grown, so reusing its holdout would leak. We retrain with
   the same recipe (incl. early stopping) so the net is calibrated, not overtrained.
2. **Empirical stats are built on the train rows only**, then scored on held-out rows.
3. Each draft signal is expressed as an **antisymmetric team-A advantage** (map/synergy as a
   team-vs-team difference, counter as the directed cross-matchup), so a positive score means
   "team A favored" and swapping teams negates it.
4. Calibration / stacking logistic regressions are fit with **cross-validation on the
   out-of-sample features**, so reported probabilities never see their own label.
5. For the per-mode test, features are **5-fold cross-fit over the full dataset** (each fold's
   stats come from the other folds) so per-mode estimates are stable.

## Result 1 — the net now subsumes most of the empirical signal (reversed from June)

Held-out validation (n = 149,270), all predictors well-calibrated (ECE ≤ 0.003):

| Predictor | log-loss | acc | AUC |
|---|---|---|---|
| always 0.5 | 0.6931 | .500 | — |
| net only | 0.6673 | .588 | **.625** |
| empirical blend only | 0.6740 | .576 | **.608** |
| net + empirical | 0.6667 | .589 | **.626** |

Standalone AUC of each raw signal: map `.608`, synergy `.584`, counter `.590`, blend `.608`, net `.625`.

- The **net out-discriminates the empirical blend** (.625 vs .608), and stacking the blend on
  top of it adds almost nothing (+.001 AUC). The embeddings have absorbed what the count
  tables know, plus interactions they can't represent.
- The stacker now assigns **~78% of the weight to the net, ~22% to the empirical side** — the
  exact mirror of June's 31/69. The June table (n = 6,031 val: net .567, blend .581) is kept
  below for the record; the flip tracks the net's training set growing 40k → 995k matches
  (shipped-model AUC .576 → .627 across the same period).

<details>
<summary>June 2026 run (n = 40,208) — the superseded result</summary>

| Predictor | log-loss | acc | AUC |
|---|---|---|---|
| net only | 0.6861 | .547 | .567 |
| empirical blend only | 0.6831 | .556 | .581 |
| net + empirical | 0.6826 | .560 | .583 |

Stacker: ~69% empirical / ~31% net. Standalone: map .568, synergy .564, counter .570.
</details>

## Result 2 — context-dependent weighting still does not help

Cross-fit over all 995,135 matches, per mode. `map/syn/cnt` are the standardized weights each
empirical signal earns; the AUC columns compare **fixed** (shipped weights), **global-refit**
(one logistic over all rows), and **mode-refit** (a logistic refit within the mode):

| Mode | n | map | syn | cnt | AUC fixed | global-refit | mode-refit |
|---|---|---|---|---|---|---|---|
| Gem Grab | 157,653 | +.24 | −.03 | +.19 | .600 | .602 | .602 |
| Brawl Ball | 159,981 | +.26 | −.10 | +.23 | .595 | .599 | .600 |
| Knockout | 168,114 | +.27 | −.02 | +.12 | .588 | .592 | .592 |
| Hot Zone | 158,166 | +.41 | −.10 | +.23 | .631 | .636 | .636 |
| Heist | 188,403 | +.51 | −.08 | +.16 | .641 | .648 | .647 |
| Bounty | 162,818 | +.28 | −.03 | +.11 | .587 | .592 | .592 |

- **`mode-refit` ≈ `global-refit` in every mode** (within ±0.001 AUC), on 26× the June sample.
  Context-dependent weighting remains not worth building.
- **Synergy's conditional coefficient is negative in all six modes** (−.02 to −.10): given map
  and counter, the pair-winrate tables add nothing — they are redundant, not informative.
  (Standalone, synergy still ranks at .584 — the redundancy is conditional.)
- What varies is still **how much draft matters at all**: Heist (.65) vs Bounty/Knockout
  (~.59). That's a confidence signal, not a reweighting one.

> **Why no draft-phase test?** Completed matches only contain final 3v3 comps — there are no
> partial-draft labels — so "weight signals differently as picks come in" can't be measured
> from outcomes. The engine already handles phase structurally: synergy/counter only activate
> once allies/enemies exist, and the blend renormalizes over the active signals.

## Result 3 — the applied change: shift weight from the trio to the model

Held-out AUC of **fixed** full-blend candidates (model + trio in one linear mix, prob-unit
values as `score_candidate` blends them; no fitting, so no overfit):

| Full-blend weighting (model/map/syn/cnt) | held-out AUC |
|---|---|
| June-2026 shipped `.20 / .32 / .15 / .23` | 0.6245 |
| pre-June `.20 / .40 / .15 / .15` | 0.6248 |
| **chosen `model-40` `.40 / .25 / .05 / .20`** | **0.6262** |
| `model-50` `.50 / .20 / .05 / .15` | 0.6260 |
| `model-60` `.60 / .16 / .02 / .12` | 0.6257 |
| `model-70` `.70 / .10 / .00 / .10` | 0.6252 |
| net-only `.90 / 0 / 0 / 0` | 0.6247 |
| refit ceiling (train-fit, val-scored) | 0.6267 |

- The curve **plateaus at model .40–.50 and falls off toward net-only** — the trio still earns
  its keep as a complement, just not as the majority partner. Note net-only ≈ the June-2026
  shipped weights: leaving the weights untuned wastes the whole model upgrade.
- `model-40` beat the shipped weights in **200/200 paired bootstrap resamples** and sits within
  .0005 AUC of the linear ceiling.
- The unconstrained refit's raw weights (e.g. a negative counter coefficient) are collinearity
  artifacts — the net has absorbed the counter signal — which is exactly why the decision is
  made on fixed candidates, not the refit.

**Applied change** ([`scoring.py`](../backend/bsdraft/engine/scoring.py)): `model 0.20 → 0.40`,
`map 0.32 → 0.25`, `synergy 0.15 → 0.05`, `counter 0.23 → 0.20` (role, mastery, personal
unchanged).

**Why keep synergy at .05 instead of 0?** Match-level redundancy is not candidate-level
uselessness: mid-draft, with two allies picked and no model-visible enemy comp, the pair
tables are still the only "fits with what we have" signal, and the blend renormalizes over
active signals — a small weight keeps that behavior (and its explanation in the UI) alive at
negligible cost (.0002 AUC vs the best syn-0 candidate).

## Limitations

- **Low ceiling.** Ranked matchmaking equalizes teams (base team-A win-rate 0.510), so no
  weighting scheme pulls far past ~0.63 AUC pooled. The honest claim is "a small, real draft
  edge," not "predicts winners."
- **Match-level ≠ candidate-level.** The reweight direction transfers; exact magnitudes don't.
- The reweight's downstream effect on *pick rankings* can't be validated against outcomes (no
  pick-level labels), so the change stays on the conservative side of the plateau.
- **These weights chase the model's quality.** The 69/31 → 22/78 flip shows the optimal blend
  moves with the net's training set. Re-run the suite after major dataset growth or an
  architecture change; the sweep exists so that check is one command.
- **Masked-model note (2026-08-12).** The shipped net is now trained on masked partial drafts
  (see [MODEL_CARD.md](MODEL_CARD.md)), and the engine feeds it the live board directly instead
  of completing teams with top-meta picks. This suite evaluates all signals at *full comps*
  (match-level labels are all that exist), so its weight calibration point is unchanged, and the
  retrain was gated by a paired full-comp comparison against the previous checkpoint. What did
  change is the model signal's mid-draft behavior: it now reads closer to 0.5 early (honest
  marginal) instead of the wider completion-based spread the weights were originally swept
  against. That can't be validated here for the usual reason (no pick-level labels) — but note
  the ablation scripts train their own *unmasked* nets internally, so their net-vs-empirical
  comparison remains apples-to-apples across reruns.
- **Role is confidence-scaled, not flat (2026-08-18).** `role_fit` is a hand-set mode×class prior
  and is *not* part of the validated ablation above (the sweep tunes only map/model/counter/
  synergy). Because Ranked compresses empirical `map_wr` into a narrow band while role spans a wide
  hand-set range, at its nominal 0.10 weight role was delivering ~29% of the first-pick ranking
  spread — co-equal with map — and, being map-invariant, it flat-topped the same archetypes across
  every map of a mode. Role now shrinks toward neutral by each candidate's own map-data confidence:
  `role_eff = 0.5 + (1 − map_conf)·(role_fit − 0.5)`. On well-sampled maps role nearly drops out
  (Crystal Arcade: role's effective ranking influence fell 29% → 10%, map now primary) so rankings
  track real per-map win-rate; on freshly-rotated / zero-data maps `map_conf → 0` and the full
  archetype prior remains, which is exactly where it earns its keep. The weight and renormalization
  are unchanged — only the role *value* is gated. Not outcome-validated (no pick-level labels), same
  caveat as the rest of the trio.

## Within-team synergy — the class-synergy term (2026-08-27)

**Question that started it:** the recommender kept surfacing redundant picks — a second tank
after a tank, Grom after Sprout (two immobile throwers, both dived by one assassin). Why doesn't
the win-prob net discount that on its own?

**Root cause (architecture).** The net's team strength is `S(team) = MLP(mean(brawler_emb), ctx)`.
Mean-pooling is order-invariant but *collapses composition*: strip the MLP's one ReLU and `S`
reduces to a pure sum of independent per-brawler contributions (`TEAM_SIZE` is fixed, so the mean's
denominator never varies), making a brawler's marginal value independent of its teammates —
redundancy is then *unrepresentable*. The only explicit interaction term, the counter head
`PA·QB − PB·QA`, is exclusively cross-team; there was **no within-team interaction term at all**.

**The fix that shipped.** A learnable symmetric **class×class** synergy matrix `M` (7 archetypes),
entering the logit as `T(A) − T(B)` with `T(team) = Σ_{i<j} M[cls(i), cls(j)]` (see
`models/winprob.py` / `models/serve.py`, `--class-synergy`). Archetype-level rather than
per-brawler on purpose: it pools *every* same-class pairing in the data into one estimate — 28
parameters over ~1.8M matches — where a per-brawler synergy embedding is starved (redundant pairs
co-occur rarely) and swings wildly across retrains. The sign is never assumed; the data decides.

**What the data actually supports — and it's weak.** Isolated from individual brawler strength,
the redundancy signal is *comparable in size to seed noise*:

- **Throwers (Artillery) are the one archetype whose same-class anti-synergy reproduces** across
  seeds (diagonal ≈ −0.03 to −0.07; negative in every run) — the Grom+Sprout intuition, learned.
- **Two tanks is NOT penalized, correctly.** Raw outcomes say stacked tanks *win* (Brawl Ball
  win-rate by tank count: 0→0.48, 1→0.52, 2→0.55, 3→0.59 over 573k matches), and it is **not a
  skill artifact** — `corr(tank_count, tier) = −0.14` (weaker players stack tanks and still win).
  The class term's Tank diagonal sits near 0: tanks win on *individual strength* (the strength
  term), not on stacking synergy — so the model keeps them, which is right.
- Every other archetype's diagonal is noise around zero (flips sign across seeds; full-matrix
  seed-to-seed correlation only ≈ +0.6).

**Two levers that did *not* work** (removed, not shipped):

- **Rank-weighting** the loss toward high-tier games (to learn the high-rank truth that stacked
  throwers stop winning above Mythic) was *counterproductive* in every variant — at high rank the
  few double-thrower comps that appear are played deliberately/well, so the residual anti-synergy
  *shrank*. It also trims the (correct) tank behavior. Confirmed across both the per-brawler and
  class parameterizations.
- **L2-regularizing** the class matrix to isolate "throwers only" *backfired*: shrinking toward 0
  drags the thrower signal (~0.05) down into the noise band (~0.02), where it flips sign across
  seeds — seed correlation went from +0.59 (plain) to −0.08 (regularized). Signal and noise are
  the same size; a magnitude penalty can't separate them.

**Shipping decision.** The plain, all-rank class-synergy term is shipped as a *gentle, honest
nudge*: it reproducibly discounts throwers and leaves everything else ~unchanged. Cost is
**+0.0034 full-comp val logloss / −0.008 AUC** vs. the strength+counter baseline — right at the
~0.0035 seed-noise floor, i.e. roughly accuracy-neutral. The broader lesson: **redundancy
discounting is barely learnable from this (mid-ladder) data** — the effect is real only for
throwers and only weakly; anything stronger or cleaner would need high-rank data (unavailable) or
a deliberate heuristic, not more model tuning.

## Reproduce

```bash
PYTHONPATH=backend .venv/bin/python backend/scripts/ablate_components.py   # -> docs/ablation.json
PYTHONPATH=backend .venv/bin/python backend/scripts/ablate_context.py      # -> docs/ablation_context.json
PYTHONPATH=backend .venv/bin/python backend/scripts/sweep_blend.py         # full-blend candidates
```
