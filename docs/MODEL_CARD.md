# Model Card — Brawl Stars Win-Probability Model

A compact neural model that predicts the probability that team A beats team B in a Brawl
Stars *Ranked* 3v3 match, given the map and **any known subset of each team's brawlers** —
from an empty board to a completed 3v3. It is the core signal behind the draft assistant's
pick/ban recommendations, and it scores half-drafted boards natively: unknown slots are a
trained input, not a gap to fill.

> **See also:** [model-evaluation.md](model-evaluation.md) — how this model is weighted
> against the empirical signals in the pick blend, and an ablation testing whether that
> weighting should depend on the map/mode (it shouldn't).

## Intended use

- **Primary:** rank candidate picks by their marginal effect on win-probability during a
  live ranked draft, scoring the board exactly as it stands (partial teams included).
- **Not intended for:** predicting individual match outcomes with high confidence, betting, or
  any use that assumes the draft alone determines the result (it does not — see Limitations).

## Data

- **Source:** the official [Brawl Stars API](https://developer.brawlstars.com). It is
  *player-centric* — you fetch a known player's ~25 most recent battles; there is no global
  match feed, and the draft's **ban phase is not exposed** (only the final picked teams).
- **Collection:** a snowball crawler seeds top players from country/global leaderboards, then
  harvests the other five player tags from every ranked match to expand the frontier. Matches
  are deduped by a stable key (`battleTime` + sorted player tags), since one match appears in
  up to six players' logs.
- **Size:** ~1.06M labeled unique ranked matches (1,059,778 at the current retrain). Each row
  is `(map, mode, team A brawlers[3], team B brawlers[3]) → winner`. Per-brawler power level,
  Ranked tier (the API's `trophies` field), and the queue type (`soloRanked`/`teamRanked`) are
  also stored, but they are **not** model features — the tier drives bracket-stratified
  empirical stats, and power is retained for analysis only (see *Limitations*).
- **Population & bias:** the crawler *seeds* from leaderboards but expands via battle-log tags,
  which diffuses down-ladder — so the collected set is broad ranked play, not top-ladder play.
  Measured over a 200,000-match sample, the per-match bracket (median player tier) is Diamond
  44.3%, Mythic 21.0%, Gold 15.1%, Legendary 13.4%, Masters 4.0%, Silver 2.1%, Bronze 0.1% —
  about **61% of matches are Diamond-or-below**, and Pro did not appear at all. Team-A win-rate
  is ~0.51 (no material positional label bias). The map pool is whatever has been in ranked
  rotation while collecting (35 distinct maps in the current set).

## Inputs / features

- `team_a`, `team_b`: up to three brawler indices each (contiguous embedding index over the
  full catalog). Slots not yet drafted are filled with a dedicated **mask row** — a learned
  "unknown slot" embedding sitting one index past the real brawlers (`mask_row` in the
  exported config). Artifacts without a mask row (legacy exports) accept full 3v3s only.
- `map`: index over the ranked-map pool (+ an unknown bucket).
- `mode`: one of the six current ranked modes (+ unknown).

Brawler classes (for role-fit/warnings) come from Brawlify, with a maintained override table
for the newest brawlers that Brawlify has not yet class-tagged.

## Architecture

A small PyTorch network (embedding dims 32 / 16 / 8; ~tens of thousands of parameters) with an
**antisymmetric** head. Let $E_b \in \mathbb{R}^{32}$ be a learned brawler embedding, and let the
map/mode **context** be the concatenation

$$
c = [\,e_{\text{map}},\ e_{\text{mode}}\,] \in \mathbb{R}^{16+8}.
$$

A single **strength** network scores either team in context — it is order-invariant because it reads
the *mean* brawler embedding — and a pair of low-rank embeddings $p_b, q_b \in \mathbb{R}^{16}$ act as
per-brawler "attacker" and "defender" vectors:

$$
S(T, c) = \mathrm{MLP}\!\Big(\big[\, \tfrac{1}{|T|}\sum_{b \in T} E_b,\ \ c \,\big]\Big),
\qquad
P_T = \sum_{b \in T} p_b,
\qquad
Q_T = \sum_{b \in T} q_b.
$$

The logit that team $A$ beats team $B$ adds a strength difference to a bilinear counter term, and the
win probability is its sigmoid:

$$
\ell(A, B \mid c) = \underbrace{\big(S(A, c) - S(B, c)\big)}_{\text{team strength}}
\;+\; \underbrace{\big(P_A \cdot Q_B - P_B \cdot Q_A\big)}_{\text{directed counters}},
\qquad
P(A \text{ wins} \mid c) = \sigma\!\big(\ell(A, B \mid c)\big).
$$

**Why antisymmetric.** Every term changes sign under the swap $A \leftrightarrow B$, so
$\ell(B, A \mid c) = -\,\ell(A, B \mid c)$. Since $\sigma(-z) = 1 - \sigma(z)$,

$$
P(A \text{ wins} \mid c) + P(B \text{ wins} \mid c) = \sigma(z) + \sigma(-z) = 1
$$

holds *by construction*. This bakes in a correct inductive bias (no team-order bias, no global offset
to learn) and makes team-swap data augmentation unnecessary. The bilinear pairing
$P_A \cdot Q_B - P_B \cdot Q_A$ encodes **directed** matchups (X beats Y) that an additive strength
model cannot express, while keeping the whole head sign-flipping.

**Unknown slots.** Every brawler-indexed matrix ($E$, $p$, $q$) carries one extra learned row,
$e_{\varnothing}$ / $p_{\varnothing}$ / $q_{\varnothing}$, used for undrafted slots. Antisymmetry
survives masking: mask-vs-mask counter contributions cancel exactly in the difference (the
$(3-k_A)(3-k_B)\,(p_{\varnothing} \cdot q_{\varnothing})$ terms appear identically on both sides),
so an all-unknown board predicts exactly $0.5$. On full comps no mask row is present, so these
parameters are inert at inference — though training on masked states does shape the *shared*
embeddings and MLP, which is why retrains are gated by a paired full-comp comparison (below).

## Training

The objective is **recency-weighted binary cross-entropy** on the win label $y_i \in \{0, 1\}$ ($1$
when team $A$ won). Each match is down-weighted by an exponential time-decay so the fit leans toward
the live meta across balance patches:

$$
\mathcal{L}(\theta) = -\sum_i w_i \big[\, y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i) \,\big],
\qquad
\hat p_i = \sigma\!\big(\ell(A_i, B_i \mid c_i)\big),
\qquad
w_i = 2^{-(t_{\max} - t_i)/\tau},
$$

with a configurable half-life $\tau$ (default $\approx 30$ days, so a game from a month ago carries
about half the weight of a fresh one; the weights are normalized to mean $1$).

- **Masked drafts.** Each epoch, every match is re-masked: with probability `--p-full` it keeps
  its full 3v3; otherwise a draft state $(k_A, k_B)$ of known picks is drawn uniformly from the
  14 non-trivial states in $\{0..3\}^2$ (excluding the full board and the zero-gradient empty
  board) and a uniformly random subset of each team beyond $k$ known picks is replaced by the
  mask row. Fresh masks per epoch act as free augmentation. The label is unchanged — the model
  learns $\Pr(\text{win} \mid \text{these picks are on the final teams})$.
- **Recency weighting** uses the same exponential time-decay as the empirical stats table, so the
  model and the stats both lean on recent matches; pass a non-positive half-life to disable it
  (uniform weights) for backtests.
- **Split:** random 85/15 train/val (seeded). Optimizer AdamW, weight decay 1e-4, early
  stopping on a fixed masked copy of the val split (same mixture as training, so full-comp
  regressions still move it); headline metrics are reported unmasked for comparability.
- **Baselines:** (a) constant 0.5, (b) logistic regression on signed brawler-presence features,
  (c) **paired**: the previous checkpoint evaluated on the same val rows — the only comparison
  free of data drift, and the no-regression gate for every retrain. The 1v0 state is also
  checked against a shrunk brawler-map winrate marginal (the cheapest single-pick predictor);
  the net must at least match it or the masking design is washing out low-information states.

## Evaluation

Held-out validation (158,966 of 1,059,778 matches), full comps:

| Model | Log-loss ↓ | Accuracy ↑ | AUC ↑ | ECE ↓ |
| --- | --- | --- | --- | --- |
| Always 0.5 | 0.6931 | 0.500 | – | – |
| Logistic regression | 0.6852 | 0.550 | 0.570 | – |
| Previous (unmasked) checkpoint, same val rows | 0.6671 | 0.588 | 0.626 | 0.009 |
| **Embedding net (masked, `--p-full 0.7`)** | **0.6674** | **0.588** | **0.625** | **0.009** |

- Beats both baselines; the paired full-comp delta vs the unmasked control is
  +0.0003 log-loss / −0.0009 AUC — the price of partial-draft support, held to parity by the
  70/30 full/masked training mixture (a 50/50 mixture cost +0.0010 with no partial-state
  gain). Retrains enforce this as a hard gate: `train.py --max-full-delta` (default 0.002)
  aborts without writing artifacts, so the unattended auto-retrain path can't publish or
  baseline a regressed model.
- **The gate can also lock the model in, and does so silently.** On 2026-08-20 it was found to
  have refused **38 consecutive** retrains (deltas +0.0022..+0.0031), freezing the served model
  at 2026-08-12 while `meta_report.json` kept reporting a shifted meta. Two things made it
  invisible: the only trace was a line in `crawl.out.log`, and the crawler had been IP-blocked
  for part of that window, so retrains were refitting a dataset that never grew. `collect.py`
  now counts consecutive failures across restarts and files a `model-stale` issue after three.
  Before changing the threshold, run `backend/scripts/gate_experiment.py --wait-for-data`: it
  waits for the crawl to recover, then sweeps `--p-full` x seeds with the gate disabled and
  separates the three explanations — a frozen dataset, a threshold below the run-to-run noise
  floor (production trains on one seed, so a wide spread is a coin flip every cycle), or a real
  ratchet against an incumbent that was a lucky draw. Every trial is scored against the same
  snapshotted checkpoint and the incumbent is restored afterwards, so the experiment can't
  replace the served model.
- **Partial draft states** (whole val split masked to each state; `mean |p−0.5|` is the
  average edge the model claims):

  | State (known ours v theirs) | Log-loss ↓ | AUC ↑ | ECE ↓ | mean \|p−0.5\| |
  | --- | --- | --- | --- | --- |
  | 1v0 | 0.6908 | 0.538 | 0.010 | 0.029 |
  | 1v1 | 0.6870 | 0.561 | 0.010 | 0.044 |
  | 2v1 | 0.6839 | 0.576 | 0.010 | 0.058 |
  | 2v2 | 0.6781 | 0.595 | 0.010 | 0.070 |
  | 3v2 | 0.6742 | 0.608 | 0.013 | 0.082 |
  | 3v3 | 0.6674 | 0.625 | 0.009 | 0.093 |

  Information about the rest of the draft is worth a steady log-loss improvement at every
  step, and calibration holds near 0.01 across all states. At the lowest-information state
  (1v0) the net sits at parity with a shrunk brawler-map winrate marginal (0.6908 vs 0.6905) —
  it adds nothing beyond the raw statistic there, which the blend already carries as the
  `map` signal, but it is not washed out either.
- **Calibration is the headline:** ECE ≈ 0.009 means the predicted probabilities are
  trustworthy — when it says 60%, the team wins ~60% of the time. For an assistant that
  *consumes* probabilities, calibration matters more than raw accuracy.
- Charts: see [`docs/training.png`](training.png) (validation curves for the masked mixture
  and full comps + reliability diagram).

## Limitations

- **Skill-dominated outcomes.** At top ladder both teams draft well; the *draft* explains only
  a slice of the result, capping achievable AUC (~0.62 here on ~1M matches). The tool therefore
  uses the model for *relative* pick ranking and fuses it with lower-variance empirical
  signals — it does not present any single absolute win-probability as gospel.
- **Partial-draft probabilities are population averages, not adversarial worst cases.** The
  number for an unfinished board marginalizes over how real opponents actually continued such
  drafts. It does not simulate a specific opponent finding the sharpest available counter, so
  against an opponent stronger than the data average, a pick with a rare-but-devastating answer
  is overvalued. (The retired minimax search had adversarial *semantics*, but its five-candidate
  heuristic pruning and uncalibrated min/max over noisy leaf estimates never made that a
  reliably working feature.)
- **Seat-blind.** The battle log records final teams, never pick order, so the model cannot
  condition on who picks next: the same board scores identically whether you or the enemy holds
  the next pick. Relatedly, training masks a *random subset* of the final team, while at
  inference the known picks are the *early* picks — early picks are not a perfectly random
  sample of final teams, a small conditioning mismatch that cannot be measured without pick
  order.
- **No opponent-reaction modeling.** Nothing in the live recommend path predicts "if I pick X,
  the opponent becomes more likely to answer with Y" — there is no dedicated per-brawler
  reaction/psychology component. `score_candidate`'s `counter` term (`engine/scoring.py`) only
  reads *already-drafted* enemies against the candidate; it says nothing about who gets picked
  next. Because training masks random slots of a *finished* team and marginalizes over how real
  historical drafts ended up (see *Partial-draft probabilities*, above), the net's win-prob
  surface can implicitly carry real co-occurrence patterns from the data — e.g. if Edgar
  disproportionately ends up on the opposing team in matches where Tick was drafted, that shows
  up as a population-level statistical association, not as a live "opponents draft Y in reaction
  to X" simulation, and it can't distinguish a genuine strategic counter from incidental
  meta co-occurrence. The retired minimax search (above) is the closest this project got to real
  opponent-response modeling, and it was removed as unreliable. The only place the tool predicts
  what an opponent is *likely to pick* at all is the ban engine's `_draftability`
  (`backend/bsdraft/engine/bans.py`) — a blend of the model's read of a brawler with its
  map/mode use-rate — and that estimate is unconditional on your own picks, not a reaction to
  them.
- **No ban data.** The API never exposes bans, so the model is trained on final picks; ban
  value is inferred separately from win-rate + contest rate.
- **Population shift.** The net is trained on the pooled crawl, whose mass sits in the middle
  brackets (~61% Diamond-or-below; see *Data*), not on top ladder — so a single un-conditioned
  net is fit to a mixture rather than to any one bracket, and it fits neither tail well. Premade
  coordination also differs from solo queue. Rank-bracket stat tables mitigate this on the
  empirical side; the net itself is not bracket-conditioned.
- **Power level and loadout are not model features.** The net sees brawler identity, map, and
  mode — nothing about how built a brawler is. Power level *is* collected per player-slot
  (`backend/bsdraft/collect/match.py`) but is dropped during feature construction
  (`backend/bsdraft/data/dataset.py`), and the equipped star power / gadget / gears /
  hypercharge are **never recorded in battle logs at all** — which is why per-item win rates
  have to be inferred indirectly (see [`item-winrate.md`](item-winrate.md)). Consequence: an
  under-levelled or unhypercharged brawler scores identically to a fully-built Power 11 one.
  In practice this is closer to a definition than a defect — in a 200,000-match sample
  (1.2M player-slots) Power 11 is 97.2%, Power 10 1.6% and Power 9 1.2% — so the predicted
  win% should be read as *near-max-power play*, and it will overstate a brawler you have not
  finished building. The same holds for the mastery signal, which scores owned loadout and
  comfort but deliberately not power or hypercharge (`backend/bsdraft/engine/mastery.py`).
- **Meta drift.** Brawler strength changes with patches; the model needs periodic retraining on
  fresh data (recency weighting mitigates but does not eliminate this).

## Ethical considerations

Uses only publicly available, game-provided match data via the official API. No personal data
beyond public player tags/handles is stored. Not affiliated with Supercell.

## How the engine uses it

The model is one component of a transparent fused score, combined with empirical signals from the
collected matches. Every raw win-rate is **Bayesian-shrunk** toward a prior $\pi$ (0.5 globally, or
the global rate for a per-bracket table) with a pseudo-count $\kappa = 20$, which also defines the
confidence shown in the UI:

$$
\widehat{w} = \frac{\text{wins} + \kappa\,\pi}{\text{games} + \kappa},
\qquad
\mathrm{conf} = \frac{\text{games}}{\text{games} + \kappa},
$$

where "wins" and "games" are recency-weighted counts $\sum_i w_i$ (the same exponential time-decay
used to train the model, here with a ~3-week half-life). The
map win-rate, synergy, counter, role-fit, model, and optional mastery signals are then fused into a
renormalized weighted average over whichever signals $\mathcal{A}$ are *active* at the current draft
state:

$$
\mathrm{score}(b) = \frac{\sum_{k \in \mathcal{A}} \omega_k\, v_k(b)}{\sum_{k \in \mathcal{A}} \omega_k}.
$$

That fused score drives the displayed pick ranking. The model signal itself
(`scoring.model_marginal`) is the net's read of the board as it stands with the candidate added —
partial teams passed directly when the artifact supports them (`supports_partial`), with a legacy
fallback that completes both teams with the map's top empirical picks for old artifacts. Every
component is surfaced in the UI, so recommendations are explainable rather than a black box.
