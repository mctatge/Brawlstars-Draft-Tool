# Backend Architecture — layers, data flow, and the design decisions that constrain changes

How `backend/bsdraft` is structured (collect → data → models → engine → api), and the four
cross-cutting design decisions you must not break when editing backend code.

## Data flow

**Data → stats/model → engine → API → board.** A snowball crawler (`collect/`) works around
the player-centric official API: it seeds top players, harvests all 6 tags from each ranked
match, and dedupes by a stable match key (`battleTime` + sorted tags) into `data/raw/`.
`data/dataset.py` builds training rows; `models/winprob.py` trains the embedding net; the
engine fuses the model with empirical stats built at startup from the same matches.

## Layers (`backend/bsdraft/`)

- `collect/` — async client, crawler, match parser, publish.
- `data/` — reference loaders, encoders, dataset builder, runtime release sync in `data/sync.py`.
- `models/` — train (`winprob.py`) + serve (`serve.py`).
- `engine/` — the draft brain. `engine/state.py`'s `DraftState` is the object threaded
  through nearly everything. Core: `engine.py` (the `DraftEngine` facade) / `scoring.py`
  (pick scoring) / `bans.py` (ban valuation) / `stats.py` / `mastery.py` /
  `personal.py`. Also: `playerrank.py` (tier resolution + rank index), `tiers.py`
  (Diamond/Masters bracket labels), `stats_store.py` (loads the precomputed stats artifact),
  `rank_store.py` (loads the precomputed rank-index artifact into a compact NumPy lookup),
  `drift.py` (staleness/liveness), and `composition.py` + `gameplan.py` (team-composition
  reasoning surfaced through the API). `gameplan.py` is two layers: a rule-based strategic plan
  that needs only the `DraftState`, plus a data-backed read (head-to-head grid, per-map form, ally
  pairs, the model's win prob for the finished draft) that takes the bracket `DraftStats` and the
  model from `DraftEngine`. Both halves degrade independently — no stats/model yields exactly the
  rule-based plan, and each data section drops out when its cells fall under its sample floor.
- `api/` — FastAPI app (`api/main.py`).

## Four cross-cutting design decisions

1. **Two model implementations that must stay in sync.** `models/winprob.py` is the PyTorch
   training model; `models/serve.py` reimplements its `forward()` in **pure NumPy**, loading
   the exported `winprob.npz`. The deployed API runs inference with no torch. **If you change
   the model architecture, update both** — the docstring in `serve.py` pins the exact forward
   formula (antisymmetric strength diff + low-rank counter term). `winprob.npz` is tiny
   (~50 KB) and committed; `winprob.pt` is not.

2. **Three dependency tiers.** `requirements.txt` (full: train + collect + serve),
   `requirements-collect.txt` (crawler only), `requirements-serve.txt` (deployed API —
   **no torch/sklearn/pandas**, NumPy serving only, fits Render's 512 MB free tier).
   Adding an import to a serve-path module can break the deploy build.

3. **Dependency-free core layers.** `constants.py` and `data/reference.py` are pure stdlib
   so they run without installing anything (`python -m bsdraft.data.reference`). `config.py`
   (needs `pydantic-settings`) is deliberately *not* imported by the reference layer. Keep
   third-party imports out of these two modules.

4. **Fused, renormalized scoring.** `engine/scoring.py` scores a pick as a weighted average
   over only the **active** signals (synergy needs allies, counters need a revealed enemy,
   mastery/personal need a roster), renormalized by the active weights. `DEFAULT_WEIGHTS`
   were tuned via the held-out ablation (see the comment there and
   [model-evaluation.md](model-evaluation.md)) — context-dependent per-map weighting was
   tested and found no better, so weights are global. The model signal scores the draft
   board as it stands: the net is trained on masked comps, so partial teams are first-class
   inputs (`supports_partial`), with a legacy top-meta completion fallback for old artifacts.

## Bans are valued, not ranked by threat

`engine/bans.py` answers a different question from `scoring.py`. A pick is scored on how good
the brawler is; a ban is scored on **what removing it costs**, which depends on the rest of the
ban set — so the list re-ranks as your teammates ban, and a brawler's own numbers never change.
Three things drive it, none of which a threat table can express:

- **Substitutes absorb a ban.** Denial is measured as the drop in a side's *soft maximum* over
  every comp the pool can still field, weighted by the odds that side ends up holding all three.
  Brawlers covering the same job rarely appear in a strong comp together, so each holds only part
  of the mass and neither is expensive to lose — until one is banned and the survivor inherits it.
- **Bans are global, so who picks first matters.** Draft order gives the odds each brawler lands
  on their side or ours. `ban_value = deny(them) − SELF_COST_W · deny(us)`; a brawler the draft
  hands *us* is advice about our pick, so it drops below every real ban and is flagged
  `self_deny` (still returned, so an obvious threat is never silently missing).
- **Roster.** A brawler we can't field can never be one of our picks, so banning it costs nothing.
- **Their three bans are never visible in time.** You only ever see your two teammates' bans
  before choosing, so all three enemy bans stay priced as unseen for the whole phase (and the
  list can only re-rank twice per draft). Each brawler's holding odds are scaled by the odds it
  survives them: their comps get built from the board that will actually exist, and a brawler
  they were always going to ban carries almost no weight for either side — so spending our ban
  on it comes out worthless without a rule saying so, and the ban they *aren't* making rises.

Two deliberate constraints. **No argmax anywhere** — a projected draft line ("take the best, then
the next best") is a step function whose flipped picks cascade into wholly different boards; it
reads as sophisticated and behaves like a coin toss, so everything aggregates over comps instead.
And it needs a **partial-draft artifact**: every quantity is a comp or a lone brawler read against
an *unknown* board, which is what the mask row scores and a legacy export cannot express at all —
those keep the old threat ordering (`ban_value: null`) rather than a half-built version of this.

**This is the one part of the engine with no ground truth.** Supercell's battle log has no ban
field, so unlike `DEFAULT_WEIGHTS` nothing here is fit to held-out matches. The constants are
priors chosen for behavior on live maps, not tuned parameters — treat them as such when editing.

## The two recommend endpoints are intentionally distinct

`/api/recommend` personalizes to the player's roster + history (mastery, personal win-rate),
while `/api/top_picks` is the pure population meta — every brawler at a full loadout,
**no roster filtering**. Mastery is loadout-forward — it ranks *investment* (which star powers /
gadgets / gears you own, plus comfort), not power level. Power is enforced separately as
a hard **fieldability gate**: Ranked doesn't normalize brawlers to a fixed power, and each bracket
blocks selecting a brawler below a floor (Power 9 through Diamond, Power 11 from Mythic up), so
`_roster_for` drops owned brawlers under `tiers.min_power_for_bracket(bracket)` before they're ever
scored — an un-maxed brawler you can't field in Legendary must not be recommended. The season's
free "boosted" brawlers arrive at Power 11 and are folded in *after* the gate, so an owned-but-
under-levelled free brawler (Ranked hands out a maxed copy) is still recommendable.

The free set (`main._free_brawler_ids`) unions two sources, because a wrong *omission* silently
deletes the map's best pick while a wrong *inclusion* only over-offers, and both sources are
conservative: the hand-maintained `ranked_boosted.json` (leading signal, knows *next* season) and
a signal read straight from match data (`engine/freebrawlers.py`, authoritative for *now*). The
release notes' rotation list is incomplete — a brawler can be granted free mid-season with no
announcement — but a free brawler is handed out maxed, so it shows as ~100% Power 11 with no
levelling tail in recent Ranked data. `DraftStats` accumulates that power histogram while it
already scans every match and ships the detected ids in the stats artifact, so the cloud (which
loads, never rebuilds) gets it for free. See the memory note *free-brawlers-detectable-from-match-power*.

## See also

- [MODEL_CARD.md](MODEL_CARD.md) — the win-probability model's math, training, calibration.
- [model-evaluation.md](model-evaluation.md) — signal-weighting ablations behind `DEFAULT_WEIGHTS`.
- [deployment-topology.md](deployment-topology.md) — how artifacts reach the cloud API and
  why the serve path is so constrained.
