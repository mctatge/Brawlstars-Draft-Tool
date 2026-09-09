# Plan & Architecture

## 1. Goals

1. **Practical** — a fast, phone-usable assistant giving strong ban/pick advice during live ranked drafts (20-second pick timer).
2. **Portfolio** — a rigorous end-to-end AI/ML system (data pipeline → trained model → search/optimization → product) that demonstrably goes beyond existing tools.

## 2. Landscape & how we differ

Existing tools (MetaPick AI, BrawlAICoach, Deep Draft, PL Prodigy, Brawl Time Ninja) almost
all reduce to `map win-rate + pairwise synergy + pairwise counters (+ optional role penalty)`.
The closest open-source prior art, [DraftStars](https://github.com/mcmckinley/DraftStars),
trains brawler+map embeddings on 1M+ battles for a win-probability head. We adopt that core and
layer on refinements no competitor combines (§3).

## 3. Refinement parameters (the differentiators)

Each is an **explicit, tunable parameter** in the engine — exposed in the UI and the config.

1. **Win-probability model** — learned brawler + map + mode embeddings → `P(win)` for any 3v3 vs 3v3 on any map. The backbone everything else adjusts.
2. **Seat-aware draft search** — expectimax/minimax over the 1-2-2-1 snake. First pick → safe, flexible, counter-resistant picks + proactive bans; last pick → hard counters to the revealed enemy comp. Depth-limited lookahead at the enemy's best responses. *(Built in Phase 3, retired 2026-08-12: the model is now trained on masked partial drafts and scores the live board directly — see docs/MODEL_CARD.md. The minimax + its UI toggle were removed; git history has the module.)*
3. **Per-player mastery** — multiply a candidate's value by how well *you* (and known teammates) actually perform on it: personal win-rate, games played, power level, owned star powers/gadgets. "Don't first-pick a brawler you've played twice."
4. **Role / composition balance** — track coverage across the 7 classes (Tank, Assassin, Controller, Marksman, Support, Damage Dealer, Artillery). Bonus for filling holes; penalty for redundancy. Per-mode/per-map archetype targets (e.g. closed maps punish double-thrower; open maps want a marksman).
5. **Balance-patch / recency awareness** — win-rate data lags behind balance changes: a freshly **buffed** brawler looks weak in historical data (a **nerfed** one looks strong) until the meta re-equilibrates. Handled via **time-decay sample weighting** during training (tunable half-life) and **low-confidence flags** for new/changed brawlers with thin data.
   > ✅ **"buffies" resolved:** it's a real per-brawler upgrade field (`{gadget, starPower, hyperCharge}`), *not* balance-patches — a brawler "lacking buffies" is under-built. Handled in the **mastery layer (§3.3)**.
6. **Solo vs. premade context** — synergy-heavy comps assume coordinated teammates. A "solo queue" toggle **discounts synergy** and favors self-sufficient, low-coordination picks.
7. **Rank-bracket tuning** — filter/condition on the rank population (Diamond vs. Masters vs. Pro). A meta pick at Pro can be a trap at Diamond.
8. **Explainability & confidence** — every suggestion shows a weighted breakdown (`base + synergy + counter + role-fit + your-mastery`) and a **confidence/sample-size** indicator (Bayesian-adjusted). Plus the correct **gadget / star power / gear / hypercharge** for the map.

## 4. Architecture

```
                 Brawl Stars API                 Brawlify (keyless)
                 (battle logs)                    (brawlers/maps/assets)
                       │                                  │
        ┌──────────────▼───────────────┐                  │
        │  collect/  (async crawler)    │                  │
        │  rankings → snowball tags →   │                  │
        │  battlelog → dedup by match   │                  │
        └──────────────┬───────────────┘                  │
                       │ raw JSONL                         │
        ┌──────────────▼───────────────┐   ┌──────────────▼─────────────┐
        │  data/  (build_dataset)       │   │  data/  reference loaders  │
        │  clean · featurize · split    │◄──┤  brawlers/maps/classes     │
        └──────────────┬───────────────┘   └────────────────────────────┘
                       │ processed Parquet
        ┌──────────────▼───────────────┐
        │  models/  PyTorch win-prob    │  brawler+map embeddings → P(win)
        │  train · eval · export        │  baselines: logistic reg, heuristic
        └──────────────┬───────────────┘
                       │ model.pt
        ┌──────────────▼───────────────┐   ┌────────────────────────────┐
        │  engine/  draft search        │◄──┤  refinements/  mastery,    │
        │  recommend picks/bans         │   │  roles, seat, recency, …   │
        └──────────────┬───────────────┘   └────────────────────────────┘
                       │
        ┌──────────────▼───────────────┐   ┌────────────────────────────┐
        │  api/  FastAPI                │◄──┤  frontend/  Next.js board  │
        └───────────────────────────────┘   └────────────────────────────┘
```

## 5. Data pipeline

- **Source:** official API is *player-centric* (~25 recent battles per tag, no random-match feed, no ban data). So we **seed → snowball → dedup**:
  1. Seed tags from `/rankings/{country}/players` (global + many countries) and `/rankings/{country}/brawlers/{id}`.
  2. Pull `/players/{tag}/battlelog`; keep `battle.type` ∈ {ranked, soloRanked}.
  3. Snowball: harvest the other 5 tags from each match.
  4. **Dedup** by a stable match key = `battleTime` + sorted 6 player tags (same match appears in up to 6 logs).
  5. Poll continuously (logs roll off fast); throttle under the rate limit; respect the IP-locked key.
- **Each row** → `(map, mode, teamA brawlers[3], teamB brawlers[3], result)` + trophies/power per brawler.
- **Storage:** raw JSONL in `data/raw/`, cleaned Parquet in `data/processed/` (both gitignored).
- **Bootstrap:** while fresh data accrues, develop the model on the 2023 Kaggle match dataset (~100k rows) so the pipeline is testable on day one. (Swap to fresh data before any real recommendations.)

## 6. Model

- **Inputs:** map embedding, mode embedding, team A & team B each as an **order-invariant** aggregation (mean / attention) of 3 brawler embeddings.
- **Head:** combine `[teamA, teamB, map, mode]` → MLP → logit `P(A wins)`. Enforce **antisymmetry** (`f(A,B) = 1 − f(B,A)`) so swapping teams flips the prediction — a free, correct inductive bias.
- **Training:** binary cross-entropy; **recency weighting** (§3.5) and optional **bracket** conditioning (§3.7).
- **Baselines to beat:** (a) logistic regression on brawler one-hots, (b) the naive `winrate+synergy+counter` heuristic.
- **Evaluation:** log-loss, AUC, **calibration** (reliability curve + ECE — critical, since the engine consumes probabilities), and a draft-level top-k pick-agreement metric. Charts land in `docs/`.

## 7. Draft engine

- **State:** map, seat order, bans, picks so far, side to act.
- **Recommend pick:** prune to top-K candidates by a fast marginal-value heuristic, then **expectimax/minimax** over the remaining snake (opponent picks to maximize their win-prob; we evaluate full comps at the leaves with the model). Depth-limited to fit the 20s timer. *(Superseded 2026-08-12 by the partial-draft-native model — no search pass at serve time.)*
- **Recommend ban:** value a ban as the win-prob swing of removing a brawler from the live pool, weighing enemy threats *and* protecting your comfort picks; approximate the simultaneous-ban game.
- **Refinement hooks:** the leaf/candidate score is adjusted by mastery (§3.3), role balance (§3.4), recency (§3.5), solo/premade (§3.6), bracket (§3.7); the breakdown is surfaced for explainability (§3.8).

## 8. Roadmap

- [x] **Phase 0 — Foundation:** repo scaffold, keyless reference data (105 brawlers, maps, modes), config, docs.
- [x] **Phase 0.1 — Reference cleaning:** ✓ class overrides (all 105 brawlers classified); ✓ ranked-map filter (100 maps across 5 modes); ✓ brawler embedding index; ✓ star-power/gadget tables. (`bsdraft.data.reference`, runs dependency-free)
- [x] **Phase 1 — Data pipeline:** ✓ async client, smoke test, snowball crawler, match parser, dedup + resumable state. Validated on 300 real matches (0 bad rows, balanced labels, all ids reconcile). Scaling via background crawl. Bonus: `queue_type` (soloRanked/teamRanked) captured = a real solo-vs-premade signal. Note: Bounty is in the current ranked rotation (6 modes, not 5).
- [x] **Phase 2 — Win-prob model (pipeline):** ✓ encoders, dataset builder, antisymmetric embedding model (strength + low-rank counter term), training w/ recency weighting, baselines (always-0.5, logreg), metrics (logloss/acc/AUC/ECE) + charts. On 19k matches: beats both baselines, ECE 0.028 (well-calibrated). Absolute AUC ~0.57 — expected for top-ladder draft→outcome (skill-dominated). Engine will fuse this with empirical map win-rates. Retrain on full 30k pending.
- [x] **Phase 3 — Draft engine:** ✓ stats layer; ✓ fused pick scoring (model + map win-rate + synergy + counter + role-fit, transparent breakdown + confidence); ✓ ban recommendation; ✓ composition meter; ✓ **seat-aware 1-2-2-1 snake minimax search** (top-K pruned + memoized; captures first/last-pick value; ~1.8s worst case, <50ms mid-draft — *retired 2026-08-12 in favor of the partial-draft-native model*); ✓ **composition warnings** (no-frontline / no-range / double-tank / anti-tank / mode-specific); ✓ **mastery layer** (personalize to owned roster; weights power + comfort + build **incl. buffies**; restricts picks to owned; flags gaps like "missing buffie"). All wired into API + UI and verified. **Phase 3 complete.**
- [x] **Phase 4 — API + frontend:** ✓ FastAPI (reference + recommend; engine loaded at startup); ✓ Next.js 16 draft board — map select, ban/pick phases, click-to-place, live recommendations with map/synergy/counter/role/model breakdown + confidence, composition meter, snake-turn tracking, responsive dark UI. Verified rendering + interaction via preview.
- [ ] **Phase 5 — Polish:** methodology write-up, model card, eval charts, demo GIF, deploy, license.

## 9. Reference-data notes (from the fetched Brawlify data)

- **105 brawlers.** Class distribution: Damage Dealer 19, Assassin 14, Controller 14, Tank 12, Marksman 10, Support 10, Artillery 8, **Unknown 18** (newest brawlers not yet class-tagged → needs `data/reference/class_overrides.json`).
- **1,199 maps** total (all-time + community). Filter to `disabled=false` and `gameMode ∈ {Gem Grab, Brawl Ball, Knockout, Hot Zone, Heist}`, and prefer recently-active maps for the ranked pool.
- **Draft format:** 6 bans (3/team, simultaneous), 1-2-2-1 snake, 20s/pick, picks unique across the lobby.

## 10. Open questions / assumptions

- ~~**"buffies"**~~ → resolved: a real per-brawler API field, handled in the mastery layer (§3.3).
- **Python version** — 3.9.6 present; recommend 3.11+ (code kept 3.9-compatible where cheap).
- **Deployment target** for Phase 5 (Vercel + a Python host? single Docker? local-only?).
- **Map geometry in the learned model** (walls/bushes/lanes) remains out of scope: match rows have
  no positional data. The post-draft game plan may use explicitly curated openings for profiled
  maps (with mode-level fallbacks elsewhere); those directions are rules, never model output.
