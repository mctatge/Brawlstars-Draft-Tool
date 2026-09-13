# Per-item win rates (data-driven loadout advice)

How `/api/loadout` upgrades from the effect **heuristic** (Phase 1) to **measured** gadget / star
power picks. Battle logs never record the equipped item, so per-item win rates can't be read
directly — we infer them from *ownership*, the way community stats sites do, and gate hard on sample
size + significance so the served ranking never chases noise. Until the table is built + synced the
whole path stays on the heuristic; a cell only flips a `LoadoutItem` to `source:"winrate"` once it
clears the gate.

## The estimand

For a brawler **X** and item type **T** (gadget / star power / gear), among players who currently
own **exactly one** item of type T on X, attribute X's recent matches to that one owned item. The
per-item score is that item's win-rate edge over the brawler's **other single-item owners** (not vs
zero-owners): an *investment-matched* contrast — both cohorts own exactly one item, differing only in
*which* — which cancels most of the "players who unlock things play better" confound. Skill is
further controlled by stratifying on the appearance's **Ranked tier** (Mantel–Haenszel).

## Estimator (build side, `data/itemstats_build.py`)

Runs off-box on the home machine (needs the ownership profiles); numpy-heavy, never imported on the
serve path. Per `(X, T, item i)`:

1. **Window + recency** — matches within `window_days` (35) of the newest, weighted
   `0.5 ** (age / halflife)` (21 d), matching `engine/stats.py`. Draws (`a_won is None`) are dropped
   (variance-consistent for the test).
2. **Aggregate to one weighted record per player** before anything else — kills grinder domination
   and the correlated-rows problem. A single player's share of a pool is capped at 5% (iterated to a
   fixed point).
3. **Pool stats** per tier stratum for the item cohort and the REST cohort. The independent unit is
   the **player**, not the game: `n_eff = S1²/Σgₚ²` is Kish's effective *player* count, so one
   player's many correlated games don't count as independent evidence. The variance is the **larger
   of** the binomial-at-`n_eff` floor `p̂(1−p̂)/n_eff` (the i.i.d.-player minimum; also keeps a
   homogeneous pool from claiming zero uncertainty) **and** the robust between-player sandwich
   `Σ gₚ²(p̂ₚ − p̂)² / S1²` (which takes over under genuine overdispersion) — never the naive
   per-game binomial, which would treat the ~1,200 games behind 60 players as 1,200 independent
   trials and wildly over-reject.
4. **Mantel–Haenszel risk difference** of item-vs-REST across tier strata (a stratum enters only if
   both pools clear a small `n_eff` floor): `RD = Σ uₖ(p1ₖ−p0ₖ)/Σ uₖ`, `uₖ = n1ₖn0ₖ/Nₖ`, with a
   variance combined from the pools' clustered variances. Empty REST (single-item type) or all-thin
   strata → **no cell**, and the item stays on the heuristic (never a divide-by-zero).
5. **Empirical-Bayes shrinkage** toward zero effect: `delta = RD · n_eff/(n_eff + K1)` (K1 = 30).
   This is the served `delta` ("+X.X%"). The absolute `item_winrate` is anchored on the **REST**
   rate (`rest_rate + delta`), never the pooled rate that already contains i (which double-counts).
6. **Significance gate** — a cell earns `source:"winrate"` only if `n_eff ≥ 50` (item **and** rest),
   `≥ 30` distinct single-owners, it survives **Benjamini–Hochberg FDR** at q ≤ 0.05 over the
   **de-duplicated** test family (for a 2-item type the two mirror contrasts are one hypothesis,
   sharing a q), it isn't implausibly large on thin data, and the brawler isn't season-boosted.

## Serve side

`engine/itemstats.py` (pure stdlib) loads `itemstats.json[.gz]` with an mtime-keyed cache;
`engine/loadout.py` looks up `f"{brawler}:{item_id}"` (gears resolve via `meta.gear_ids_by_name`,
since gears carry no catalog id) and, on a significant cell, overlays `source:"winrate"`, a
delta-derived `fit`, and a measured `why`. Everything else — the response shape, the owned-item
overlay on the user's seat — is unchanged.

## Pipeline

```bash
# home box, IP-locked key:
PYTHONPATH=backend python -m bsdraft.collect.profiles --limit 500 --recent-days 35            # collect a bounded profile batch
PYTHONPATH=backend python backend/scripts/export_itemstats.py                          # -> data/processed/itemstats.json.gz
PYTHONPATH=backend python -m bsdraft.collect.publish --only-itemstats                  # -> GitHub Release asset
```

For the long-running home daemon, add `--itemstats` to the crawler command. It profiles a bounded
batch each cycle (defaults: 500 new profiles, players seen in the last 35 days, re-profiled after
21 days), rebuilds the table, and publishes it alongside the other release artifacts when
`--publish` is enabled. The standalone `profiles` command remains useful for a larger one-off
backfill.

Set `ITEMSTATS_URL` on the API to the Release asset; it syncs on the refresh loop and the loadout
loader picks up the new file on the next request. Build against the **same reference snapshot** as
`stats.json` so item ids and the brawler baseline line up.

## Limitations (surfaced honestly; the number is labelled "among single-item owners")

- **Associational, not causal.** Even MH-stratified and investment-matched, single-owner cohorts are
  self-selected; the delta is a correlation of owning-i with winning, not the item's causal lift.
- **External-validity gap.** It's estimated on players who own *exactly one* item of a type; the
  tool's users own everything and run 2 gears. The **ranking** is safer than the absolute level, and
  item×(star power/gear/hypercharge) interactions only maxed players see are invisible.
- **Own ≠ equip.** Ownership proxies equipping; for gadgets/star powers owning one almost always
  means equipping it (mostly attenuation noise), but it's unverifiable — the battle log can't confirm.
- **Temporal.** Profiles are a current snapshot vs historical matches; a player who unlocked/refunded
  i after the windowed games is misclassified — bounded by the 35 d window + recency + shrinkage, not
  eliminated (fetch timestamps are stored so stale profiles can be expired).
- **Gears are the weakest leg** — "owned" not equipped, two slots, uncontrolled level, ids only from
  live profiles, and the 2026 Epic/Mythic→Buffies phase-out churns the schema. Restricted to the six
  universal gears; expect mostly heuristic fallback.
- **Coverage builds slowly.** Off-meta brawlers and most gear cells stay below the floor and remain
  heuristic; needs an ongoing profile crawl (hundreds of thousands of tags) on the home key.
- **Crawl survivorship + meta-following.** The corpus is a snowball of active, higher-skill players;
  tier stratification and the item-vs-item contrast net out most of the "good players own the meta
  item" loop, but residual skill not captured by tier can survive — sanity-check the top pick isn't
  merely the most-owned item.
- **No map/mode split (yet).** The primary estimate pools modes; where ownership correlates with
  mode-of-play a pooled delta can mislead (Simpson risk). Mode can be added as an extra MH stratum
  where sample allows.

## Scope: mode-level, not per-map — but live to the actual draft

The effect heuristic (and the comp overlay below) score against **game mode** (Bounty, Gem Grab,
…), not the specific map. `loadout_advice()` accepts `map_id` but doesn't use it yet — reserved
for a future per-map slice once there's a reason to believe modes aren't a good-enough bucket.
So a brawler's gadget/star power/gear isn't a fixed per-map answer; what actually varies it
mid-draft is the **comp overlay**: once opponents are known, `enemies=<csv>` overlays live
class-count reads on top of the mode score. That's the mechanism behind e.g. Tick's *Last
Hurrah* gadget picking up a `+ vs poke` chip only once 2+ ranged enemies are drafted — the same
brawler on the same map/mode scores differently depending on who's actually in the enemy team
slots, not on a static per-map table.

## Comp-aware overlay (heuristic, Phase 1 — 2026-08-18)

`/api/loadout` accepts an optional `enemies=<csv>` param (ids of the queried brawler's opponents;
the frontend resolves the seat flip and mirrors blind-pick zeroing). Enemy **class counts** fire
coarse reads (`dive-heavy`, `2 Tanks`, `poke-heavy` — all thresholded at ≥2 picks) that add small
per-effect deltas shaped like `_MODE_EFFECT`, summed and clamped at ±0.15 — calibrated so the
heuristic can never claim more than a strong measured signal (+5% ≡ +0.15 fit via
`_FIT_PER_DELTA`). Composition policy:

- **Measured stays authoritative.** The comp delta folds into `fit` *after* `_apply_measured`,
  recorded on `comp_delta` (`fit − comp_delta` = comp-blind fit); `why`/`source` are never
  rewritten. `_mark_best`'s measured-better test runs on the comp-blind base fit, so a comp bump
  can't launder a measured-negative item into the pick.
- **Flips are labeled, not suppressed.** When the winner differs from the comp-blind winner the
  item carries `comp_flipped` and the UI badges `★ PICK · COMP`; signed chips (`+ vs dive`)
  explain each adjusted item. No hysteresis — the clamp restricts flips to near-ties.
- **Skew-safe by shape.** No `enemies` → byte-identical comp-blind output (the overlay is absent,
  not defaulted); junk CSV degrades silently; old clients/backends interoperate unchanged.
- **Unvalidatable by construction** (logs never record the equipped item) — the table
  (`_COMP_EFFECT` in `engine/loadout.py`) is capped opinion, honestly labeled in `note`.
  The measured path here is unchanged.

Phase 2 (2026-08-18) added two reads on top:

- **`cc_heavy`** — fires when **all three** enemies carry real enemy-targeted CC in their
  gadget/SP kits (`_cc_kit_ids`: the control-bucket keyword scan corrected by
  `_CC_KIT_OVERRIDES`). The overrides come from a full-roster audit of every kit description
  (item-level FP rate 8%): Carl/Janet denied (their "pull/push" moves themselves), 17 brawlers
  added whose pulls/knock-ups/sleeps/silences the first-match-wins scan bucketed elsewhere
  (Frank, Gene, Sandy, Ollie, El Primo, Finx…). Corrected base rate is ~54% of the roster —
  which is why the threshold is the full team, not 2-of-3 (2-of-3 would fire in most drafts).
  Overrides are name-keyed and degrade to the raw scan on unknown names. Re-run the audit when
  the catalog refresh adds brawlers or reworks descriptions.
- **Gear comp offsets** — `gears.json` gears may carry a sparse `"vs": {read: delta}` dict
  (Shield vs `aggro`/`cc_heavy`, Damage vs `tanky`, Health/Speed vs `poke`), summed over fired
  reads and clamped like the accessories, with the same `comp_delta`/chips/`comp_flipped`
  contract: the top-two selection rank-caps measured-worse gears at their comp-blind base fit
  (no comp laundering), and a gear in the picks only because of the comp is flagged. Junk `vs`
  (or `base`/`modes`/`roles`) values degrade to zero/defaults per the guide's fail-safe contract.
  `cc_heavy` fires strictly on the whole `enemies` list (a hand-crafted 5-id call can't fire it
  at 3-of-5).
