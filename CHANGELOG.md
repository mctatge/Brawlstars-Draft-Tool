# Changelog

Notable, user-visible changes to [brawldraft.com](https://brawldraft.com). The site deploys
continuously from `main`, so entries are **dated, not versioned** — newest first. Routine
retrains, doc edits, and internal refactors are left out unless they changed what users see.

## 2026-08-24

- **The MATCHES counter now shows the whole dataset.** The console's header count was reading the
  server's in-memory stats window, which is deliberately capped at the 60,000 most recent matches
  to fit the small cloud instance — so it sat frozen at ~59,864 and looked like data collection
  had died. It now reports the true size of the live dataset (~1.6 million ranked matches and
  growing), counted directly from the synced match file with no extra memory cost.

- **Rank lookups can no longer take the site down — for real this time.** The 2026-08-20 fix
  (decoding the rank index in slices) bought time, not safety: the index has since grown from
  2.5 to 3.0 million players, its decode peak crept back to ~263 MB, and the server was
  OOM-killed again on the 23rd. Three durable fixes ship together: the index is now published
  as raw NumPy arrays (same ~14 MB download, but loading is ~66 MB and a third of a second —
  essentially a memcpy, so growth no longer moves the peak); if the index somehow can't load,
  the server now shows ranks as unknown until the next refresh instead of attempting a ~200 MB
  in-memory rebuild (the very thing that killed it); and a request with an absurdly long player
  tag — which quietly forced a full copy of the index per request, enough for one such request
  to crash the server — is now answered instantly without touching it.

## 2026-08-23

- **Backspace now undoes your last pick.** Press Backspace (or Delete) with the command box empty —
  which it always is right after you place something — and the most recently drafted brawler is
  removed and its slot re-armed to pick again. While you're mid-type it still edits your search text,
  and with nothing on the board it does nothing.

- **The enemy's first pick now sits on the far right.** Their draft fills right-to-left, mirroring
  the in-game layout so the board matches what you're looking at. Your own team is unchanged —
  first pick stays on the left.

- **Your rank re-checks itself every time you switch maps.** Switching maps usually means a new game,
  and rank can drift between games, so the tool now refreshes your live tier automatically instead of
  waiting for you to hit Load — no more drafting a game at last game's rank. It only updates on a
  successful lookup, so a momentary hiccup keeps your last known rank rather than dropping it.

- **Reset now also clears the search box**, so a fresh draft starts with an empty command line.

- **The Heist map "Safe(r) Zone" is back in the pool.** It's live in the ranked rotation, but the
  upstream Brawlify/BrawlAPI catalog still flags it disabled, so it was being filtered out entirely.
  A small force-enable override now keeps it in the ranked pool (and survives the next catalog
  refresh); it surfaces on the site once the crawler has collected enough Heist games on it. Until
  the next retrain the model reads it as a generic Heist map.

## 2026-08-21

- **Your personalized picks now price how far your copy is from the maxed brawler.** Every meta
  number — map win rate, the model's read, synergy, counter — describes a Power 11 brawler on a full
  loadout, because that is what almost the entire corpus is. Scoring an under-leveled or half-built
  copy off that table was silent extrapolation. Now the gap is priced and subtracted from your own
  pick's %: a **measured power-level deficit** (about 4 win-rate points at Power 10, 7.5 at Power 9,
  from a within-player estimate held behind a placebo gate), plus **estimated deficits** for a
  missing star power, gadget, or gear slot. The whole deduction is capped, and it fades out as your
  own record on that brawler grows — a record built on that exact copy already contains the
  handicap. A missing hypercharge is shown but left unpriced; nothing in the match log can measure
  it yet. Relatedly, **mastery is no longer blended into the score** — it is a display-only
  investment bar now, and your personal win rate applies as a small capped nudge rather than a
  weighted signal, so personalization breaks near-ties without floating a weak brawler up the board.

- **Arrow keys now steer the brawler grid from wherever you are.** The grid keeps a live cursor, and
  an arrow pressed outside it moves that cursor by the same step it would from inside — so from a
  fresh board `→` lands on the second brawler and `↓` on the row below, instead of spending a press
  to arrive on the first tile. Previously the keys did nothing until focus had been walked into the
  grid. Typing still owns the caret: with text in the command box, `←`/`→` move through it as before.

- **The post-draft game plan now shows the data, not just the advice.** It used to be a pure
  class-lookup: your three roles, the enemy's three class tips, and mode do's and don'ts. Those
  stay, but the panel now also reads from the same collected matches and win-prob model that rank
  the pick board. A **head-to-head grid** puts your three brawlers against each of theirs, colored
  by verdict, with your comp's average against each enemy — plus three callouts: the matchup to
  lean on, the one at risk, and the enemy your comp does worst against overall. Below it, **form on
  this map** (which of your picks actually performs here, who to play through, who is the weak
  link) and **your pairs** (which two want to play together). At the top, the model's **win
  probability for the finished draft**, and a **style-clash** line reading your comp's shape
  against theirs. Each measured number is shown with the effective sample behind it, and a cell too
  thin to say anything is left out rather than shown at low confidence — the "lean on" and "risk"
  callouts additionally have to clear a two-sigma bar, so a small sample can't take a headline on
  noise. Those two are also split at even, so "lean on" always names a matchup you actually win and
  "risk" one you actually lose. The rule-based half and the measured half are now grouped under
  their own headings, each labelled, so the two never blur.

- **New page: the model dossier.** `/model` is a technical write-up of the machine learning behind
  the board — the antisymmetric logit and the recency-weighted loss set in real LaTeX, nine
  hand-drawn architecture diagrams (the strength path, the counter cross, the mask rows, the two
  decay clocks, the training loop, calibration vs. draft state, signal shares), the full parameter
  table, and the held-out numbers with the places the model loses. Two pieces are interactive: drag
  the logit to watch P(A) + P(B) stay pinned at exactly 1, and click any cell of the 4x4 draft-state
  grid to see what masking does to it. Equations are typeset at build time, so the page ships no
  math JavaScript. "How it works" stays the plain-English version and links across to it.

## 2026-08-20

- **Fixed: an emptied roster could borrow another visitor's.** If the Ranked power-floor filter
  left you nothing fieldable (say, no Power-11 brawlers in a Mythic+ bracket), the backend treated
  your empty roster as "no roster sent" and fell back to the roster it had loaded last — which, on
  a shared host, is whichever player looked theirs up most recently. Your "personalized" picks
  could quietly be scored against someone else's brawlers. An explicitly sent empty roster now
  personalizes against exactly that: nothing owned, just the season's free boosted brawlers.

- **Bigger brawler portraits in the picker, and the boxes around them are gone.** Each grid cell had
  grown to 88px while the portrait inside stayed pinned at 44px, so every brawler sat in the left
  half of a muted grey box with dead space beside it. The portrait now fills its cell and the box is
  gone entirely — with a rarity-colored border on every portrait, a second grey frame around it was
  only competing. Portraits went from 44px to ~62px and a row now holds 11 instead of 8 (unchanged
  at 6 on phones, just larger). Hovering lifts a portrait and glows it in its rarity color.

- **Brawler portrait outlines now show rarity, like in-game.** Portrait borders used to repeat the
  brawler's class color, which the class chips already show. They now use the in-game rarity
  border colors (Rare green through Legendary yellow), and Ultra Legendary brawlers (Sirius, Kaze)
  get the game's animated prismatic ring — a rotating conic gradient with a soft color-cycling
  glow. The picker-grid and top-meta tile hover accents follow rarity too, and the active-slot
  accent ring still takes precedence while a slot is hot.

- **Diamond and below: pick suggestions split into meta + personal columns.** In blind-pick
  brackets everyone on the team picks at once, so marking "which pick is you" never made sense
  there — the seat checkboxes are gone at Diamond and below (Mythic+ keeps them and is unchanged).
  Instead, once your tag loads, the suggestion rail shows two columns for the whole pick phase:
  the map's meta picks, for advising teammates, beside your own best picks — filtered to what you
  can field, weighted by your mastery and your record. Enter locks the top personal pick (marked
  ⏎); with no tag you get the single meta list as before. The brawler picker stays unfiltered in
  these brackets, since teammates' picks are logged from it too.

- **Tab jumps to your first pick on the first press.** The shortcut skipped any unused ban slots as
  advertised, but refused to fire while the cursor sat in the type-to-place box — which is exactly
  where the board parks it on load and after every placement. So the first Tab merely walked focus
  into the brawler grid, and only the second one jumped. It now fires from that box. Every other
  field (the tag box, the map picker) still tabs through normally, and focus is never trapped: once
  the cursor is on your first pick, Tab goes back to plain focus traversal.

- **Looking up your Ranked tier no longer takes the whole site down.** The rank lookup loads an
  index of every crawled player's tier — 2.5 million of them now. Decoding it briefly needed
  ~350 MB for a 28 MB result, which was over the API's memory limit, so the server was killed and
  restarted every single time anyone entered a tag: a few minutes of downtime for everyone, per
  lookup. The index is now decoded in slices, peaking at ~110 MB, with byte-identical results.

- **The map picker now offers only the maps Ranked actually rotates.** It was listing every map
  still in the game's files for the five ranked modes — 113 of them, including pairs you can't
  queue, like "Heist: Pit Stop". The list is now the 27 maps that actually carry their mode's
  ranked games — four per mode (Gem Grab shows seven while the season boundary straddles two
  rotations). A map that's left the rotation keeps a fading tail of old games, so the cut is
  relative: every live map sits within ~8% of its mode's busiest, while a retired one sits 20x
  below. Maps we've barely seen played were ones the model had nothing to say about anyway.
- **Your Ranked tier no longer reports last season's rank as fact.** On a season reset the live
  profile lookup is the only source that knows you've been reset — our match data is a crawl
  snapshot with no season stamp. When that lookup came back "no tier yet this season", the badge
  quietly fell through to the pre-reset snapshot and showed the tier you'd just lost. It now
  says you haven't placed. And when the live lookup can't run at all, the badge still shows the
  snapshot but marks it with a `?` and says so on hover, instead of stating it flatly.

## 2026-08-19

- **Upgrade planner now ranks by value per coin, prerequisites included.** The first cut ranked
  purchases by raw win-rate value with cost as a side note — so its top suggestion could be a star
  power on a brawler still at Power 2 with no items, which you couldn't even field. Every
  recommendation is now the full *package* from where your account actually stands (the power
  climb to your bracket's floor, a first gadget + star power on an unbuilt brawler, the Starr Road
  unlock), priced as a whole and ranked by how much ranked win rate it buys per coin. Ranked's
  power floor is treated as the hard gate it is: below Power 9 (through Diamond) or Power 11
  (Mythic and up) a brawler gets one "make it ranked-ready" card and no item cards. Your live
  Ranked tier sets the floor (pin P9/P11 yourself if the lookup fails — unknown assumes Power 11,
  the safer guess); the bracket's own stats drive the meta read; roster depth discounts a 30th
  option versus a 1st; this season's free brawlers are discounted; unlocks only appear for the
  Starr Road tier you can buy from, with credit prices corrected (Epic 925 / Mythic 1,900 /
  Legendary 3,800 / Ultra Legendary 5,500). Cards show the package steps with per-step cost, a
  value-per-coin meter, and kind filters (Power / Gadgets / Star Powers / Gears / Hypercharges /
  Unlocks).

## 2026-08-18

- **Recommendations now favor the specific map over the game mode.** The mode-archetype nudge
  (e.g. Controllers in Gem Grab, Tanks in Brawl Ball) was quietly about as influential as real map
  win-rates and applied the same on every map of a mode — so the same brawlers surfaced regardless
  of the actual map. It's now scaled down by how much real data exists for that brawler on that
  map: on well-played maps the pick follows the genuine win-rate, while freshly-rotated maps still
  lean on the archetype guidance where there's no data yet. Brawlers no longer flat-top every map
  of their mode.
- **Faster analysis.** The recommend step is ~3.7× faster — the win-probability model now scores
  every candidate in one batched pass instead of one at a time. Identical picks, just quicker.
- **Keyboard-first drafting: Tab and arrow keys.** Press **Tab** to jump straight to your first
  pick, skipping any unused ban slots — handy when only a few bans are used. Browse the brawler
  grid with the **arrow keys**: press ↓ from the search box to drop into the grid, arrow between
  brawlers, and hit Enter to place the highlighted one. Tab still steps through form fields (like
  the tag box) normally and never traps focus, and the same keys work identically on Windows, macOS,
  and Linux.
- **Removed the confusing gear "level" from the loadout popover.** Owned gears used to show a bare
  "Lv3." Brawl Stars removed gear upgrade levels back in 2022 — gears are now a flat purchase at full
  power — so that number (always 3) meant nothing. Owned gears now just show their name.
- **Loadout advice now adjusts to the enemy comp.** The hover popover's gadget / star-power / gear
  picks were frozen at pick time; now the drafted enemy team feeds a bounded overlay: class-count
  reads (dive-heavy, 2 Tanks, poke-heavy) plus a CC-heavy read that fires when *every* enemy
  carries real crowd control in their kit (keyword scan corrected by a full-roster audit — Frank's
  pull, Sandy's sleep and friends were being missed; Carl/Janet's self-dashes no longer count).
  Adjusted items show signed chips ("+ vs dive"), the popover header names the reads it saw, and a
  pick that wins *only because* of the comp is badged ★ PICK · COMP. Gears join in via curated
  counter offsets (Shield vs dive/CC, Damage vs tanks, Health/Speed vs poke). Capped at ±0.15 fit
  so the comp nudges rather than overrules the mode read, measured win rates stay authoritative
  where they exist, and advice refreshes as picks land (Mythic+ drafts only — blind-pick brackets
  can't see the enemy team).
- **Bolt no longer shows Brock's gadgets.** The upstream catalog API serves "Rocket Laces" and
  "Rocket Fuel" under Bolt as well as Brock, and the committed snapshot had carried the duplicate
  since day one — polluting Bolt's loadout advice (the kit-description effect classifier read
  Brock's rockets as Bolt's). The snapshot is fixed, and the catalog fetch now strips any
  accessory served under two brawlers (its description names the real owner) or refuses the
  payload when it can't tell — so the next auto-refresh can't reintroduce it. A test now pins
  accessory-id uniqueness across the committed catalog.
- **Removed the buffie signal everywhere.** The roster API reports which buffies you *own* but
  never how many *exist* per brawler, so the "MISSING BUFFIE" tag misfired on every brawler with
  no buffies released (R-T among them — verified against maxed top-100 rosters). Dropped from the
  pick-card gap tags, the mastery build score (renormalized 3:2:2 over star power / gadget /
  gears), and the purchase advisor.
- **Ranked power floor now actually enforced on the live site.** The live roster service predated
  the gate and omitted `power`, so under-floor brawlers (e.g. a Power-9 Sandy at Legendary) still
  appeared in personalized picks. Redeployed; the picker now marks them "needs Power 11 in
  <bracket>" and they're excluded from your pick's recommendations.
- **Boosted-rotation season flips are now hands-off.** The `valid_until` fail-safe was only
  checked at process start, but the deployed API is kept warm for days — it could serve an expired
  FREE rotation long after a season flipped. The rotation logic now runs on every request against
  an explicit **UTC** clock (all serving hosts agree; hand-staged dates are targetable), and an
  upcoming rotation can be staged with an `active_from` date or exact instant: Season 1
  (Berry / Tara / Meg) serves through 2026-08-18 UTC, and Season 2 (Trunk / Willow / Kaze) takes
  over automatically at 2026-08-19 10:00 UTC — the overnight window between them deliberately
  serves no FREE set, since a wrong badge is worse than a missing one. Non-string dates fail safe
  instead of erroring, a boosted-watch rewrite carries staged dates forward for seasons whose
  names still match, and the committed file is schema-checked in tests.
- Loadout hover: both gear slots are starred on your own seat, not just the single best gear.

## 2026-08-17

- **Purchase advisor** shipped at `/purchases`: enter your tag and get your highest-value next
  purchases — power-11 climbs, gadgets, star powers, gears, hypercharges, new-brawler unlocks —
  ranked by meta strength × purchase impact, with costs as context.
- **Ranked power-floor gate**: drafts never recommend a brawler you can't select in your bracket
  (below Power 9 through Diamond, below Power 11 from Mythic up).

## 2026-08-12

- **Partial-draft-native model**: masked training bakes mid-draft evaluation directly into the
  win-probability net; the deep-search toggle is retired (the lookahead is now implicit).

## 2026-08-11

- **Seat-scoped personalization**: mark which of your team's three picks is *you* — only that
  seat's suggestions filter to your roster and history; teammates draft from the full meta.
- **Season's free "boosted" brawlers**: the three maxed brawlers Ranked hands everyone are
  recommendable even when unowned, with FREE badges (scraped from the official release notes).
- **Loadout hover**: hover any drafted brawler for gadget / star-power / gear advice, filtered to
  what you own on your own seat.
- **Data-driven loadout picks**: gadget/star-power suggestions backed by measured win rates from
  single-item-owner inference, not just heuristics.
- Fixes: ban slots are 3–6 (cursor no longer snaps back to skipped bans); tag-less visitors can no
  longer see the operator's roster; the crawler survives network outages.

## 2026-08-10

- **Console redesign**: the draft board became the tactical-telemetry console (THE CALL, pick
  orders, signal meters, confidence).
- Wendy + Nori added to the reference; the model vocabulary is pinned into the artifact so map
  refreshes can't silently shift embeddings.
- CI watchers added: balance-notes scrape, catalog diff, pipeline staleness.
- Scoring fusion rebalanced toward the trained model per the 995k-match ablation rerun.

## 2026-08-07 → 08-08

- Content pages (draft guide, how-it-works, FAQ), social share card, logo + brand assets, and
  env-gated (dark) AdSense wiring with a privacy page.

## 2026-07-16

- **Balance-change watch**: meta-report artifact, automatic retrain when drift trips, and a daily
  alert pipeline.
- `/api/rank` resolves tiers from a published rank-index artifact; the player tag is remembered
  across visits.

## 2026-06-17 → 06-22 — initial public launch

- Live at **brawldraft.com**: Python win-probability model + draft engine behind FastAPI, Next.js
  board, $0/mo deploy (Render + Cloudflare Pages) self-updating via home crawler → GitHub Release
  artifacts.
- Per-visitor roster personalization through a Cloudflare Tunnel (the Supercell key is IP-locked
  to the home machine); live Ranked-tier resolution; rank-bracket-conditioned stats; blind-pick
  mode for Diamond and below; live model hot-swap; meta-shift banner.
