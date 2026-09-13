# Purchase advisor — "what to upgrade next"

A personalized page (`/purchases`) that ranks a player's most *efficient* next purchases from
their live roster and Ranked bracket: power climbs, gadgets, star powers, gears, hypercharges, and
new-brawler unlocks. It is the inverse of the [loadout advisor](item-winrate.md): that tells you
which *owned* item to equip; this tells you which *unowned* purchase is most worth the coins.
(Buffies affect draft readiness, but are not yet purchase recommendations — see the Buffy note
under *Data reality*.)

Read this before touching `engine/purchases.py`, the `/api/purchases` endpoint, the
`economy.json` reference, or `frontend/components/PurchaseAdvisor.tsx`.

## Data reality — what the API can and can't tell us

The Supercell player endpoint exposes **ownership**, never **currency balances**. So we can compute
what a player is *missing* and what it *costs*, but never what they can *afford*. Confirmed against
a live roster (105-brawler account), every field the advisor needs is present and parsed by
`engine/mastery.py::parse_roster`:

- `power` (1–11) per brawler → exact power-deficit and its coin/power-point cost.
- owned gadget / star-power ids → diff against the catalog (`R.load_brawlers()`) to find the missing one.
- owned gears (id+name+level) → diff against the six universal gears (`reference/gears.json`).
- `has_hypercharge` (bool) → hypercharge slot state.
- roster membership → a brawler *absent* from the roster is a Credit-unlock candidate.

The player's **Ranked bracket** comes from `/api/rank` (live-first via the keyed tunnel); it sets
the power floor below and picks the bracket's stats table.

**Blind spots:** all currency balances; the equipped loadout (ownership only); and the upstream
*catalogs* of what gears / hypercharges / Buffies exist per brawler (none are catalog-backed).
Hypercharge and Buffy availability are handled by curated policies.

**Buffies affect picks, but are not advised as purchases yet.** The roster carries a
`buffies: {gadget, starPower, hyperCharge}` object, whose flags say only which gameplay Buffies the
player owns. `data/reference/buffies.json` now resolves the previously ambiguous all-false case by
listing the brawlers whose three gameplay Buffies have actually shipped. The draft scorer uses that
policy and applies a transparent, estimated 0.010 readiness deficit per missing Buffy; old or
partial roster data stays neutral, and brawlers without released Buffies are never flagged.

The purchase advisor still omits them. Ranking a purchase needs a reviewed cost/package model and
a value prior in the advisor's relative-lift units, not merely the readiness correction used to
compare two fielded copies. Adding a half-specified "Buy a Buffy" row here would look precise while
mixing those units. The fourth, cosmetic Buffy is excluded from both paths because it has no match
effect and is not part of the roster's three-boolean gameplay object.

## Scoring (product decisions, revised 2026-08-19)

The first version ranked by raw win-rate value with cost as context only, and assumed Ranked
normalizes everyone to Power 11. Both were wrong in practice: the top suggestion for the owner's
own account was "buy Damian's star power" on a Damian parked at Power 2 with no items — an item
that needs a ~7,700-coin climb first and can't even be fielded in their bracket. The rules now:

**Rank by value per cost.** `value_score = value_lift / cost_equiv × 1000` — win-rate lift per
1,000 coin-equivalents, with `cost_equiv = coins + 0.5·power_points + 1.0·credits`
(`scoring.resource_weights`). Balances stay unknowable, so this is efficiency, not affordability.

**Prerequisites are part of the purchase.** Every rec is the *package* that realizes its value
from the player's current state, with `steps[]` in purchase order and `cost` summed across them:
the power climb to the bracket's floor, a first gadget + star power for an unbuilt brawler, the
Starr Road unlock for an unowned one. Value is **additive over the package** — the climb carries
the value of *access* (a bare fieldable brawler) plus every item it realizes, owned or bought — so
the same end state is worth the same whichever path reaches it.

**The Ranked power floor is a hard gate.** Ranked does **not** normalize power: brawlers play at
their real level and each bracket blocks selecting one below a floor — Power 9 through Diamond,
Power 11 from Mythic up (`tiers.min_power_for_bracket`; the same gate the draft board applies).
A brawler below the floor is unfieldable, so nothing bought on it has realizable value: it gets
exactly **one** rec (the "ranked-ready" package) and no item recs. When neither the rank nor an
explicit floor is known, the **stricter Power-11 floor is assumed** (`scoring.unknown_floor`) —
hiding a few P9–10 item recs from a Diamond player beats recommending unrealizable buys to a
Mythic one. The page lets the user pin P9/P11 (`power_floor` in the request, persisted locally).

**Meta-weighted, roster-aware for draft likelihood.** Every rec on a brawler is multiplied by
`meta_factor × depth_factor` (× `boosted_discount` if it's this season's free brawler):

- `meta_factor = exp((winrate − 0.5) / 0.125)` over the games-weighted win rate across the
  current ranked maps, from the *bracket's* stats table when one exists. Live ranked-pool
  strengths run ~.44–.59 ⇒ factors ~.64–2.0, so kind structure still dominates and meta
  re-orders within a kind.
- `depth_factor = 1 / (1 + max(0, n − 8) / 12)` where `n` is the build-weighted number of
  *stronger* brawlers the player can already field (a bare P11 counts half; this season's boosted
  brawlers count at full), taken over the overall pool **and** each ranked mode with ≥100 real
  games — each mode rate shrunk toward the brawler's overall rate by 300 pseudo-games, so a
  30-game blip can't buy full value while a 400-game specialist keeps most of its edge — keeping
  the smallest. Your top ~8 options are all full value; a 30th is worth ~1/3. The rationale names
  the mode (and the rate there) when a specialist view is what counts.

Mains/mastery never enter: ownership decides what's buyable, the meta decides what's worth it.

**Priors** (`impact_priors`, relative lift units — only ratios matter): first star power 3.5 ≳
first gadget 3.0 > hypercharge 6.0 at 5k coins > gears 0.7 (core slot) / 0.25 (spare) > second
copies 0.6 — per 1k coin-eq at equal strength that is gadget 3.0 > star power 1.75 > hypercharge
1.2 > core gear 0.7 > second gadget 0.6, so a hypercharge on a main beats a second copy or a core
gear on a slightly stronger brawler; `access` 3.0 for a bare fieldable brawler; `power_level`
1.5 per level of real stats (+10% of base each; 9→11 ≈ +11% HP/dmg); `readiness` 0.5 = the share
of a brawler's ranked-ready value a ≤Diamond climb to 11 earns now (Mythic needs Power 11). Gear
slots open at Power 8 and 10; a *spare* gear (both slots filled) is only recommended on a
significant positive measured delta — unmeasured spares are filler. One power-related rec per
fieldable brawler: the climb to 11 while short of it, the hypercharge as a straight buy once there.

**Measured item deltas** ([itemstats](item-winrate.md), estimand item-vs-other-owners): on a
*first* item the delta only picks which one to buy; on a *second* item or a gear it scales the
value — signed, capped at ×0.5..×2 per ±10pp (`scoring.delta_scale`). A significant *negative*
delta never wins selection over an unmeasured alternative (the data says it's the worse buy).

**New-brawler unlocks** are only offered on the Starr Road tier the player can actually buy
from (the lowest of Epic → Mythic → Legendary → Ultra Legendary with an unowned brawler — the
road is walked tier by tier), only for brawlers the **global** table has ≥40 real games for (an
unreleased or retired collab brawler is never vouched for, and doesn't hold a tier open either;
the check runs on the global table because a thin bracket's counts say nothing about existence),
and priced as the full package: credits + the climb to the floor + a first gadget and star power.
Rare / Super Rare are Trophy Road rewards since 2025-06, not purchases.

Each rec carries `confidence`: `measured` (itemstats cell) / `heuristic` (prior) /
`eligibility_only` (hypercharge — no value model yet), plus `value_lift`, `cost_equiv`,
`cost_estimated` (a step had no known price and got a nominal one), `steps[]`, `gate` and
`target_power`. The API reserves `min_per_kind` best-of-kind slots so a legitimately
low-efficiency kind (unlocks — a separate currency plus a full climb) stays discoverable below
the overall top; the page filters by kind.

## The economy table — `data/reference/economy.json`

Hand-maintained researched constants: costs, power gates, impact priors, scoring knobs,
hypercharge availability. **None of this is in any API and it drifts on balance updates** —
verified against the Brawl Stars wiki + Supercell release notes on 2026-08-19 (no 2025–26
change to upgrade/item prices; Starr Road credits Epic 925 / Mythic 1,900 / Legendary 3,800 /
Ultra Legendary 5,500). Because cost is now the ranking denominator, **a stale price misranks,
not just mislabels** — keep it fresh. Every section has a code fallback (`_DEFAULT_*` in
`engine/purchases.py`) overlaid per key by the file, so a missing or partial section degrades to
the defaults rather than to a free package; a rarity priced `null` gets the dearest known price as
a nominal and the rec is flagged `cost_estimated`. The `impact_priors` section is only honoured
when it is v2-shaped (carries `access` / `power_level` / `gear_core`) so a legacy 0..1 table can't
be mixed into lift units. A patchnotes-fed refresh is a possible follow-up.

Hypercharge availability uses a `mode`/`list` policy: `all_except` treats every Power-11 brawler as
eligible unless listed (matches the ~all-brawlers-have-one reality, and only fires at Power 11);
flip to `only` + an explicit list for conservative, under-covering behavior. Missing section ⇒ no
hypercharge recs (fail-safe).

## Wiring

- **Endpoint** `POST /api/purchases` ([api/main.py](../backend/bsdraft/api/main.py)) mirrors
  `/api/recommend`: the client fetches the roster from the keyed tunnel (`ROSTER_BASE`) and POSTs it
  with `rank_bracket` (and an optional pinned `power_floor`) to the public host (`API_BASE`), which
  holds the stats + itemstats and can't fetch a roster itself (IP-locked out of Supercell).
  `OwnedBrawler` carries `power`/`has_hypercharge` for the advisor. The response echoes
  `rank_bracket` / `power_floor`.
- **Engine** `engine/purchases.py` → `DraftEngine.recommend_purchases` (which passes the
  bracket's stats table like a draft does); reuses the loaded `stats` and `IS.get_itemstats()`.
  No new artifact; serve path stays torch/pandas-free.
- **Frontend** `app/purchases/page.tsx` + `components/PurchaseAdvisor.tsx`: roster + rank fetched
  in parallel, tag reused from `localStorage["bsdraft.tag"]`, floor pin in
  `localStorage["bsdraft.purchaseFloor"]`, `getPurchases()` in `lib/api.ts`. Cards show the
  package steps with per-step cost, the value-per-coin meter relative to the list's best, and
  kind filter chips.

## Constraints

- **Live-only, no fallback.** Full ownership comes only from the keyed roster tunnel; the public
  host and the (stale, power-less) `profiles.jsonl` can't substitute. The page is dead during the
  recurring 403 IP-rotation outages or when the home machine is offline — it degrades to a clear
  "roster service is down" state. Same single-point-of-failure as board personalization.
- **Local testing:** the roster path needs the IP-locked key + CORS, so verify end-to-end on the
  deployed site — or, on the home machine, run `backend-local` (port 8099, CORS `*`) +
  `frontend-local-api` from `.claude/launch.json`: the local backend has the key, so roster, rank
  and purchases all work from `localhost:3000`. See [dev-commands.md](dev-commands.md).
