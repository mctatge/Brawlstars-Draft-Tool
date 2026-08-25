// Typed client for the draft API (FastAPI backend).

export type Brawler = { id: number; name: string; cls: string; rarity: string; image_url: string };
export type GameMap = { id: number; name: string; mode: string; image_url: string; games: number };
export type Reference = { brawlers: Brawler[]; maps: GameMap[]; modes: string[]; brackets: string[]; boosted: number[] };

export type PickRec = {
  brawler_id: number; name: string; cls: string; score: number; map_winrate: number;
  synergy: number | null; counter: number | null; role_fit: number;
  win_prob: number | null; confidence: number;
  mastery: number | null; personal_winrate: number | null; personal_games: number | null;
  owned: boolean; gaps: string[];
  breakdown: Record<string, number>;
};
export type BanRec = {
  brawler_id: number; name: string; cls: string; threat: number;
  map_winrate: number; use_rate: number; confidence: number;
  // Projected swing in your win probability from banning this brawler, given everything already
  // banned and who picks first — the sort key. Null when the backend has no model to project
  // with, where the list falls back to raw threat order.
  ban_value: number | null;
  replacement: string | null;   // who the enemy builds around instead
  self_deny: boolean;           // the draft projects this brawler onto *your* side
};
export type Warning = { text: string; severity: string };
export type RoleTip = { name: string; cls: string; role: string };
export type ThreatTip = { name: string; cls: string; tip: string };
export type EnemyRead = { archetype: string; playstyle: string; clash: string };
// One of your brawlers' win rate on this map. `tag` is anchor / solid / weak.
export type MapForm = { name: string; cls: string; winrate: number; games: number; tag: string };
export type PairRate = { a: string; b: string; winrate: number; games: number; edge: string };
// `winrate` is null when that (ours, theirs) cell has too thin a sample to show.
export type H2HCell = { name: string; winrate: number | null; games: number; edge: string };
export type H2HRow = {
  enemy: string; enemy_cls: string; vs: H2HCell[]; mean: number | null;
  mean_cells?: number;   // how many cells survived the floor to form `mean`
  mean_games?: number;   // their combined effective sample
};
export type H2HCallout = {
  ours?: string | null; theirs?: string | null; enemy?: string | null; enemy_cls?: string | null;
  name?: string | null; winrate?: number | null; games?: number | null;
  cells?: number | null;  // focus only: how many cells its average is over
  edge?: string | null;
};
export type HeadToHead = {
  grid: H2HRow[];
  // Each is independently null: `focus` needs a row averaging 2+ surviving cells and a sub-even
  // average, `best` a cell above even, `danger` one below. A grid with nothing to say on an axis
  // omits that chip instead of inventing one, so never assume all three are present.
  focus: H2HCallout | null;    // the enemy your comp does worst against overall
  danger: H2HCallout | null;   // your worst losing cell
  best: H2HCallout | null;     // your best winning cell
};
export type ModelRead = { win_prob: number; verdict: string; note: string };
export type GamePlan = {
  objective: string; win_condition: string; archetype: string; playstyle: string;
  roles: RoleTip[]; threats: ThreatTip[]; tips: string[]; avoid: string[]; compensate: string[];
  // Data-backed half — each is independently empty/null when its cells are too thin to speak
  // from. Backed by the same collected matches + win-prob model that rank the pick board.
  // Optional because the frontend and the API deploy separately: a Pages build can go live
  // against a Render instance that predates these fields, so the panel must read them
  // defensively (`?? []`) rather than assume they're on the wire.
  enemy?: EnemyRead | null;
  map_read?: MapForm[];
  pairs?: PairRate[];
  head_to_head?: HeadToHead | null;
  model_read?: ModelRead | null;
};
export type RecommendResponse = {
  phase: string; picks: PickRec[]; bans: BanRec[];
  composition: Record<string, number>; warnings: Warning[];
  game_plan: GamePlan | null; next_to_act: string | null;
};

export type OwnedGear = { id: number; name: string; level: number };
export type OwnedBrawler = {
  id: number; mastery: number; gaps: string[];
  // Specific items the player owns on this brawler — populated by /api/roster. Used to restrict
  // loadout suggestions on the user's own pick to what they have, and read server-side by
  // /api/purchases. They ride along on the recommend request too (that payload is an unprojected
  // `.filter()` over these same objects — see `fieldableOwned` in DraftBoard.tsx), where the
  // backend accepts but ignores them.
  owned_star_powers: number[]; owned_gadgets: number[]; owned_gears: OwnedGear[];
  // Progression state from /api/roster, sent on the recommend request as well. `has_hypercharge` is
  // only used by the purchase advisor, but `power` MUST stay on the recommend payload: the backend
  // gate is `power == 0 || power >= floor`, so an entry that arrives without `power` defaults to 0
  // and passes unconditionally — slimming it out of the POST body silently disables the Ranked
  // power-floor gate. These are optional only because an older backend may not return them; that is
  // not license to project them away when POSTing.
  power?: number; has_hypercharge?: boolean;
};
export type RosterResponse = {
  loaded: boolean; tag: string; name: string; owned: OwnedBrawler[]; error?: string | null;
};

export type LoadoutItem = {
  id: number | null; name: string; kind: "gadget" | "star_power" | "gear";
  image_url: string; effect: string; description: string;
  fit: number; recommended: boolean; why: string; source: string;
  // Enemy-comp overlay (optional so old backends still type-check): applied fit adjustment,
  // signed reason chips ("+ vs dive"), and whether the pick only wins because of the comp.
  comp_delta?: number; comp_why?: string[]; comp_flipped?: boolean;
};
export type LoadoutResponse = {
  brawler_id: number; brawler_name: string; cls: string; mode: string;
  gadgets: LoadoutItem[]; star_powers: LoadoutItem[]; gears: LoadoutItem[]; note: string;
  comp_reads?: string[];         // fired enemy-comp reads, e.g. ["dive-heavy (2 Tank/Assassin)"]
};

export type PurchaseKind =
  "power_upgrade" | "gadget" | "star_power" | "gear" | "hypercharge" | "new_brawler";
export type PurchaseStep = { kind: PurchaseKind; label: string; cost: Record<string, number> };
export type PurchaseRec = {
  brawler_id: number; brawler_name: string; kind: PurchaseKind;
  value_score: number;                       // the sort key: win-rate lift per 1,000 coin-equivalents
  value_lift: number;                        // relative lift the whole package realizes
  cost: Record<string, number>;              // the package incl. prerequisites, e.g. { coins: 4050, power_points: 890 }
  cost_equiv?: number | null;                // coin-equivalents (null ⇒ no price known)
  cost_estimated?: boolean;                  // a step had no known price and got a nominal one
  meta_winrate: number;
  confidence: "measured" | "heuristic" | "eligibility_only";
  rationale: string;
  steps?: PurchaseStep[];                    // the package in purchase order (1 step = a plain buy)
  item_id: number | null; item_name: string | null; target_power: number | null;
  item_delta: number | null; gate: string | null;
};
export type PurchasesResponse = {
  tag: string; name: string; scope: string;
  // Optional so a not-yet-redeployed backend still type-checks (Pages and Render roll separately).
  rank_bracket?: string | null;              // bracket the power floor came from (null ⇒ unknown)
  power_floor?: number;                      // Power an owned brawler needs to be fieldable (9 or 11)
  recommendations: PurchaseRec[];
};

export type RankInfo = {
  found: boolean; tag: string; tier: number | null; tier_label: string | null;
  bracket: string | null; source: string | null;
  // Set when the tier came from our match data *and* the live lookup that would have corrected
  // it couldn't run. Crawl rows carry no season stamp, so after a Ranked reset they report a tier
  // the player has already lost — show it, but don't show it as fact.
  stale?: boolean; error?: string | null;
};

export type Health = {
  status: string; model: boolean; matches: number; roster: boolean;
  refresh_seconds: number; last_check: number | null; last_change: number | null;
};

export type MetaShift = {
  brawler_id: number; name: string; kind: string;
  wr_before: number; wr_after: number; use_before: number; use_after: number; z: number;
};
export type Meta = {
  shifted: boolean; n_recent: number; n_prior: number;
  new_brawlers: string[]; shifts: MetaShift[]; note: string;
};

export type TopPick = {
  brawler_id: number; name: string; cls: string; score: number; map_winrate: number;
};
export type TopPicksBody = {
  map_id: number; mode: string; our_team: number[]; their_team: number[]; bans: number[];
  rank_bracket?: string | null; top: number;
};
export type TopPicksResponse = {
  map_id: number; mode: string; rank_bracket: string | null; picks: TopPick[];
};

export type RecommendBody = {
  map_id: number; mode: string; our_team: number[]; their_team: number[]; bans: number[];
  we_pick_first: boolean; solo_queue: boolean; rank_bracket?: string | null; phase: "pick" | "ban";
  personalize: boolean; personal_tag?: string | null; top: number;
  // The player's owned brawlers + mastery + loadout gaps, sent so the public backend (which can't
  // fetch the roster itself — IP-locked out of Supercell) can personalize the suggestions.
  roster?: OwnedBrawler[] | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
// Roster needs a live Supercell call, which only works from an IP whitelisted with the key.
// Point this at a whitelisted host (e.g. a Cloudflare Tunnel to the home machine) to enable
// per-visitor personalization on the public site; defaults to the main API otherwise.
const ROSTER_BASE = process.env.NEXT_PUBLIC_ROSTER_BASE || API_BASE;

export async function getReference(): Promise<Reference> {
  const res = await fetch(`${API_BASE}/api/reference`);
  if (!res.ok) throw new Error(`reference: ${res.status}`);
  return res.json();
}

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`health: ${res.status}`);
  return res.json();
}

export async function getMeta(): Promise<Meta> {
  const res = await fetch(`${API_BASE}/api/meta`);
  if (!res.ok) throw new Error(`meta: ${res.status}`);
  return res.json();
}

export async function getRank(tag: string): Promise<RankInfo> {
  // Through ROSTER_BASE (the keyed tunnel), not API_BASE: a live battle-log lookup gives the
  // player's *current* tier, whereas the keyless API can only return the crawl snapshot, which
  // goes stale across a Ranked season reset. Falls back to API_BASE when no tunnel is set.
  const res = await fetch(`${ROSTER_BASE}/api/rank?tag=${encodeURIComponent(tag)}`);
  if (!res.ok) throw new Error(`rank: ${res.status}`);
  return res.json();
}

export async function getRoster(tag?: string | null): Promise<RosterResponse> {
  const qs = tag ? `?tag=${encodeURIComponent(tag)}` : "";
  const res = await fetch(`${ROSTER_BASE}/api/roster${qs}`);
  if (!res.ok) throw new Error(`roster: ${res.status}`);
  return res.json();
}

// Fire-and-forget: ask the scoring backend to pre-build this tag's personal stats so the first
// personalized pick doesn't block on its dataset scan. Deliberately API_BASE, not ROSTER_BASE:
// rank resolution goes through the keyed tunnel (a different machine in production), and a warm
// there does nothing for the host that actually scores /api/recommend. Best-effort by design —
// failures (including a 404 from a backend that predates the endpoint) are swallowed, because
// the pick-phase build still covers the tag lazily.
export function warmPersonal(tag: string): void {
  fetch(`${API_BASE}/api/warm?tag=${encodeURIComponent(tag)}`).catch(() => {});
}

export async function getTopPicks(body: TopPicksBody): Promise<TopPicksResponse> {
  const res = await fetch(`${API_BASE}/api/top_picks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`top_picks: ${res.status}`);
  return res.json();
}

export async function recommend(body: RecommendBody): Promise<RecommendResponse> {
  const res = await fetch(`${API_BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`recommend: ${res.status}`);
  return res.json();
}

export async function getPurchases(
  roster: OwnedBrawler[], tag?: string | null, name?: string | null, top = 24,
  rankBracket?: string | null, minPerKind = 0, powerFloor?: 9 | 11 | null,
): Promise<PurchasesResponse> {
  // Scored on API_BASE (holds the stats + item win-rate table); the roster was fetched from
  // ROSTER_BASE (the keyed tunnel) and is POSTed here, mirroring the recommend path — the public
  // backend can't fetch a roster itself (IP-locked out of Supercell). rank_bracket sets the
  // Ranked power floor (9 through Diamond, 11 from Mythic up) that decides what's fieldable and
  // picks the bracket's stats table; power_floor pins the floor explicitly (user override); with
  // neither, the backend assumes the stricter Power 11.
  const res = await fetch(`${API_BASE}/api/purchases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      roster, tag: tag || null, name: name || null, top,
      rank_bracket: rankBracket || null, min_per_kind: minPerKind,
      power_floor: powerFloor || null,
    }),
  });
  if (!res.ok) throw new Error(`purchases: ${res.status}`);
  return res.json();
}

export async function getLoadout(brawlerId: number, mode: string, mapId?: number | null,
                                 enemies?: number[]): Promise<LoadoutResponse> {
  const qs = new URLSearchParams({ brawler: String(brawlerId), mode });
  if (mapId != null) qs.set("map_id", String(mapId));
  if (enemies && enemies.length) qs.set("enemies", enemies.join(","));  // comp-aware overlay
  const res = await fetch(`${API_BASE}/api/loadout?${qs.toString()}`);
  if (!res.ok) throw new Error(`loadout: ${res.status}`);
  return res.json();
}
