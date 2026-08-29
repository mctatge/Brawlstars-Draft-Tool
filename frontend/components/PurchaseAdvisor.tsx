"use client";

// "What to upgrade next" — enters a player tag, fetches their live roster (owned items + power
// levels via the keyed roster tunnel) and Ranked bracket, and ranks the most *efficient* purchases
// they haven't made: power climbs, gadgets, star powers, gears, hypercharges, new-brawler unlocks.
// The inverse of the board's loadout popover — it surfaces the best UNOWNED item rather than
// locking it. Scored on the backend (see engine/purchases.py) by win-rate value per coin with every
// prerequisite priced into the package: the power climb to the item's gate and to the bracket's
// Ranked power floor (Power 9 through Diamond, 11 from Mythic up — below it a brawler can't be
// fielded at all), a first gadget + star power for an unbuilt brawler, the Starr Road unlock.
// Balances stay unknowable (the API only exposes ownership), so it's value per coin, not "can I
// afford it".
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getReference, getRoster, getRank, getPurchases,
  type Brawler, type OwnedBrawler, type PurchaseRec, type PurchaseKind, type PurchaseStep, type RankInfo,
} from "@/lib/api";
import Logo from "@/components/Logo";

const NAV = [
  { href: "/", label: "Draft board" },
  { href: "/purchases", label: "Upgrades" },
  { href: "/guide", label: "Draft guide" },
  { href: "/how-it-works", label: "How it works" },
  { href: "/faq", label: "FAQ" },
];

const KIND: Record<PurchaseKind, { label: string; color: string; glyph: string }> = {
  new_brawler:   { label: "UNLOCK",       color: "var(--blue)",   glyph: "◈" },
  hypercharge:   { label: "HYPERCHARGE",  color: "var(--gold)",   glyph: "⚡" },
  star_power:    { label: "STAR POWER",   color: "var(--accent)", glyph: "★" },
  gadget:        { label: "GADGET",       color: "var(--green)",  glyph: "⚙" },
  gear:          { label: "GEAR",         color: "var(--muted)",  glyph: "⛭" },
  power_upgrade: { label: "POWER",        color: "var(--accent)", glyph: "▲" },
};

const COST_LABEL: Record<string, string> = {
  coins: "Coins", power_points: "Power Points", credits: "Credits",
};

// Filter chips, in display order. "all" shows the single ranked list; a kind narrows it — handy
// because credits (unlocks) are a separate budget from coins, so unlocks sit low in value-per-coin.
const FILTERS: Array<{ key: "all" | PurchaseKind; label: string }> = [
  { key: "all", label: "ALL" }, { key: "power_upgrade", label: "POWER" }, { key: "gadget", label: "GADGETS" },
  { key: "star_power", label: "STAR POWERS" }, { key: "gear", label: "GEARS" },
  { key: "hypercharge", label: "HYPERCHARGES" }, { key: "new_brawler", label: "UNLOCKS" },
];
// Ask for enough to fill a filtered view; the backend reserves a few best-of-kind slots so a kind
// that's legitimately low-efficiency (unlocks) is still discoverable below the overall top.
const TOP = 36;
const MIN_PER_KIND = 3;

// The Ranked power floor: Power 9 through Diamond, Power 11 from Mythic up (a hard block — below it
// a brawler can't be selected). "auto" takes it from the live rank lookup; when that fails the
// backend assumes the stricter Power 11 (the safer failure for an advisor). The user can pin it —
// e.g. a Legendary player knocked to Diamond by the season reset who is planning for Mythic.
type FloorChoice = "auto" | 9 | 11;
const FLOOR_KEY = "bsdraft.purchaseFloor";
const P11_BRACKETS = new Set(["Mythic", "Legendary", "Masters", "Pro"]);
const floorForBracket = (b: string | null | undefined): number | null => (b ? (P11_BRACKETS.has(b) ? 11 : 9) : null);
function readFloorChoice(): FloorChoice {
  if (typeof window === "undefined") return "auto";
  const v = localStorage.getItem(FLOOR_KEY);
  return v === "9" ? 9 : v === "11" ? 11 : "auto";
}

const CONF: Record<PurchaseRec["confidence"], { label: string; color: string; title: string }> = {
  measured:         { label: "MEASURED", color: "var(--green)",
                      title: "Backed by a measured item win-rate from match data" },
  heuristic:        { label: "ESTIMATE", color: "var(--muted)",
                      title: "Ranked by the brawler's meta strength × a purchase-impact prior" },
  eligibility_only: { label: "ELIGIBLE", color: "var(--gold)",
                      title: "A high-value slot you're eligible to fill — no measured value model yet" },
};

// Roster fetch fails with raw operator text; translate to something a visitor can act on.
function rosterFailReason(error: string): string {
  const e = error.toLowerCase();
  if (e.includes("purchases:")) return "couldn't score your roster — the API may be restarting, try again";
  if (e.includes("404") || e.includes("not found")) return "no player with that tag — check it and load again";
  if (e.includes("429")) return "roster service is busy — try again in a moment";
  if (e.includes("403") || e.includes("auth/ip") || e.includes("no api token") || e.includes("no player tag"))
    return "the roster service is down right now — try again later";
  return "couldn't reach the roster service — try again later";
}

const pct = (v: number) => `${Math.round(v * 100)}%`;
const num = (n: number) => n.toLocaleString();

function DocNav({ current }: { current: string }) {
  return (
    <nav className="panel flex flex-wrap items-center gap-x-1 gap-y-1 px-3 py-2 mb-6">
      <a href="/" className="flex items-center gap-2 mr-3" aria-label="Brawl Draft home">
        <Logo size={22} />
        <span className="brand-gradient text-[14px]">BRAWL DRAFT</span>
      </a>
      <div className="flex flex-wrap gap-x-1 ml-auto">
        {NAV.map((n) => {
          const on = n.href === current;
          return (
            <a key={n.href} href={n.href} aria-current={on ? "page" : undefined}
              className="mono text-[11px] tracking-[0.06em] uppercase px-2.5 py-1.5 border ctl"
              style={on
                ? { color: "var(--text)", borderColor: "var(--accent)", boxShadow: "inset 0 0 0 1px color-mix(in srgb, var(--accent) 40%, transparent)" }
                : { color: "var(--muted)", borderColor: "transparent" }}>
              {n.label}
            </a>
          );
        })}
      </div>
    </nav>
  );
}

function ValueMeter({ v, max }: { v: number; max: number }) {
  // Value per coin, relative to the best rec in this list (the units are arbitrary lift-per-1k-coins).
  const w = Math.max(4, Math.min(100, (v / (max || 1)) * 100));
  return <div className="meter w-full"><i style={{ width: `${w}%`, background: "var(--gold)" }} /></div>;
}

const SHORT_COST: Record<string, string> = { coins: "coins", power_points: "PP", credits: "credits" };
const costText = (cost: Record<string, number>) =>
  Object.entries(cost).filter(([, n]) => n > 0).map(([k, n]) => `${num(n)} ${SHORT_COST[k] || k}`).join(" · ");

function CostChips({ cost, gate, steps, estimated }: {
  cost: Record<string, number>; gate: string | null; steps: PurchaseStep[]; estimated: boolean;
}) {
  const entries = Object.entries(cost).filter(([, n]) => n > 0);
  const isPackage = steps.length > 1;
  if (!entries.length && !gate && !isPackage) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5 mt-2">
      {entries.map(([k, n]) => {
        const approx = estimated && k === "credits";   // the nominal is only ever a credit price
        return (
          <span key={k} className="mono text-[10px] px-1.5 py-0.5 border border-[var(--line)] text-[var(--muted)] tabular-nums"
            title={(isPackage ? "Total for the whole package below. " : "") + (approx ? "Approximate — no known price for this rarity." : "")}>
            {approx && <span className="text-[var(--dim)]">~</span>}{num(n)} {COST_LABEL[k] || k}
          </span>
        );
      })}
      {gate && (
        <span className="mono text-[10px] px-1.5 py-0.5 border tabular-nums"
          style={{ borderColor: "color-mix(in srgb, var(--gold) 45%, transparent)", color: "var(--gold)" }}
          title="Needs a higher power level first — the climb is priced into the cost.">
          ▲ {gate}
        </span>
      )}
      {isPackage && (
        <span className="mono text-[10px] text-[var(--dim)] tabular-nums w-full leading-relaxed"
          title="Everything this purchase needs, in order — the prices above are the total of these steps.">
          {steps.map((st, i) => {
            const c = costText(st.cost);
            return (
              <Fragment key={i}>
                {i > 0 && <span className="text-[var(--dim)]"> → </span>}
                <span className="inline-block">
                  <span style={{ color: KIND[st.kind]?.color || "var(--muted)" }}>{KIND[st.kind]?.glyph}</span>{" "}
                  <span className="text-[var(--muted)]">{st.label}</span>
                  {c && <span> ({c})</span>}
                </span>
              </Fragment>
            );
          })}
        </span>
      )}
    </div>
  );
}

function RecCard({ r, rank, b, max }: { r: PurchaseRec; rank: number; b?: Brawler; max: number }) {
  const kind = KIND[r.kind];
  const conf = CONF[r.confidence];
  return (
    <div className="card-rec panel flex gap-3 p-3 anim-rise" style={{ "--glow": kind.color } as React.CSSProperties}>
      <div className="mono text-[13px] tabular-nums text-[var(--dim)] w-5 shrink-0 pt-1 text-right">{rank}</div>
      {b
        ? <img src={b.image_url} alt={b.name} title={b.name} width={48} height={48}
            className="shrink-0 object-cover self-start"
            style={{ width: 48, height: 48, border: `2px solid ${kind.color}` }} />
        : <div className="shrink-0 self-start" style={{ width: 48, height: 48, background: "var(--panel2)", border: "2px solid var(--line)" }} />}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 flex-wrap leading-tight">
          <span className="mono text-[8px] px-1.5 py-0.5 font-bold tracking-[0.08em] shrink-0"
            style={{ background: kind.color, color: "#0a0a0c" }}>{kind.glyph} {kind.label}</span>
          <span className="text-[14px] font-bold text-[var(--text)] truncate">{r.brawler_name}</span>
          {r.item_name && <span className="mono text-[11px] text-[var(--muted)] truncate">{r.item_name}</span>}
          <span className="mono text-[9px] px-1 py-0.5 ml-auto shrink-0" style={{ color: conf.color }}
            title={conf.title}>{conf.label}</span>
        </div>
        <p className="text-[12px] text-[var(--muted)] mt-1.5 leading-snug">{r.rationale}</p>
        <div className="flex items-center gap-2 mt-2">
          <span className="mono text-[9px] tracking-[0.12em] text-[var(--dim)] shrink-0"
            title="Win-rate value per coin spent, prerequisites included — relative to the best rec in this list">VALUE / COIN</span>
          <ValueMeter v={r.value_score} max={max} />
          <span className="mono text-[10px] tabular-nums shrink-0" style={{ color: "var(--gold)" }}
            title="The brawler's smoothed win rate across the ranked map pool">
            WR {pct(r.meta_winrate)}
          </span>
        </div>
        <CostChips cost={r.cost} gate={r.gate} steps={r.steps ?? []} estimated={!!r.cost_estimated} />
      </div>
    </div>
  );
}

function TagBar({ tag, setTag, onLoad, onClear, loading }: {
  tag: string; setTag: (s: string) => void; onLoad: () => void; onClear: () => void; loading: boolean;
}) {
  return (
    <div className="panel px-3 py-2.5 mb-4 flex flex-wrap items-center gap-x-3 gap-y-2">
      <span className="label">◇ Player</span>
      <form className="flex items-center gap-1.5" onSubmit={(e) => { e.preventDefault(); onLoad(); }}>
        <div className="relative flex items-center">
          <input value={tag} onChange={(e) => setTag(e.target.value.toUpperCase())}
            id="bs-player-tag" name="bs-player-tag" autoComplete="on"
            autoCapitalize="characters" spellCheck={false} enterKeyHint="search"
            placeholder="#GZ95SFSKJ3"
            className="mono bg-[var(--panel2)] border border-[var(--line)] pl-2.5 pr-7 py-1.5 text-[13px] w-44 outline-none focus:border-[var(--accent)] ctl" />
          {tag && (
            <button type="button" onClick={onClear} aria-label="Forget saved tag" title="Forget saved tag"
              className="absolute right-1.5 grid place-items-center w-5 h-5 leading-none text-[var(--muted)] hover:text-[var(--red)] ctl">✕</button>
          )}
        </div>
        <button type="submit" disabled={loading || !tag.trim()} className="seg px-3 py-1.5 disabled:opacity-40">
          {loading ? "…" : "LOAD ↵"}
        </button>
      </form>
      <span className="mono text-[10px] text-[var(--dim)] ml-auto hidden sm:inline">
        find your tag in-game under your profile
      </span>
    </div>
  );
}

function SkeletonList() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="panel flex gap-3 p-3">
          <div className="skeleton w-12 h-12 shrink-0" />
          <div className="flex-1 space-y-2 pt-1">
            <div className="skeleton h-3 w-1/3" />
            <div className="skeleton h-2.5 w-2/3" />
            <div className="skeleton h-2 w-1/2" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function PurchaseAdvisor() {
  const [byId, setById] = useState<Map<number, Brawler>>(new Map());
  const [tag, setTag] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<"idle" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [ownedCount, setOwnedCount] = useState(0);
  const [recs, setRecs] = useState<PurchaseRec[]>([]);
  const [rank, setRank] = useState<RankInfo | null>(null);
  const [floor, setFloor] = useState<number>(11);
  const [floorChoice, setFloorChoice] = useState<FloorChoice>("auto");
  const [filter, setFilter] = useState<"all" | PurchaseKind>("all");
  // The loaded roster is kept so a floor change only re-scores (one cheap call to the public
  // backend) instead of re-fetching the roster through the tunnel.
  const rosterRef = useRef<{ owned: OwnedBrawler[]; tag: string; name: string; rank: RankInfo | null } | null>(null);
  const reqId = useRef(0);

  // Every fetch sequence carries an explicit request id; a newer sequence (new tag, new floor, or
  // clear) invalidates older ones so a slow response can't overwrite a newer state. `score` takes
  // the id from its caller rather than bumping it, so `load` → `score` is ONE sequence.
  const score = useCallback(async (choice: FloorChoice, mine: number) => {
    const r = rosterRef.current;
    if (!r) return;
    setLoading(true);
    try {
      const bracket = r.rank?.found ? r.rank.bracket : null;
      const res = await getPurchases(r.owned, r.tag, r.name, TOP, bracket, MIN_PER_KIND,
                                     choice === "auto" ? null : choice);
      if (mine !== reqId.current) return;
      setFloor(res.power_floor ?? 11);
      setRecs(res.recommendations); setError(null); setStatus("ready");
    } catch (e) {
      if (mine !== reqId.current) return;
      // A failed (re-)score is a public-API hiccup, not a roster outage: keep whatever list is
      // showing and surface the error inline so the floor chips stay usable.
      setError(String(e));
      setStatus((prev) => (prev === "ready" ? "ready" : "error"));
    } finally {
      if (mine === reqId.current) setLoading(false);
    }
  }, []);

  const load = useCallback(async (rawTag: string, choice: FloorChoice) => {
    const t = rawTag.trim();
    if (!t) return;
    const mine = ++reqId.current;
    setLoading(true); setError(null); setRecs([]); setStatus("idle");   // new account ⇒ skeleton, not the old list
    try {
      // Roster + Ranked bracket in parallel (both via the keyed tunnel). The bracket sets the power
      // floor that decides which owned brawlers are fieldable; if the lookup fails the backend
      // assumes the stricter Power 11 and the user can pin the floor themselves.
      const [roster, rankInfo] = await Promise.all([
        getRoster(t),
        getRank(t).catch((): RankInfo | null => null),
      ]);
      if (mine !== reqId.current) return;
      if (!roster.loaded) {
        setStatus("error"); setError(roster.error || "roster unavailable"); setRecs([]);
        return;
      }
      localStorage.setItem("bsdraft.tag", roster.tag || t);
      rosterRef.current = { owned: roster.owned, tag: roster.tag || t, name: roster.name, rank: rankInfo };
      setName(roster.name); setOwnedCount(roster.owned.length); setRank(rankInfo);
      await score(choice, mine);
    } catch (e) {
      if (mine !== reqId.current) return;
      setStatus("error"); setError(String(e)); setRecs([]);
    } finally {
      if (mine === reqId.current) setLoading(false);
    }
  }, [score]);

  const chooseFloor = useCallback((choice: FloorChoice) => {
    setFloorChoice(choice);
    localStorage.setItem(FLOOR_KEY, String(choice));
    if (rosterRef.current) score(choice, ++reqId.current);
  }, [score]);

  // Load the reference (brawler art/names) and auto-run for a previously saved tag.
  useEffect(() => {
    let live = true;
    getReference().then((ref) => {
      if (!live) return;
      setById(new Map(ref.brawlers.map((b) => [b.id, b])));
    }).catch(() => {});
    const choice = readFloorChoice();
    setFloorChoice(choice);
    const saved = typeof window !== "undefined" ? localStorage.getItem("bsdraft.tag") : null;
    if (saved) { setTag(saved); load(saved, choice); }
    return () => { live = false; };
  }, [load]);

  const clear = useCallback(() => {
    reqId.current++;                       // invalidates any in-flight load/score …
    rosterRef.current = null;
    setLoading(false);                     // … whose finally would otherwise never clear this
    setTag(""); setRecs([]); setStatus("idle"); setError(null); setName(""); setRank(null); setFilter("all");
    localStorage.removeItem("bsdraft.tag");
  }, []);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const r of recs) c[r.kind] = (c[r.kind] || 0) + 1;
    return c;
  }, [recs]);
  // A kind filter that no longer matches anything (floor flipped, new account) falls back to ALL
  // rather than a blank list under a hidden chip.
  const effFilter: "all" | PurchaseKind = filter !== "all" && !(counts[filter] || 0) ? "all" : filter;
  const shown = useMemo(() => (effFilter === "all" ? recs : recs.filter((r) => r.kind === effFilter)), [recs, effFilter]);
  const maxScore = useMemo(() => recs.reduce((m, r) => Math.max(m, r.value_score), 0), [recs]);

  const body = useMemo(() => {
    if (loading && !recs.length) return <SkeletonList />;
    if (status === "error")
      return (
        <div className="panel p-5 text-center" style={{ borderColor: "color-mix(in srgb, var(--red) 40%, transparent)" }}>
          <div className="mono text-[13px]" style={{ color: "var(--red)" }}>⚠ {rosterFailReason(error || "")}</div>
          <div className="mono text-[10px] text-[var(--dim)] mt-2" title={error || undefined}>
            Your roster loads live from Supercell, so this needs the roster service to be online.
          </div>
        </div>
      );
    if (status === "ready" && !recs.length)
      return (
        <div className="panel p-6 text-center">
          <div className="text-[15px] font-bold" style={{ color: "var(--green)" }}>◆ Fully maxed</div>
          <p className="text-[12px] text-[var(--muted)] mt-2">
            Nothing left to buy on {name || "this account"} that we'd rank — every meta brawler you own is built out.
          </p>
        </div>
      );
    if (status === "ready") {
      const bracket = rank?.found ? rank.bracket : null;
      const autoFloor = floorForBracket(bracket);
      const floorOpts: Array<{ key: FloorChoice; label: string; title: string }> = [
        { key: "auto",
          label: bracket ? `AUTO · ${bracket.toUpperCase()} → P${autoFloor}` : "AUTO · RANK UNKNOWN → P11",
          title: bracket
            ? `Your live Ranked tier resolved to ${bracket}, where brawlers below Power ${autoFloor} can't be selected`
            : "Couldn't resolve your Ranked tier (not placed this season?), so the stricter Power 11 floor is assumed — pin P9 if you play below Mythic" },
        { key: 9, label: "P9 FLOOR", title: "Bronze through Diamond: brawlers need Power 9 to be selected" },
        { key: 11, label: "P11 FLOOR", title: "Mythic and up: brawlers need Power 11 to be selected — pin this if you're planning for Mythic" },
      ];
      return (
        <>
          <div className="mono text-[10px] text-[var(--dim)] mb-2 flex flex-wrap gap-x-1.5 gap-y-1 items-center">
            <span>▸ analyzed {ownedCount} owned brawler{ownedCount === 1 ? "" : "s"}
              {name && <> on <span className="text-[var(--muted)]">{name}</span></>}</span>
            <span>·</span>
            <span>ranked by value per coin, prerequisites priced in</span>
          </div>
          <div className="flex flex-wrap items-center gap-1 mb-2" role="group" aria-label="Ranked power floor">
            <span className="mono text-[9px] tracking-[0.12em] text-[var(--dim)] mr-1"
              title={`Ranked blocks any brawler below the floor — anything you own under Power ${floor} is priced as a climb first, and no item on it is recommended until then`}>
              FLOOR · P{floor}
            </span>
            {floorOpts.map((o) => {
              const on = floorChoice === o.key;
              return (
                <button key={String(o.key)} type="button" aria-pressed={on} title={o.title} disabled={loading}
                  onClick={() => chooseFloor(o.key)}
                  className="mono text-[10px] tracking-[0.06em] px-2 py-1 border ctl tabular-nums disabled:opacity-50"
                  style={on
                    ? { color: "var(--text)", borderColor: "var(--gold)", boxShadow: "inset 0 0 0 1px color-mix(in srgb, var(--gold) 40%, transparent)" }
                    : { color: "var(--muted)", borderColor: "var(--line)" }}>
                  {o.label}
                </button>
              );
            })}
            {loading && <span className="mono text-[10px] text-[var(--dim)] ml-1" aria-live="polite">re-scoring…</span>}
            {error && !loading && (
              <span className="mono text-[10px] ml-1" style={{ color: "var(--red)" }} role="alert" title={error}>
                ⚠ {rosterFailReason(error)}
              </span>
            )}
          </div>
          <div className="flex flex-wrap gap-1 mb-3" role="group" aria-label="Filter by purchase kind">
            {FILTERS.map((f) => {
              const n = f.key === "all" ? recs.length : (counts[f.key] || 0);
              const on = effFilter === f.key;
              if (f.key !== "all" && !n) return null;
              return (
                <button key={f.key} type="button" aria-pressed={on} onClick={() => setFilter(f.key)}
                  className="mono text-[10px] tracking-[0.06em] px-2 py-1 border ctl tabular-nums"
                  style={on
                    ? { color: "var(--text)", borderColor: "var(--accent)", boxShadow: "inset 0 0 0 1px color-mix(in srgb, var(--accent) 40%, transparent)" }
                    : { color: "var(--muted)", borderColor: "var(--line)" }}>
                  {f.label} <span className="text-[var(--dim)]">{n}</span>
                </button>
              );
            })}
          </div>
          <div className="space-y-2">
            {shown.map((r) => {
              const i = recs.indexOf(r);
              return <RecCard key={`${r.brawler_id}-${r.kind}-${r.item_id ?? i}`} r={r} rank={i + 1} b={byId.get(r.brawler_id)} max={maxScore} />;
            })}
          </div>
        </>
      );
    }
    // idle
    return (
      <div className="panel p-6 text-center">
        <div className="text-[15px] font-bold text-[var(--text)]">Enter your player tag to begin</div>
        <p className="text-[12px] text-[var(--muted)] mt-2 max-w-md mx-auto leading-relaxed">
          We read your account's owned brawlers, power levels, loadouts, and Ranked tier, then rank
          the purchases that buy the most ranked win rate per coin — a first gadget on a maxed meta
          brawler before a long power climb, and never an item on a brawler you can't field yet.
        </p>
      </div>
    );
  }, [loading, status, error, recs, shown, name, ownedCount, byId, rank, floor, floorChoice, chooseFloor, effFilter, counts, maxScore]);

  return (
    <div className="p-3 md:p-5 max-w-3xl mx-auto w-full">
      <DocNav current="/purchases" />

      <header className="mb-6">
        <div className="label mb-3" style={{ color: "var(--accent)" }}>▸ UPGRADE PLANNER</div>
        <h1 className="display text-[clamp(1.8rem,5vw,2.8rem)] mb-3">What to upgrade next</h1>
        <p className="text-[14px] leading-relaxed text-[var(--muted)] max-w-2xl">
          Your most efficient next purchases, ranked by how much ranked win rate they buy per coin —
          with every prerequisite priced in: the power climb to your bracket's floor, a first build
          on an unbuilt brawler, the unlock. We read what you own, not what you can afford.
        </p>
      </header>

      <TagBar tag={tag} setTag={setTag} onLoad={() => load(tag, floorChoice)} onClear={clear} loading={loading} />
      {body}
    </div>
  );
}
