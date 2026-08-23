"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Brawler, PickRec, BanRec, Reference, RecommendResponse, Warning, RosterResponse, GamePlan, Health, Meta, RankInfo, TopPick,
  LoadoutResponse, LoadoutItem, OwnedBrawler,
  getReference, getRoster, recommend, getHealth, getMeta, getRank, getTopPicks, getLoadout,
} from "@/lib/api";
import AdSlot from "@/components/AdSlot";
import Logo from "@/components/Logo";

const CLASS_COLOR: Record<string, string> = {
  Tank: "#e0566f", Assassin: "#b15be0", Controller: "#3b82f6", Marksman: "#3ec46d",
  Support: "#e8c34a", "Damage Dealer": "#e8843a", Artillery: "#39c3c0", Unclassified: "#6b7280",
};
const CLASS_SHORT: Record<string, string> = {
  Tank: "TANK", Assassin: "ASSN", Controller: "CTRL", Marksman: "MARK",
  Support: "SUPP", "Damage Dealer": "DMG", Artillery: "ARTY", Unclassified: "UNCL",
};
// Portrait-outline colors matched to the in-game rarity borders (hexes from the brawlify
// catalog in data/reference/brawlers.json). Ultra Legendary doesn't use its flat hex for the
// border — it gets the animated prismatic ring (.ultra-ring), like in-game Sirius/Kaze.
const RARITY_COLOR: Record<string, string> = {
  Common: "#b9eaff", Rare: "#68fd58", "Super Rare": "#5ab3ff",
  Epic: "#d850ff", Mythic: "#fe5e72", Legendary: "#fff11e", "Ultra Legendary": "#e1fb2a",
};
const SEV_COLOR: Record<string, string> = { critical: "#ff3b30", warn: "#e8c34a", info: "#5aa0ff" };
// Ranked tier accent colors, low → high (matched to the in-game rank emblems).
const BRACKET_COLOR: Record<string, string> = {
  Bronze: "#c8814b", Silver: "#a7b6c8", Gold: "#e8c34a", Diamond: "#45d4e8",
  Mythic: "#b15be0", Legendary: "#e0566f", Masters: "#b9533a", Pro: "#34c759",
};
const bracketColor = (b?: string | null) => (b && BRACKET_COLOR[b]) || "#e8c34a";

const TIER_SUB: Record<string, number> = { I: 1, II: 2, III: 3 };
function splitTier(label: string): { name: string; sub: number } {
  const parts = label.trim().split(/\s+/);
  const last = parts[parts.length - 1];
  return TIER_SUB[last] ? { name: parts.slice(0, -1).join(" "), sub: TIER_SUB[last] } : { name: label, sub: 0 };
}

// stacked upward chevrons (military-rank insignia) marking the sub-tier 1–3
function TierChevrons({ n }: { n: number }) {
  return (
    <span className="inline-flex flex-col items-center" style={{ gap: 1 }} aria-hidden="true">
      {Array.from({ length: n }).map((_, i) => (
        <svg key={i} width="10" height="4.5" viewBox="0 0 11 5">
          <polyline points="1.5,4 5.5,1.4 9.5,4" fill="none" stroke="currentColor"
            strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ))}
    </span>
  );
}

type Zone = "ban" | "our" | "their";
type Slot = { zone: Zone; index: number };
type Step =
  | { kind: "ban"; slot: Slot; index: number }
  | { kind: "pick"; side: "us" | "them"; slot: Slot; n: number }
  | { kind: "done" };

const BAN_N = 6, TEAM_N = 3, PICK_N = 6;
const ROSTER_POLL_MS = 5 * 60 * 1000; // re-check the player's roster/inventory every 5 min
const PICK_ORDER = [0, 1, 1, 0, 0, 1]; // 1-2-2-1 snake; 0 = first-pick team
// Ranks that draft "blind": a shared ban phase, then you pick your own team without ever
// seeing the enemy's picks. In Brawl Stars this is Diamond and below.
const BLIND_PICK_BRACKETS = new Set(["Bronze", "Silver", "Gold", "Diamond"]);
// Ranked never boosts brawlers to a fixed power — they play at their real level, and each bracket
// hard-blocks selecting a brawler below a per-brawler floor: Power 9 through Diamond, Power 11 from
// Mythic up. So an owned brawler under the floor can't be fielded, only the ones at/above it. (The
// season's free "boosted" brawlers arrive at Power 11 and clear any floor.)
const P11_BRACKETS = new Set(["Mythic", "Legendary", "Masters", "Pro"]);
const minPowerForBracket = (b?: string | null) => (b && P11_BRACKETS.has(b) ? 11 : 9);
const pct = (x: number) => `${Math.round(x * 100)}%`;
const two = (x: number) => Math.round(x * 100);
const cssVars = (vars: Record<string, string | number | undefined>) => vars as React.CSSProperties;
const scoreColor = (v: number) => (v >= 0.5 ? "var(--green)" : "var(--gold)");

function pickSlotSequence(wePickFirst: boolean): { side: "us" | "them"; zone: Zone; index: number }[] {
  const seq: { side: "us" | "them"; zone: Zone; index: number }[] = [];
  const counts = { us: 0, them: 0 };
  for (const team of PICK_ORDER) {
    const side: "us" | "them" = (team === 0) === wePickFirst ? "us" : "them";
    const zone: Zone = side === "us" ? "our" : "their";
    seq.push({ side, zone, index: counts[side] });
    counts[side]++;
  }
  return seq;
}

// --- one-line rationale, synthesized client-side from the scored signals ---
function pickReason(r: PickRec): string {
  // gaps are loadout warnings ("no star power"), shown as tags — not a positive reason.
  const c: [number, string][] = [];
  if (r.counter != null) c.push([r.counter - 0.5, "counters the enemy"]);
  if (r.synergy != null) c.push([r.synergy - 0.5, "synergy with your team"]);
  c.push([r.map_winrate - 0.5, "top winrate on this map"]);
  if (r.personal_winrate != null && (r.personal_games ?? 0) >= 3) c.push([(r.personal_winrate - 0.5) * 1.1, "your proven pick"]);
  if (r.mastery != null) c.push([(r.mastery - 0.5) * 0.7, "high mastery"]);
  if (r.win_prob != null) c.push([(r.win_prob - 0.5) * 0.8, "the model favors it"]);
  c.sort((a, b) => b[0] - a[0]);
  return c[0] && c[0][0] > 0.004 ? c[0][1] : "balanced pick here";
}
// A ban's worth is what it denies *given the rest of the ban set* — so the line explains the
// projection (who replaces it, whether you'd have taken it), not the brawler's raw stat line.
// Falls back to the stat line when the backend couldn't project (no model).
function banReason(r: BanRec): string {
  if (r.ban_value == null) {
    const strong = r.map_winrate >= 0.53, popular = r.use_rate >= 0.2;
    if (strong && popular) return "high pick rate, wins here";
    if (popular) return "very popular pick";
    if (strong) return "strong on this map";
    return "known meta threat";
  }
  if (r.self_deny) return "you'd take this yourself — pick it, don't ban it";
  if (r.ban_value <= 0.0005)
    return r.replacement ? `replaceable — ${r.replacement} covers it` : "barely dents their draft";
  return r.replacement ? `cuts their draft · next best ${r.replacement}` : "nothing left to replace it";
}
// Swing is a win-probability delta, shown in points. Small numbers are the honest scale for one
// ban, so give it a sign and one decimal rather than rounding it into a flat "0%".
function swingLabel(v: number): string {
  return `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)}`;
}
const SWING_HINT = "projected gain in your win probability if this brawler is banned, given the bans already placed and who picks first";
// present signals for a pick, in a stable display order
function pickSignals(r: PickRec): { k: string; v: number }[] {
  const s: { k: string; v: number }[] = [{ k: "MAP", v: r.map_winrate }];
  if (r.synergy != null) s.push({ k: "SYN", v: r.synergy });
  if (r.counter != null) s.push({ k: "CTR", v: r.counter });
  s.push({ k: "ROLE", v: r.role_fit });
  if (r.win_prob != null) s.push({ k: "MDL", v: r.win_prob });
  if (r.mastery != null) s.push({ k: "MST", v: r.mastery });
  if (r.personal_winrate != null) s.push({ k: "YOU", v: r.personal_winrate });
  return s;
}
// Loadout gaps ("no star power") abbreviated for the half-width columns, which can't fit the full
// string. The row tooltip keeps the long form; the score now docks for them via the readiness deficit.
const GAP_SHORT: Record<string, string> = {
  "no star power": "SP",
  "no gadget": "GDG",
  "no hypercharge": "HC",
};
function gapTags(gaps: string[] | undefined): string[] {
  return (gaps || []).map((g) => GAP_SHORT[g] || g.replace(/^no /, "").toUpperCase());
}

// eased number that animates toward `target` — a readout "locking in"
function useCountUp(target: number, duration = 420) {
  const [val, setVal] = useState(0);
  const fromRef = useRef(0);
  const rafRef = useRef(0);
  useEffect(() => {
    const from = fromRef.current;
    let startTs = 0;
    cancelAnimationFrame(rafRef.current);
    const tick = (now: number) => {
      if (!startTs) startTs = now;
      const t = Math.min(1, (now - startTs) / duration);
      const e = 1 - Math.pow(1 - t, 3);
      setVal(from + (target - from) * e);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
      else fromRef.current = target;
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, duration]);
  return val;
}
function CountUp({ value, className }: { value: number; className?: string }) {
  const v = useCountUp(value);
  return <span className={className}>{Math.round(v).toLocaleString()}</span>;
}

// `fluid` fills the parent's width instead of a fixed box — kept square by aspect-ratio, since the
// source portraits aren't square and object-cover crops them.
function Avatar({ b, size = 56, dim, ring, active, fluid }: { b?: Brawler; size?: number; dim?: boolean; ring?: string; active?: boolean; fluid?: boolean }) {
  const border = ring || (b ? RARITY_COLOR[b.rarity] || "#26303f" : "#26303f");
  // An explicit ring (the active-slot accent) beats the rarity treatment. The ultra border
  // lives entirely in .ultra-ring: an inline `border` shorthand would reset its border-image.
  const ultra = !ring && b?.rarity === "Ultra Legendary";
  const cls = `object-cover ${fluid ? "block" : ""} ${active ? "slot-active" : ""} ${ultra ? "ultra-ring" : ""}`;
  const box = fluid ? { width: "100%", aspectRatio: "1 / 1" } : { width: `${size}px`, height: `${size}px` };
  if (!b) return <div style={{ ...box, borderColor: border }} className="bg-[var(--panel2)] border" />;
  return (
    <img src={b.image_url} alt={b.name} title={b.name} width={size} height={size}
      className={cls}
      style={cssVars({ ...box, opacity: dim ? "0.3" : "1", border: ultra ? undefined : `1px solid ${border}`, "--ring": border })} />
  );
}

// thin squared telemetry meter with a key + numeric readout
function Meter({ k, v }: { k: string; v: number }) {
  const w = Math.max(3, Math.min(100, ((v - 0.35) / 0.30) * 100));
  const good = v >= 0.5;
  return (
    <div className="flex items-center gap-2">
      <span className="mono w-9 shrink-0 text-[9px] tracking-[0.12em] text-[var(--dim)]">{k}</span>
      <div className="meter flex-1"><i style={{ width: `${w}%`, background: good ? "var(--green)" : "var(--red)" }} /></div>
      <span className="mono w-6 text-right text-[10px] tabular-nums" style={{ color: good ? "var(--green)" : "var(--muted)" }}>{two(v)}</span>
    </div>
  );
}

// compact inline "CTR 61" signal chips for the ranked list rows
function SigLine({ sig }: { sig: { k: string; v: number }[] }) {
  return (
    <span className="mono text-[10px] tabular-nums text-[var(--dim)]">
      {sig.map((s, i) => (
        <span key={s.k}>
          {i > 0 && <span className="text-[var(--line-strong)]"> · </span>}
          {s.k} <span style={{ color: s.v >= 0.5 ? "var(--green)" : "var(--muted)" }}>{two(s.v)}</span>
        </span>
      ))}
    </span>
  );
}

function FirstPickToggle({ wePickFirst, onToggle }: { wePickFirst: boolean; onToggle: () => void }) {
  const c = wePickFirst ? { col: "var(--blue)", side: "YOU" } : { col: "var(--red)", side: "ENEMY" };
  return (
    <button onClick={onToggle}
      className="seg ml-auto shrink-0 inline-flex items-center gap-2 px-2.5 py-2" data-on="true"
      style={cssVars({ "--seg-c": c.col })}
      title="Who picks first · click to switch">
      <span className="text-[10px] tracking-[0.16em]">FIRST PICK</span>
      <span className="text-[11px] font-bold px-2 py-0.5" style={{ background: c.col, color: "#0a0a0c" }}>{c.side}</span>
    </button>
  );
}

// A checkbox sitting directly under one "Your team" slot: check the one that is YOU. Personalization
// — owned+free filtering, mastery and your win-rates — then applies only to that seat's turn;
// teammate picks stay on the pure meta. Single-select: checking one clears the others.
function SeatCheck({ checked, disabled, onToggle, seat }: {
  checked: boolean; disabled: boolean; onToggle: () => void; seat: number;
}) {
  // The visible box is an inner <span>: a native <button>'s own background-color is unreliable
  // (the UA button appearance suppresses it), whereas a span paints its fill dependably.
  return (
    <button type="button" role="checkbox" aria-checked={checked} disabled={disabled}
      onClick={onToggle} aria-label={`Mark pick ${seat} as you`}
      title={disabled ? "load your tag to personalize"
        : checked ? "This pick is you — click to clear" : `Check if pick #${seat} is you`}
      className="grid place-items-center border-0 bg-transparent p-0 leading-none disabled:opacity-30 disabled:cursor-not-allowed">
      <span className="grid place-items-center w-[18px] h-[18px] border transition-colors"
        style={{ borderColor: checked ? "var(--gold)" : "var(--line-strong)",
                 background: checked ? "var(--gold)" : "var(--panel2)" }}>
        {checked && <span className="text-[11px] font-bold leading-none" style={{ color: "#0a0a0c" }}>✓</span>}
      </span>
    </button>
  );
}

// The backend hands back raw operator-facing text (`str(e)` off the Supercell client — "HTTP 403:
// auth/IP error — check the token and that this machine's public IP…"). Translate to something a
// visitor can act on, and say plainly when the fault is ours, not their tag. Raw text goes in the
// title attribute so the operator can still read it on hover.
function rosterFailReason(error: string): string {
  const e = error.toLowerCase();
  if (e.includes("404") || e.includes("not found")) return "no player with that tag — check it and load again";
  if (e.includes("429")) return "roster service is busy — retrying shortly";
  if (e.includes("403") || e.includes("auth/ip") || e.includes("no api token"))
    return "roster service is down — personalization is off";
  return "couldn't reach the roster service — personalization is off";
}

// One-line status under the seat checkboxes: what to do, why it's unavailable, or who's being
// personalized.
function SeatHint({ ready, chosen, name, active, hasTag, error }: {
  ready: boolean; chosen: boolean; name?: string; active: boolean;
  hasTag: boolean; error?: string | null;
}) {
  if (!ready) {
    if (!hasTag)
      return <div className="mono text-[9px] text-[var(--dim)] mt-2.5">◦ check the box under your pick — load your tag first</div>;
    if (error)
      return (
        <div className="mono text-[9px] mt-2.5" style={{ color: "var(--red)" }} title={error}>
          ⚠ {rosterFailReason(error)}
        </div>
      );
    return <div className="mono text-[9px] text-[var(--dim)] mt-2.5">◦ loading your roster…</div>;
  }
  if (!chosen)
    return <div className="mono text-[9px] text-[var(--dim)] mt-2.5">◦ check the box under your pick to personalize it</div>;
  return (
    <div className="mono text-[10px] mt-2.5 inline-flex items-center gap-1" style={{ color: "var(--gold)" }}
      title={active ? "personalizing this pick now" : "personalizes when it's your pick's turn"}>
      ◈ {(name || "YOU").toUpperCase()} · YOUR PICK{active && " · NOW"}
    </div>
  );
}

// ===== Loadout hover popover — which gadget / star power / gear to run on a drafted brawler.
// Effect-based advice from /api/loadout; on the user's own seat it's filtered to what they own.
const normName = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");

function ItemRow({ it, marked, locked, tag, tagTitle, compFlip }: {
  it: { name: string; image_url?: string; effect?: string; description?: string; comp_why?: string[] };
  marked?: boolean; locked?: boolean; tag?: string; tagTitle?: string; compFlip?: boolean;
}) {
  return (
    <div className="flex gap-2 items-start py-[3px]" style={{ opacity: locked ? 0.5 : 1 }}>
      {it.image_url
        ? <img src={it.image_url} alt="" width={22} height={22} loading="lazy"
            className="mt-[1px] shrink-0" style={{ width: 22, height: 22, objectFit: "cover" }} />
        : <span className="mt-[1px] shrink-0 grid place-items-center text-[10px] text-[var(--dim)]"
            style={{ width: 22, height: 22, background: "var(--panel2)", border: "1px solid var(--line)" }}>⚙</span>}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5 flex-wrap leading-tight">
          <span className="text-[12px] font-semibold" style={{ color: marked ? "var(--gold)" : undefined }}>{it.name}</span>
          {tag && <span className="mono text-[9px] text-[var(--dim)]" title={tagTitle}>{tag}</span>}
          {marked && <span className="mono text-[8px] px-1 py-0.5 leading-none font-bold" style={{ background: "var(--gold)", color: "#0a0a0c" }}>★ PICK{compFlip ? " · COMP" : ""}</span>}
          {locked && <span className="mono text-[8px] px-1 leading-none border border-[var(--line)] text-[var(--dim)]">🔒 own it</span>}
          {(it.comp_why ?? []).map((c) => (
            <span key={c} className="mono text-[8px] px-1 py-0.5 leading-none border"
              style={c.startsWith("+")
                ? { borderColor: "var(--accent)", color: "var(--accent)" }
                : { borderColor: "var(--line)", color: "var(--dim)" }}>{c}</span>
          ))}
        </div>
        {(it.effect || it.description) && (
          <div className="mono text-[10px] text-[var(--muted)] leading-snug line-clamp-2">
            {it.effect && <span style={{ color: "var(--accent)" }}>{it.effect}</span>}
            {it.effect && it.description ? " · " : ""}{it.description}
          </div>
        )}
      </div>
    </div>
  );
}

function AccSection({ title, items, isMySeat, ownedIds }: {
  title: string; items: LoadoutItem[]; isMySeat: boolean; ownedIds: Set<number>;
}) {
  if (!items || items.length === 0) return null;
  const owns = (it: LoadoutItem) => it.id != null && ownedIds.has(it.id);
  const sorted = [...items].sort((a, b) =>
    (isMySeat ? Number(owns(b)) - Number(owns(a)) : 0) || b.fit - a.fit);
  // Marked "pick": the best owned item on your seat, else the best-fit item for the mode.
  const mark = isMySeat ? sorted.find(owns) : (sorted.find((it) => it.recommended) || sorted[0]);
  const noneOwned = isMySeat && !mark;
  return (
    <div className="mt-2 first:mt-0">
      <div className="label mb-0.5">{title}</div>
      {noneOwned && <div className="mono text-[10px] text-[var(--dim)]">you don&apos;t own a {title.toLowerCase()} here yet</div>}
      {sorted.map((it) => (
        <ItemRow key={it.id ?? it.name} it={it} marked={mark === it} locked={isMySeat && !owns(it)}
          // ·COMP badge only off-seat: the my-seat star is the client-side owned-max, so the
          // server's comp_flipped (about the unfiltered pool) would lie there. Chips still render.
          compFlip={!isMySeat && mark === it && !!it.comp_flipped} />
      ))}
    </div>
  );
}

function GearSection({ gears, isMySeat, ownedGears }: {
  gears: LoadoutItem[]; isMySeat: boolean; ownedGears: { id: number; name: string; level: number }[];
}) {
  if (!gears || gears.length === 0) return null;
  const byName = new Map(gears.map((g) => [normName(g.name), g]));
  if (isMySeat) {
    if (ownedGears.length) {
      const enriched = ownedGears
        .map((og) => ({ og, g: byName.get(normName(og.name)) }))
        .sort((a, b) => (b.g?.fit ?? 0) - (a.g?.fit ?? 0));
      // You run TWO gears in a match, so star the best two you own (fewer if you own fewer).
      const bestNames = new Set(enriched.slice(0, 2).map((e) => e.og.name));
      return (
        <div className="mt-2">
          <div className="label mb-0.5">GEARS · YOURS</div>
          {/* No level shown: Brawl Stars removed gear upgrade levels (Oct 2022); gears are now a flat
              purchase at full power, so the API's legacy `level` (always 3) is meaningless. */}
          {enriched.map(({ og, g }) => (
            <ItemRow key={og.id}
              it={{ name: og.name, effect: g?.effect, description: g?.description, comp_why: g?.comp_why }}
              marked={bestNames.has(og.name)} />
          ))}
        </div>
      );
    }
    return (
      <div className="mt-2">
        <div className="label mb-0.5">GEARS</div>
        <div className="mono text-[10px] text-[var(--dim)]">no gears built here — worth getting:</div>
        {gears.filter((g) => g.recommended).map((g) => <ItemRow key={g.name} it={g} locked />)}
      </div>
    );
  }
  return (
    <div className="mt-2">
      <div className="label mb-0.5">GEARS</div>
      {gears.filter((g) => g.recommended).map((g) => (
        <ItemRow key={g.name} it={g} marked compFlip={!!g.comp_flipped} />
      ))}
    </div>
  );
}

type SlotRect = { left: number; top: number; bottom: number; width: number };

function LoadoutPopover({ data, isMySeat, owned, accent, rect }: {
  data: LoadoutResponse | "loading" | undefined;
  isMySeat: boolean; owned?: OwnedBrawler; accent: string; rect: SlotRect;
}) {
  // Fixed, viewport-clamped positioning: centered under the slot but kept fully on-screen, and
  // flipped above the slot when there isn't room below (the your-team/enemy rows sit low on the page).
  const W = 292;
  const vw = typeof window !== "undefined" ? window.innerWidth : 1280;
  const vh = typeof window !== "undefined" ? window.innerHeight : 720;
  const left = Math.max(8, Math.min(rect.left + rect.width / 2 - W / 2, vw - W - 8));
  const spaceBelow = vh - rect.bottom;
  const spaceAbove = rect.top;
  const openUp = spaceBelow < 300 && spaceAbove > spaceBelow;
  const pos: React.CSSProperties = openUp
    ? { left, bottom: vh - rect.top + 8, maxHeight: Math.min(spaceAbove - 16, vh * 0.72) }
    : { left, top: rect.bottom + 8, maxHeight: Math.min(spaceBelow - 16, vh * 0.72) };
  return (
    <div onClick={(e) => e.stopPropagation()}
      className="fixed z-50 w-[292px] max-w-[92vw] panel p-3 text-left cursor-default anim-fade overflow-y-auto"
      style={{ ...pos, borderColor: "var(--line-strong)", boxShadow: "0 14px 34px rgba(0,0,0,0.55)" }}>
      {!data || data === "loading" ? (
        <div className="mono text-[11px] text-[var(--muted)]">Loading loadout…</div>
      ) : (
        <>
          <div className="flex items-center justify-between mb-2 pb-2 border-b border-[var(--line)]">
            <span className="label" style={{ color: accent }}>◈ {data.brawler_name || "Loadout"}</span>
            {isMySeat
              ? <span className="mono text-[9px]" style={{ color: "var(--gold)" }}>YOUR INVENTORY</span>
              : <span className="mono text-[9px] text-[var(--dim)]">{data.mode}</span>}
          </div>
          {!!data.comp_reads?.length && (
            <div className="mono text-[10px] text-[var(--dim)] -mt-1 mb-1.5">
              vs: <span style={{ color: "var(--accent)" }}>{data.comp_reads.join(" · ")}</span>
            </div>
          )}
          <AccSection title="GADGET" items={data.gadgets} isMySeat={isMySeat}
            ownedIds={new Set(owned?.owned_gadgets ?? [])} />
          <AccSection title="STAR POWER" items={data.star_powers} isMySeat={isMySeat}
            ownedIds={new Set(owned?.owned_star_powers ?? [])} />
          <GearSection gears={data.gears} isMySeat={isMySeat} ownedGears={owned?.owned_gears ?? []} />
          {data.note && (
            <div className="mono text-[9px] text-[var(--dim)] mt-2 pt-1.5 border-t border-[var(--line)] leading-snug">{data.note}</div>
          )}
        </>
      )}
    </div>
  );
}

function RankWidget({ tag, setTag, rankInfo, loading, onCheck, onClear }: {
  tag: string; setTag: (s: string) => void; rankInfo: RankInfo | null; loading: boolean;
  onCheck: () => void; onClear: () => void;
}) {
  return (
    <div className="panel px-3 py-2.5 mb-3 flex flex-wrap items-center gap-x-3 gap-y-2">
      <span className="label">◇ Player</span>
      <form className="flex items-center gap-1.5" onSubmit={(e) => { e.preventDefault(); onCheck(); }}>
        <div className="relative flex items-center">
          <input value={tag} onChange={(e) => setTag(e.target.value.toUpperCase())}
            id="bs-player-tag" name="bs-player-tag" autoComplete="on"
            autoCapitalize="characters" spellCheck={false} enterKeyHint="search"
            placeholder="#GZ95SFSKJ3"
            className="mono bg-[var(--panel2)] border border-[var(--line)] pl-2.5 pr-7 py-1.5 text-[13px] w-44 outline-none focus:border-[var(--accent)] ctl" />
          {tag && (
            <button type="button" onClick={onClear} aria-label="Forget saved tag" title="Forget saved tag"
              className="absolute right-1.5 grid place-items-center w-5 h-5 leading-none text-[var(--muted)] hover:text-[var(--red)] ctl">
              ✕
            </button>
          )}
        </div>
        <button type="submit" disabled={loading || !tag.trim()}
          className="seg px-3 py-1.5 disabled:opacity-40">
          {loading ? "…" : "LOAD ↵"}
        </button>
      </form>
      {rankInfo?.found && rankInfo.tier_label && (() => {
        const c = bracketColor(rankInfo.bracket);
        const { name, sub } = splitTier(rankInfo.tier_label!);
        return (
          <span className="mono ml-auto text-[12px] px-2.5 py-1 font-semibold inline-flex items-center gap-1.5 border"
            style={{ background: c + "18", color: c, borderColor: c + "66" }}
            aria-label={rankInfo.stale ? rankInfo.tier_label! + " (may be out of date)" : rankInfo.tier_label!}
            title={rankInfo.tier_label! + (rankInfo.source === "live"
              ? " · from a live lookup"
              : rankInfo.stale
                ? " · from our match data, and we couldn't reach the live check — if a new season just started this may be last season's tier"
                : " · from our match data")}>
            {name.toUpperCase()}
            {sub > 0 && <TierChevrons n={sub} />}
            {rankInfo.stale && <span className="opacity-60 font-normal" aria-hidden="true">?</span>}
          </span>
        );
      })()}
      {rankInfo && !rankInfo.found && (
        <span className="mono ml-auto text-[11px] text-[var(--muted)]">{(rankInfo.error || "tag not found").toUpperCase()}</span>
      )}
    </div>
  );
}

function MetaBanner({ meta }: { meta: Meta }) {
  const buffs = meta.shifts.filter((s) => s.kind === "buff").slice(0, 3).map((s) => s.name);
  const nerfs = meta.shifts.filter((s) => s.kind === "nerf").slice(0, 3).map((s) => s.name);
  const parts: string[] = [];
  if (meta.new_brawlers.length) parts.push(`NEW ${meta.new_brawlers.join(", ")}`);
  if (buffs.length) parts.push(`▲ ${buffs.join(", ")}`);
  if (nerfs.length) parts.push(`▼ ${nerfs.join(", ")}`);
  return (
    <div className="panel relative overflow-hidden px-3 py-2 mb-3 flex items-center gap-3"
      style={{ borderColor: "#e8c34a66" }}>
      <span className="absolute left-0 top-0 bottom-0 w-1" style={{ background: "var(--gold)" }} />
      <span className="label shrink-0" style={{ color: "var(--gold)" }}>▲ Meta shift</span>
      <span className="mono text-[11px] text-[var(--muted)] truncate">
        {parts.join("  ·  ").toUpperCase() || "RECENT BALANCE CHANGE"}
      </span>
      <span className="mono text-[10px] text-[var(--dim)] ml-auto shrink-0 hidden sm:inline">STATS CATCHING UP</span>
    </div>
  );
}

// The big command readout: whose turn, what phase, and the static per-action time budget.
// The 20s marker is the in-game draft window (a reference, not a running timer).
function StatusReadout({ step, tag, title, sub, accent, window }: {
  step: Step; tag: string; title: string; sub: string; accent: string; window: string;
}) {
  return (
    <div className="panel relative overflow-hidden anim-fade" style={{ borderColor: accent + "66" }}>
      <span className="absolute left-0 top-0 bottom-0 w-1 slot-active" style={cssVars({ "--ring": accent, background: accent })} />
      <div className="flex items-center gap-3 md:gap-4 pl-4 pr-3 py-2.5">
        <span className="mono text-[12px] px-2 py-1 border shrink-0 tabular-nums" style={{ borderColor: accent, color: accent }}>{tag}</span>
        <div className="min-w-0 flex-1">
          <div className="display text-lg md:text-xl leading-none" style={{ color: accent }}>{title}</div>
          <div className="label mt-1 truncate">{sub}</div>
        </div>
        {step.kind !== "done" && (
          <div className="text-right shrink-0 pl-2 border-l border-[var(--line)]"
            title="You get about 20 seconds per action in a ranked draft. This is that budget, not a live timer.">
            <div className="mono text-[9px] tracking-[0.15em] text-[var(--dim)]">{window}</div>
            <div className="mono font-bold text-2xl leading-none tabular-nums text-[var(--muted)]">
              20<span className="text-[12px] text-[var(--dim)]">S</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <span className="inline-block w-9 h-9 animate-spin"
      style={{ border: "2px solid var(--line)", borderTopColor: "var(--accent)" }} aria-hidden="true" />
  );
}

function BootScreen({ show, error, onRetry }: { show: boolean; error: string | null; onRetry: () => void }) {
  return (
    <div className="min-h-screen p-4 md:p-6 max-w-[1240px] mx-auto">
      <header className="flex items-center gap-2.5 mb-5">
        <Logo size={26} />
        <span className="brand-gradient text-lg tracking-tight">BRAWL DRAFT</span>
        <span className="label">// RANKED DRAFT CONSOLE</span>
      </header>
      <div className="min-h-[55vh] grid place-items-center">
        {error ? (
          <div className="panel p-6 text-center max-w-sm anim-fade">
            <div className="label mb-2" style={{ color: "var(--red)" }}>◇ LINK FAILURE</div>
            <div className="text-sm font-semibold text-[var(--text)] mb-1">Couldn&rsquo;t reach the draft server</div>
            <div className="mono text-[11px] text-[var(--muted)] mb-4 break-words">{error}</div>
            <button onClick={onRetry} className="seg px-4 py-2">RETRY ↵</button>
          </div>
        ) : show ? (
          <div className="text-center anim-fade">
            <Spinner />
            <div className="mt-4 mono text-[13px] font-semibold text-[var(--text)] caret">BOOTING DRAFT SERVER</div>
            <div className="mt-2 mono text-[11px] text-[var(--muted)] max-w-xs mx-auto">
              First hit can take ~30-45s on the free tier. It stays fast once warm.
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default function DraftBoard() {
  const [ref, setRef] = useState<Reference | null>(null);
  const [roster, setRoster] = useState<RosterResponse | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [tag, setTag] = useState("");
  const [rankInfo, setRankInfo] = useState<RankInfo | null>(null);
  const [rankLoading, setRankLoading] = useState(false);
  const [mapId, setMapId] = useState<number | null>(null);
  const [bans, setBans] = useState<(number | null)[]>(Array(BAN_N).fill(null));
  const [our, setOur] = useState<(number | null)[]>(Array(TEAM_N).fill(null));
  const [their, setTheir] = useState<(number | null)[]>(Array(TEAM_N).fill(null));
  const [wePickFirst, setWePickFirst] = useState(true);
  const [activeOverride, setActiveOverride] = useState<Slot | null>(null);
  const [solo, setSolo] = useState(true);
  const [recs, setRecs] = useState<RecommendResponse | null>(null);
  // Blind-pick dual columns only: the personalized pick list fetched alongside the general one.
  const [personalRecs, setPersonalRecs] = useState<RecommendResponse | null>(null);
  const [personalLoading, setPersonalLoading] = useState(false);
  const [personalErr, setPersonalErr] = useState<string | null>(null);
  const [topPicks, setTopPicks] = useState<TopPick[]>([]);
  const [railOk, setRailOk] = useState(true);
  const [warnings, setWarnings] = useState<Warning[]>([]);
  const [query, setQuery] = useState("");
  const [gridFocus, setGridFocus] = useState(0); // roving tab-stop index into the brawler grid (arrow-key nav)
  const [mySeat, setMySeat] = useState<number | null>(null); // which "our" slot is the user (in pick order)
  const [hoverSlot, setHoverSlot] = useState<{ zone: Zone; index: number; rect: SlotRect } | null>(null); // drafted slot under the cursor
  const [loadouts, setLoadouts] = useState<Record<string, LoadoutResponse | "loading">>({}); // cache: `${bid}:${mode}`
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [bootNonce, setBootNonce] = useState(0);
  const [slowBoot, setSlowBoot] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const focusSearch = () => searchRef.current?.focus({ preventScroll: true });

  useEffect(() => {
    let cancelled = false;
    const started = Date.now();
    (async () => {
      for (let attempt = 0; !cancelled; attempt++) {
        try {
          const r = await getReference();
          if (cancelled) return;
          const best = [...r.maps].filter((m) => m.games > 0).sort((a, b) => b.games - a.games)[0] || r.maps[0];
          setRef(r);
          if (best) setMapId(best.id);
          setErr(null);
          getHealth().then(setHealth).catch(() => {});
          getMeta().then(setMeta).catch(() => {});
          return;
        } catch (e) {
          if (cancelled) return;
          if (Date.now() - started > 90_000) { setErr(String(e)); return; }
          await new Promise((res) => setTimeout(res, Math.min(5000, 800 * (attempt + 1))));
        }
      }
    })();
    const savedTag = localStorage.getItem("bsdraft.tag");
    if (savedTag) {
      setTag(savedTag);
      getRank(savedTag).then(setRankInfo).catch(() => {});
    }
    const savedSeat = localStorage.getItem("bsdraft.myseat");  // null when never chosen — must stay null (personalization off by default)
    if (savedSeat != null) {
      const n = Number(savedSeat);
      if (Number.isInteger(n) && n >= 0 && n < TEAM_N) setMySeat(n);
    }
    return () => { cancelled = true; };
  }, [bootNonce]);

  useEffect(() => {
    if (ref || err) { setSlowBoot(false); return; }
    const t = setTimeout(() => setSlowBoot(true), 500);
    return () => clearTimeout(t);
  }, [ref, err, bootNonce]);

  // autofocus the keyboard-first placer once the board is live, so logging is instant
  useEffect(() => { if (ref) focusSearch(); }, [ref]);

  const rosterTag = rankInfo?.tag ?? null;

  useEffect(() => {
    // No tag → no personalization. Never poll tag-less: the backend used to answer a tag-less
    // request with the operator's own roster, leaking their identity to every visitor.
    if (!rosterTag) { setRoster(null); return; }
    // Different player → drop the previous one's roster up front, so the `cur.loaded` check in the
    // catch below can never mistake a stale roster for this tag's. Re-polls don't re-run this effect.
    setRoster(null);
    let cancelled = false;
    let last = 0;
    const pull = () => {
      last = Date.now();
      getRoster(rosterTag)
        .then((r) => { if (!cancelled) setRoster(r); })
        .catch((e) => {
          // Surface the failure instead of swallowing it: an upstream outage (the roster tunnel's
          // Supercell key 403ing on an IP rotation) used to present as inert seat checkboxes with
          // no explanation. But don't wipe a roster that already loaded — a blip on a later poll
          // shouldn't drop personalization mid-draft.
          if (!cancelled)
            setRoster((cur) => cur?.loaded ? cur : {
              loaded: false, tag: rosterTag, name: "", owned: [],
              error: String((e as Error)?.message || e),
            });
        });
    };
    pull();
    const id = setInterval(pull, ROSTER_POLL_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible" && Date.now() - last > 60_000) pull();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => { cancelled = true; clearInterval(id); document.removeEventListener("visibilitychange", onVisible); };
  }, [rosterTag]);

  const byId = useMemo(() => {
    const m = new Map<number, Brawler>();
    ref?.brawlers.forEach((b) => m.set(b.id, b));
    return m;
  }, [ref]);
  const map = useMemo(() => ref?.maps.find((m) => m.id === mapId) || null, [ref, mapId]);
  const mode = map?.mode || "";
  const blindPick = !!(rankInfo?.found && rankInfo.bracket && BLIND_PICK_BRACKETS.has(rankInfo.bracket));
  const used = useMemo(
    () => new Set([...bans, ...our, ...(blindPick ? [] : their)].filter((x): x is number => x != null)),
    [bans, our, their, blindPick]
  );
  const ownedSet = useMemo(() => new Set((roster?.owned || []).map((o) => o.id)), [roster]);
  const ownedByBrawler = useMemo(() => {
    const m = new Map<number, OwnedBrawler>();
    (roster?.owned || []).forEach((o) => m.set(o.id, o));
    return m;
  }, [roster]);
  // Opponents of a drafted brawler, for comp-aware loadout advice: their team for our picks
  // (zeroed under blind pick, mirroring the recommend body), our team for theirs. Sorted so the
  // cache key is order-independent.
  const enemiesFor = (bid: number): number[] => {
    const list = our.some((x) => x === bid) ? (blindPick ? [] : their) : our;
    return list.filter((x): x is number => x != null).sort((a, b) => a - b);
  };
  const loadoutKey = (bid: number) => `${bid}:${mode}:${enemiesFor(bid).join(",")}`;
  // Lazily fetch a drafted brawler's loadout advice; cache per (brawler, mode, enemy comp) so
  // re-hovering is instant, switching maps within a mode reuses it, and a landed enemy pick changes
  // the key so the next hover refetches comp-aware advice. A failed fetch is evicted so the next
  // hover retries.
  const ensureLoadout = (bid: number) => {
    if (!mode) return;
    const key = loadoutKey(bid);
    if (loadouts[key]) return;
    setLoadouts((c) => ({ ...c, [key]: "loading" }));
    getLoadout(bid, mode, mapId, enemiesFor(bid))
      .then((r) => setLoadouts((c) => ({ ...c, [key]: r })))
      .catch(() => setLoadouts((c) => { const n = { ...c }; delete n[key]; return n; }));
  };
  // A pick landing mid-hover (keyboard placement) changes the hovered slot's comp key, and
  // onMouseEnter won't re-fire — the browser keeps hover on the re-rendered button — which would
  // strand the open popover on "Loading…". Re-ensure the hovered slot whenever its key inputs move.
  useEffect(() => {
    if (!hoverSlot || hoverSlot.zone === "ban") return;
    const bid = (hoverSlot.zone === "our" ? our : their)[hoverSlot.index];
    if (bid != null) ensureLoadout(bid);
    // ensureLoadout/enemiesFor are render-scoped; deps below are exactly the cache-key inputs.
  }, [hoverSlot, our, their, mode, blindPick]);  // eslint-disable-line react-hooks/exhaustive-deps
  const boostedSet = useMemo(() => new Set(ref?.boosted || []), [ref]);
  const personalizeReady = !!roster?.loaded;
  const bracket = rankInfo?.found ? rankInfo.bracket : null;
  const personalTag = rankInfo?.found ? rankInfo.tag : null;
  // Owned brawlers you can actually field in this bracket: at/above its power floor (Power 11 from
  // Mythic up, else 9). A power of 0 means the roster didn't report it — leave it in rather than
  // hide a brawler on missing data. Below Mythic the floor is 9, so nothing you own is realistically
  // excluded; the gate bites in Mythic+, where an un-maxed owned brawler is simply unselectable.
  const powerFloor = minPowerForBracket(bracket);
  const fieldableOwned = useMemo(
    () => (roster?.owned || []).filter((o) => (o.power ?? 0) === 0 || (o.power ?? 0) >= powerFloor),
    [roster, powerFloor]
  );
  const fieldableSet = useMemo(() => new Set(fieldableOwned.map((o) => o.id)), [fieldableOwned]);

  const pickSeq = useMemo(
    () => blindPick
      ? Array.from({ length: TEAM_N }, (_, index) => ({ side: "us" as const, zone: "our" as const, index }))
      : pickSlotSequence(wePickFirst),
    [blindPick, wePickFirst]
  );
  const order = useMemo<Slot[]>(() => [
    ...Array.from({ length: BAN_N }, (_, i): Slot => ({ zone: "ban", index: i })),
    ...pickSeq.map((s): Slot => ({ zone: s.zone, index: s.index })),
  ], [pickSeq]);
  const active = useMemo<Slot | null>(() => {
    const empty = (s: Slot) => (s.zone === "ban" ? bans : s.zone === "our" ? our : their)[s.index] == null;
    if (activeOverride && empty(activeOverride)) return activeOverride;
    // Ranked bans are 3–6, not a fixed 6. Once you've moved on — any later slot is already filled —
    // an empty ban slot counts as skipped, so the cursor flows forward through the queue instead of
    // snapping back to it after every pick. To fill a skipped ban later, click that slot directly.
    const filledAfter: boolean[] = order.map(() => false);
    let seen = false;
    for (let i = order.length - 1; i >= 0; i--) {
      filledAfter[i] = seen;
      if (!empty(order[i])) seen = true;
    }
    return order.find((s, i) => empty(s) && !(s.zone === "ban" && filledAfter[i])) ?? null;
  }, [activeOverride, order, bans, our, their]);
  const step: Step = useMemo(() => {
    if (!active) return { kind: "done" };
    if (active.zone === "ban") return { kind: "ban", slot: active, index: active.index + 1 };
    return { kind: "pick", side: active.zone === "our" ? "us" : "them", slot: active, n: 0 };
  }, [active]);
  const phase: "ban" | "pick" = step.kind === "ban" ? "ban" : "pick";
  // overall step position → pick number within the 6-pick snake
  const orderPos = useMemo(
    () => (active ? order.findIndex((s) => s.zone === active.zone && s.index === active.index) : -1),
    [active, order]
  );
  const pickNo = orderPos >= 0 ? orderPos - BAN_N + 1 : 0;

  // Personalize only the user's OWN pick: when the active slot is the seat they marked as
  // themselves. Teammate/enemy/ban slots draw from the full pool (their rosters aren't ours).
  // Seat semantics only exist in the snake draft (Mythic+): under blind pick everyone on the team
  // picks at once, so there is no "your seat's turn" — personalization there is the dual-column
  // rail instead, and the whole grid stays unfiltered (you log teammates' picks too).
  const myTurn = personalizeReady && !blindPick && mySeat != null && active != null
    && active.zone === "our" && active.index === mySeat;
  const chooseSeat = (i: number) => setMySeat((cur) => {
    const next = cur === i ? null : i;   // clicking the current seat clears it (personalization off)
    if (next == null) localStorage.removeItem("bsdraft.myseat");
    else localStorage.setItem("bsdraft.myseat", String(next));
    return next;
  });

  useEffect(() => {
    if (blindPick && their.some((x) => x != null)) {
      setTheir(Array(TEAM_N).fill(null));
      setActiveOverride(null);
    }
  }, [blindPick, their]);

  // Tab jumps the active slot to the FIRST pick, skipping any remaining ban slots — handy when only
  // 3–5 bans are used and you want to move straight into picking. Tab is the browser's focus key, so
  // it's intercepted narrowly to stay accessible: plain Tab only (Shift/Ctrl/Alt/Meta chords pass
  // through, so Shift+Tab still steps focus back and Alt/⌘-Tab reach the OS), and of the editable
  // fields only the brawler search box is hijacked. That box is where focus lives for the whole draft
  // (the board focuses it on load and after every placement), so exempting it cost a wasted press: the
  // first Tab merely walked focus into the brawler grid and only the second one jumped. Every other
  // field — the tag box, any <select>, contenteditable — keeps native Tab traversal.
  // Focus is still never trapped: the jump is one-shot, because the next Tab finds the cursor already
  // on the first pick and falls through to the browser. It fires only when there's an empty first-pick
  // slot to jump into and you're not already sitting on it. "First pick" = pickSeq[0] (the snake's
  // first pick, or seat 0 under blind pick).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Tab" || e.shiftKey || e.ctrlKey || e.altKey || e.metaKey) return;
      const el = document.activeElement as HTMLElement | null;
      const tag = el?.tagName;
      const isField = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || !!el?.isContentEditable;
      if (isField && el !== searchRef.current) return; // other form fields (tag box, selects) → native Tab
      const first = pickSeq[0];
      if (!first) return;
      if (active && active.zone === first.zone && active.index === first.index) return; // already there — don't trap focus
      const firstEmpty = (first.zone === "our" ? our : their)[first.index] == null;
      if (!firstEmpty) return; // nothing to jump into
      e.preventDefault();
      setActiveOverride({ zone: first.zone, index: first.index });
      focusSearch();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pickSeq, active, our, their]);

  useEffect(() => {
    if (!mapId || !mode) return;
    // Roster + mastery + your win-rates apply to YOUR pick only (the seat you marked); a teammate
    // or enemy pick is scored on the pure meta, so its recommendations aren't filtered to what you own.
    const body = {
      map_id: mapId, mode,
      our_team: our.filter((x): x is number => x != null),
      their_team: blindPick ? [] : their.filter((x): x is number => x != null),
      bans: bans.filter((x): x is number => x != null),
      we_pick_first: wePickFirst, solo_queue: solo, phase,
      personalize: myTurn,
      personal_tag: myTurn ? personalTag : null,
      // Only brawlers you can actually field this bracket (owned + at/above the power floor); a
      // below-floor brawler is unselectable in-game, so it must not be recommended. Boosted/free
      // brawlers are added server-side (they arrive at Power 11), so they don't belong here.
      roster: myTurn ? fieldableOwned : null,
      rank_bracket: bracket, top: 12,
    };
    setLoading(true);
    const t = setTimeout(() => {
      recommend(body)
        .then((r) => { setRecs(r); setWarnings(r.warnings || []); })
        .catch((e) => setErr(String(e)))
        .finally(() => setLoading(false));
    }, 120);
    return () => clearTimeout(t);
  }, [mapId, mode, our, their, bans, wePickFirst, solo, phase, myTurn, fieldableOwned, bracket, personalTag, blindPick]);

  // Second fetch for the blind-pick dual columns: the personalized list (owned + fieldable +
  // boosted, mastery and your own win-rates folded in). Runs alongside the general fetch above —
  // under blind pick there's no seat/turn to gate on, the whole pick phase is "your pick".
  useEffect(() => {
    if (!blindPick || !personalizeReady) { setPersonalRecs(null); setPersonalErr(null); return; }
    if (!mapId || !mode || phase !== "pick") return;
    const body = {
      map_id: mapId, mode,
      our_team: our.filter((x): x is number => x != null),
      their_team: [],                     // blind pick: the enemy is never visible
      bans: bans.filter((x): x is number => x != null),
      we_pick_first: true, solo_queue: solo, phase: "pick" as const,
      personalize: true, personal_tag: personalTag,
      roster: fieldableOwned, rank_bracket: bracket, top: 12,
    };
    setPersonalLoading(true);
    const t = setTimeout(() => {
      recommend(body)
        .then((r) => { setPersonalRecs(r); setPersonalErr(null); })
        .catch(() => setPersonalErr("couldn't score your picks — retrying on the next board change"))
        .finally(() => setPersonalLoading(false));
    }, 120);
    return () => clearTimeout(t);
  }, [blindPick, personalizeReady, mapId, mode, our, bans, solo, phase, fieldableOwned, bracket, personalTag]);

  useEffect(() => {
    if (!mapId || !mode) return;
    const body = {
      map_id: mapId, mode,
      our_team: our.filter((x): x is number => x != null),
      their_team: blindPick ? [] : their.filter((x): x is number => x != null),
      bans: bans.filter((x): x is number => x != null),
      rank_bracket: bracket, top: 10,
    };
    let cancelled = false;
    const t = setTimeout(() => {
      getTopPicks(body)
        .then((r) => { if (!cancelled) { setTopPicks(r.picks); setRailOk(true); } })
        .catch(() => { if (!cancelled) setRailOk(false); });
    }, 120);
    return () => { cancelled = true; clearTimeout(t); };
  }, [mapId, mode, our, their, bans, bracket, blindPick]);

  const setZone = (zone: Zone, idx: number, val: number | null) => {
    const apply = (arr: (number | null)[]) => arr.map((x, i) => (i === idx ? val : x));
    if (zone === "ban") setBans(apply);
    else if (zone === "our") setOur(apply);
    else setTheir(apply);
  };

  const place = (bid: number) => {
    if (!active || used.has(bid)) return;
    setZone(active.zone, active.index, bid);
    setActiveOverride(null);
    setQuery("");
    focusSearch();
  };

  const reset = () => {
    setBans(Array(BAN_N).fill(null));
    setOur(Array(TEAM_N).fill(null));
    setTheir(Array(TEAM_N).fill(null));
    setActiveOverride(null);
    setLoadouts({});   // comp-keyed entries are stale across drafts; refetch is cheap
    focusSearch();
  };

  const checkRank = async () => {
    const t = tag.trim();
    if (!t) return;
    setRankLoading(true);
    try {
      const info = await getRank(t);
      setRankInfo(info);
      if (info.found) {
        setTag(info.tag);
        localStorage.setItem("bsdraft.tag", info.tag);
      }
    } catch {
      setRankInfo({ found: false, tag: t, tier: null, tier_label: null, bracket: null, source: null, error: "lookup failed" });
    } finally {
      setRankLoading(false);
    }
  };

  const clearTag = () => {
    setTag("");
    setRankInfo(null);
    localStorage.removeItem("bsdraft.tag");
  };

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (ref?.brawlers || [])
      .filter((b) => {
        if (!q) return true;
        const name = b.name.toLowerCase();
        return name.startsWith(q) || name.split(/[\s&.-]+/).some((w) => w.startsWith(q));
      })
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [ref, query]);

  // On your own pick, restrict to brawlers you can actually field — owned and at/above the bracket's
  // power floor, or this season's free "boosted" brawlers; otherwise anything unused is placeable.
  const placeable = (id: number) => !used.has(id) && (!myTurn || fieldableSet.has(id) || boostedSet.has(id));
  // the brawler Enter will place: first valid match for the current query
  const topMatch = useMemo(
    () => (query.trim() && step.kind !== "done" ? filtered.find((b) => placeable(b.id)) : undefined),
    [query, filtered, used, myTurn, fieldableSet, boostedSet, step.kind]
  );

  // ---- brawler-grid keyboard navigation (roving tabindex) ----
  // The grid keeps a live cursor — the roving tab stop — that outlives leaving the grid. Arrow keys
  // move it; Enter/Space on the focused tile places it (native <button> behavior — no extra handler).
  // An arrow pressed *outside* the grid moves that same cursor by the same step, so with it resting on
  // the first tile ArrowRight lands on the second brawler rather than onto the first: entering the
  // grid costs no extra press. ArrowUp off the top row returns to the search box. A disabled tile
  // (used / can't-field / draft done) isn't focusable, so nav skips over it and always lands on a
  // placeable brawler.
  const tileDisabled = (b: Brawler) =>
    used.has(b.id) || (myTurn && !fieldableSet.has(b.id) && !boostedSet.has(b.id)) || step.kind === "done";
  // Columns come from the live layout — the track list is auto-fill, so the count follows the panel's
  // width — which keeps ArrowUp/Down moving a true row at every width.
  const gridCols = () => {
    const el = gridRef.current;
    if (!el) return 6;
    return getComputedStyle(el).gridTemplateColumns.split(" ").filter(Boolean).length || 6;
  };
  // nearest placeable tile at/after `from`, scanning in `dir` (+1 / −1); −1 if none remain that way
  const seekTile = (from: number, dir: 1 | -1) => {
    for (let i = from; i >= 0 && i < filtered.length; i += dir)
      if (!tileDisabled(filtered[i])) return i;
    return -1;
  };
  // one arrow-key step of the cursor from tile `from`; −1 when nothing placeable lies that way
  const stepTile = (from: number, key: string) => {
    const c = gridCols();
    if (key === "ArrowRight") return seekTile(from + 1, 1);
    if (key === "ArrowLeft") return seekTile(from - 1, -1);
    if (key === "ArrowDown") return seekTile(from + c, 1);
    if (key === "ArrowUp") return seekTile(from - c, -1);
    return -1;
  };
  const focusTile = (i: number) => {
    (gridRef.current?.querySelector(`[data-tile="${i}"]`) as HTMLElement | null)?.focus();
    setGridFocus(i);
  };
  // A new query is a new list, so the cursor returns to its origin — the first tile. Otherwise a
  // cursor left deep in the full list would clamp to the tail of a short result set, and arrow entry
  // would start from the last match instead of the first.
  useEffect(() => { setGridFocus(0); }, [query]);

  // Enter the grid from outside it by *moving* the cursor, exactly as the same key would from inside:
  // one press gets you where a press from the cursor's tile would, with no wasted press landing on it.
  // The fallback covers the steps that have nowhere to go — the edge of the grid, or a filtered list
  // too short for a full row — by entering on the cursor's own tile, so an arrow always gets you in.
  // False means neither worked (draft done, every tile used/unfieldable): leave the key to the browser.
  const enterGrid = (key: string) => {
    const from = Math.max(0, Math.min(gridFocus, filtered.length - 1));
    let t = stepTile(from, key);
    if (t < 0) t = seekTile(from, 1);
    if (t < 0) t = seekTile(from, -1);
    if (t < 0) return false;
    focusTile(t);
    return true;
  };

  // The grid is the page's keyboard surface, so a bare arrow key anywhere outside it jumps straight in
  // — no Tab hop through the board first. Skipped wherever arrows already mean something: inside the
  // grid (its own handler navigates), in a <select>, <textarea>, or contenteditable, or in an <input>
  // — the tag box keeps its caret, and the search box has its own rule (it enters the grid too, but
  // only when there's no text to move through). Modifier chords pass through to the browser/OS.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!e.key.startsWith("Arrow") || e.shiftKey || e.ctrlKey || e.altKey || e.metaKey) return;
      const el = document.activeElement as HTMLElement | null;
      if (!el || gridRef.current?.contains(el)) return;
      const tag = el.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable) return;
      if (enterGrid(e.key)) e.preventDefault(); // only once it's actually moving focus — else arrows still scroll
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [gridFocus, filtered, used, myTurn, fieldableSet, boostedSet, step.kind]);

  const rotation = useMemo(
    () => (ref?.maps || []).filter((m) => m.games > 0).sort((a, b) =>
      a.mode === b.mode ? b.games - a.games : a.mode.localeCompare(b.mode)),
    [ref]
  );

  const isCurrent = (zone: Zone, index: number) =>
    step.kind !== "done" && step.slot.zone === zone && step.slot.index === index;

  const SlotBox = ({ zone, index, accent, size = 56 }: { zone: Zone; index: number; accent: string; size?: number }) => {
    const arr = zone === "ban" ? bans : zone === "our" ? our : their;
    const bid = arr[index];
    const b = bid != null ? byId.get(bid) : undefined;
    const current = isCurrent(zone, index);
    // Loadout hint on drafted pick slots (not bans). On the seat you marked as yourself it filters to
    // what you own; teammate/enemy slots show the mode's general best-fit loadout. Under blind pick
    // there are no seats (a stale saved seat must not claim a teammate's slot), so it's always general.
    const canHint = zone !== "ban" && bid != null;
    const isMySeat = personalizeReady && !blindPick && zone === "our" && index === mySeat;
    const hovered = hoverSlot?.zone === zone && hoverSlot?.index === index;
    return (
      <button
        onClick={() => { if (bid != null) setZone(zone, index, null); setActiveOverride({ zone, index }); focusSearch(); }}
        onMouseEnter={(e) => {
          if (!canHint) return;
          const r = e.currentTarget.getBoundingClientRect();
          setHoverSlot({ zone, index, rect: { left: r.left, top: r.top, bottom: r.bottom, width: r.width } });
          ensureLoadout(bid!);
        }}
        onMouseLeave={() => setHoverSlot((h) => (h && h.zone === zone && h.index === index ? null : h))}
        className="relative shrink-0 group"
        title={b ? `${b.name} · hover for loadout · click to clear` : "click to make this the active slot"}>
        <span key={bid ?? "empty"} className="block anim-snap">
          {b
            ? <Avatar b={b} size={size} dim={zone === "ban"} ring={current ? accent : undefined} active={current} />
            : <span className={`slot grid place-items-center ${current ? "slot-active" : ""}`}
                style={cssVars({ width: `${size}px`, height: `${size}px`, "--ring": accent, borderColor: current ? accent : "var(--line)" })}>
                <span className="text-base" style={{ color: current ? accent : "var(--dim)" }}>+</span>
              </span>}
        </span>
        {canHint && b && <span aria-hidden className="mono absolute -bottom-1 -right-1 text-[8px] px-0.5 leading-none border border-[var(--line)] bg-[var(--panel2)] text-[var(--dim)] opacity-0 group-hover:opacity-100 transition-opacity">⚙</span>}
        {canHint && hovered && hoverSlot && (
          <LoadoutPopover data={loadouts[loadoutKey(bid!)]} isMySeat={isMySeat}
            owned={ownedByBrawler.get(bid!)} accent={accent} rect={hoverSlot.rect} />
        )}
      </button>
    );
  };

  if (!ref)
    return (
      <BootScreen show={slowBoot} error={err}
        onRetry={() => { setErr(null); setSlowBoot(false); setBootNonce((n) => n + 1); }} />
    );

  // ---- status readout config ----
  const statusCfg =
    step.kind === "ban"
      ? { tag: `BAN ${String(step.index).padStart(2, "0")}/0${BAN_N}`, title: "BAN PHASE", sub: "Ban a threat · type to place · TAB skips to first pick", accent: "var(--red)", window: "BAN WINDOW" }
      : step.kind === "done"
      ? { tag: "DRAFT SET", title: "DRAFT COMPLETE", sub: "Game plan is locked in below", accent: "var(--green)", window: "" }
      : step.side === "us"
      ? { tag: `PICK ${pickNo}/0${PICK_N}`, title: "YOUR PICK", sub: "Lock a recommendation, or type any brawler + ⏎", accent: "var(--blue)", window: "PICK WINDOW" }
      : { tag: `PICK ${pickNo}/0${PICK_N}`, title: "ENEMY PICK", sub: "Log the enemy's choice · type + ⏎ is fastest", accent: "var(--red)", window: "PER PICK" };

  const recTitle = step.kind === "ban" ? "BAN TARGETS" : step.kind === "done" ? "COMPLETE" : step.side === "us" ? "PICK ORDERS" : "STRONG HERE";
  const callAccent = step.kind === "ban" ? "var(--red)" : step.kind === "pick" && step.side === "them" ? "var(--red)" : "var(--blue)";

  const pickList = recs?.picks || [];
  const banList = recs?.bans || [];
  // Diamond and below with a tag loaded: no seat to mark (everyone picks at once), so the rail
  // splits into META + personal pick columns instead of one seat-personalized list.
  const dualCols = blindPick && !!rosterTag && phase === "pick" && step.kind !== "done";
  const personalPicks = dualCols ? (personalRecs?.picks || []) : [];
  // Under the dual columns, Enter locks YOUR best pick — the meta column advises teammates.
  // Everywhere else (and while your list is still loading) it stays the general #1.
  const topPick = phase === "pick" ? (personalPicks[0] || pickList[0]) : undefined;
  const topBan = phase === "ban" ? banList[0] : undefined;

  return (
    <div className="p-3 md:p-5 max-w-[1240px] mx-auto">
      {/* ===== COMMAND BAR ===== */}
      <header className="panel flex flex-wrap items-center gap-2 px-3 py-2.5 mb-3">
        <div className="flex items-center gap-2 mr-1">
          <Logo size={24} />
          <span className="brand-gradient text-[15px]">BRAWL DRAFT</span>
          <span className="label hidden md:inline">// CONSOLE</span>
        </div>
        <a href="/purchases" title="What to upgrade next — personalized purchase advisor"
          className="mono text-[11px] uppercase tracking-[0.06em] px-2 py-1.5 border border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)] hover:border-[var(--line-strong)] ctl hidden sm:inline">
          Upgrades ↗
        </a>
        <select value={mapId ?? ""} onChange={(e) => setMapId(Number(e.target.value))}
          className="mono ctl bg-[var(--panel2)] border border-[var(--line)] pl-2.5 pr-7 py-1.5 text-[12px] max-w-[240px]">
          {rotation.map((m) => (
            <option key={m.id} value={m.id}>{m.mode.toUpperCase()} · {m.name}</option>
          ))}
        </select>
        <button onClick={() => setSolo((v) => !v)} data-on="true" className="seg px-2.5 py-1.5"
          style={cssVars({ "--seg-c": "var(--line-strong)" })}>
          {solo ? "SOLO Q" : "PREMADE"}
        </button>
        <div className="w-full sm:w-auto sm:ml-auto flex items-center gap-3">
          {health != null && (
            <span className="mono text-[11px] flex items-center gap-1.5 tabular-nums"
              title="Ranked matches in the live dataset · auto-refreshes every few minutes">
              <span className="live-dot w-1.5 h-1.5 inline-block" style={{ background: "var(--green)" }} />
              <span className="text-[var(--dim)] hidden sm:inline">ONLINE ·</span>
              <CountUp value={health.matches} className="text-[var(--green)] font-semibold" />
              <span className="text-[var(--dim)]">MATCHES</span>
            </span>
          )}
          <button onClick={reset} className="seg px-2.5 py-1.5">RESET</button>
        </div>
      </header>

      <RankWidget tag={tag} setTag={setTag} rankInfo={rankInfo} loading={rankLoading} onCheck={checkRank} onClear={clearTag} />

      {err && <div className="panel px-3 py-2 mb-3 mono text-[12px]" style={{ borderColor: "var(--red)", color: "var(--red)" }}>◇ {err}</div>}
      {meta?.shifted && <MetaBanner meta={meta} />}

      <div className="mb-3">
        <StatusReadout step={step} {...statusCfg} />
      </div>

      {/* ===== MAIN GRID: [board + placer] | [the call + meta] ===== */}
      <div className="grid gap-3 lg:grid-cols-[1fr_400px] items-start">
        {/* LEFT */}
        <div className="flex flex-col gap-3 min-w-0 order-2 lg:order-1">
          {/* BOARD */}
          <div className="panel p-3 order-2">
            <div className="flex items-center gap-3 mb-4 pb-3 border-b border-[var(--line)]">
              {map && (
                <>
                  <img src={map.image_url} alt={map.name} className="w-14 h-14 object-cover border border-[var(--line)]" />
                  <div className="min-w-0">
                    <div className="font-semibold text-[15px] truncate">{map.name}</div>
                    <div className="mono text-[10px] text-[var(--muted)] tracking-wide">
                      {map.mode.toUpperCase()} · {map.games.toLocaleString()} GAMES
                    </div>
                  </div>
                </>
              )}
              {blindPick ? (
                <span className="seg ml-auto shrink-0 inline-flex items-center gap-1.5 px-2.5 py-2" data-on="true"
                  style={cssVars({ "--seg-c": "var(--violet)" })}
                  title="Diamond and below draft blind: a ban phase, then you pick without seeing the enemy. No snake, no counter-picking.">
                  🙈 BLIND PICK
                </span>
              ) : (
                <FirstPickToggle wePickFirst={wePickFirst} onToggle={() => setWePickFirst((v) => !v)} />
              )}
            </div>

            <div className="label mb-2" style={{ color: "var(--red)" }}>⊘ Bans</div>
            <div className="flex flex-wrap gap-2 mb-6">
              {bans.map((_, i) => <SlotBox key={i} zone="ban" index={i} accent="var(--red)" size={48} />)}
            </div>

            <div className={`grid grid-cols-1 gap-x-6 gap-y-6 ${blindPick ? "" : "sm:grid-cols-2"}`}>
              <div>
                <div className="label mb-2" style={{ color: "var(--blue)" }}>◤ Your team</div>
                <div className="flex gap-2">
                  {our.map((_, i) => (
                    <div key={i} className="flex flex-col items-center gap-1.5">
                      <SlotBox zone="our" index={i} accent="var(--blue)" />
                      {!blindPick && (
                        <SeatCheck checked={mySeat === i} disabled={!personalizeReady}
                          onToggle={() => personalizeReady && chooseSeat(i)} seat={i + 1} />
                      )}
                    </div>
                  ))}
                </div>
                {!blindPick && (
                  <SeatHint ready={personalizeReady} chosen={mySeat != null} name={roster?.name} active={myTurn}
                    hasTag={!!rosterTag} error={roster?.error} />
                )}
                {blindPick && (
                  // No seat checkboxes here: everyone on the team picks at the same time, so there's
                  // no "your seat's turn" — the personalized column in the rail covers you instead.
                  <div className="mt-3 space-y-1.5">
                    <div className="mono text-[10px] text-[var(--muted)]">🙈 ENEMY HIDDEN AT DIAMOND · PICKS OPTIMIZE YOUR OWN COMP.</div>
                    {roster?.error && (
                      <div className="mono text-[9px]" style={{ color: "var(--red)" }} title={roster.error}>
                        ⚠ {rosterFailReason(roster.error)}
                      </div>
                    )}
                  </div>
                )}
              </div>
              {!blindPick && (
                <div>
                  <div className="label mb-2" style={{ color: "var(--red)" }}>◢ Enemy team</div>
                  <div className="flex gap-2">{their.map((_, i) => <SlotBox key={i} zone="their" index={i} accent="var(--red)" />)}</div>
                </div>
              )}
            </div>

            {recs?.composition && Object.keys(recs.composition).length > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-6 pt-3 border-t border-[var(--line)] items-center">
                <span className="label mr-1">Comp</span>
                {Object.entries(recs.composition).map(([cls, n]) => (
                  <span key={cls} className="mono text-[10px] px-2 py-0.5 border anim-snap"
                    style={{ borderColor: (CLASS_COLOR[cls] || "#333") + "66", color: CLASS_COLOR[cls] || "#aaa" }}>
                    {(CLASS_SHORT[cls] || cls).toUpperCase()}{n > 1 ? ` ×${n}` : ""}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* COMMAND INPUT — keyboard-first placer, pinned to the top of the column so your
              fastest action (type + ⏎ to log a ban/pick) never needs a scroll */}
          <div className="panel px-3 py-2 order-1">
            <div className="flex items-center gap-2">
              <span className="label shrink-0" style={{ color: "var(--accent)" }}>▸ INPUT</span>
              <input ref={searchRef} value={query} onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  // Enter places into the active slot: the typed match when you're typing, otherwise
                  // the board's #1 suggestion (top pick/ban for the current phase) — the keyboard-fast
                  // "lock the call" without reaching for the mouse.
                  if (e.key === "Enter") {
                    e.preventDefault();
                    if (query.trim()) { if (topMatch) place(topMatch.id); }
                    else { const top = topPick || topBan; if (top) place(top.brawler_id); }
                  } else if (
                    e.key === "ArrowDown" ||
                    // Left/Right drop into the grid only when the caret has nothing left to move
                    // through in that direction — Right at the end of the text, Left at the start —
                    // so mid-text they stay caret keys for editing, but "type a name then arrow to
                    // the match you want" works without first emptying the box. An empty box has the
                    // caret at both edges, so it still enters on either. ArrowUp is left alone: the
                    // grid sits below this box and its top row already sends ArrowUp back here, so
                    // entering on Up would just ping-pong.
                    (e.key === "ArrowRight" && e.currentTarget.selectionStart === query.length && e.currentTarget.selectionEnd === query.length) ||
                    (e.key === "ArrowLeft" && e.currentTarget.selectionStart === 0 && e.currentTarget.selectionEnd === 0)
                  ) {
                    if (enterGrid(e.key)) e.preventDefault();
                  } else if (e.key === "Escape") setQuery("");
                }}
                placeholder="TYPE TO PLACE  ·  e.g.  bro ↵"
                aria-label="Type a brawler name and press Enter to place it in the active slot"
                className="mono flex-1 min-w-0 bg-transparent border-none px-1 py-2 text-[13px] tracking-wide outline-none placeholder:text-[var(--dim)]" />
              {topMatch ? (
                <span className="mono text-[11px] px-2 py-1 shrink-0 anim-fade flex items-center gap-1.5"
                  style={{ background: "var(--accent)", color: "#0a0a0c" }}>
                  ⏎ {topMatch.name.toUpperCase()}
                </span>
              ) : (
                <span className="kbd shrink-0">⏎ PLACE</span>
              )}
            </div>
          </div>

          {/* BRAWLER GRID — mouse / browse fallback, below the board */}
          <div className="panel p-3 order-3">
            <div className="flex items-center justify-between mb-2.5">
              <span className="label">{filtered.length} BRAWLERS · TAP, TYPE, OR ◂▸ ARROWS</span>
              {myTurn && <span className="mono text-[10px]" style={{ color: "var(--gold)" }}>◈ OWNED{powerFloor === 11 ? " · P11" : ""} + FREE</span>}
            </div>
            <div ref={gridRef}
              onKeyDown={(e) => {
                const idxAttr = (e.target as HTMLElement).dataset?.tile;
                if (idxAttr == null) return; // key came from something other than a tile
                const i = Number(idxAttr);
                let target = -1;
                if (e.key.startsWith("Arrow")) {
                  target = stepTile(i, e.key);
                  if (target < 0 && e.key === "ArrowUp") { e.preventDefault(); focusSearch(); return; } // off the top row → back to search
                } else if (e.key === "Home") target = seekTile(0, 1);
                else if (e.key === "End") target = seekTile(filtered.length - 1, -1);
                else if (e.key === "Escape") { e.preventDefault(); focusSearch(); return; } // back to search
                else return;
                e.preventDefault();
                if (target >= 0) focusTile(target);
              }}
              // Tracks are sized so a 375px phone still fits 6 across (a 48px floor drops it to 5,
              // costing a third of the brawlers visible without scrolling). p-1, not pr-1: scrolling
              // clips at the padding box, and the top-match / keyboard-focus rings sit up to 4px
              // outside a tile — with no padding they'd be shaved on the edge rows.
              className="grid grid-cols-[repeat(auto-fill,minmax(44px,1fr))] sm:grid-cols-[repeat(auto-fill,minmax(58px,1fr))] gap-1.5 max-h-[300px] overflow-y-auto p-1">
              {filtered.map((b, i) => {
                const isUsed = used.has(b.id);
                const isBoosted = boostedSet.has(b.id);
                // dimmed only on your own pick, when you can't field it — neither at/above the power
                // floor nor a free "boosted" brawler. Owned-but-under-floor reads differently from
                // not-owned: the game blocks it purely on power level, so say so.
                const restricted = myTurn && !fieldableSet.has(b.id) && !isBoosted;
                const underPower = restricted && ownedSet.has(b.id);
                const isTop = topMatch?.id === b.id;
                // roving tabindex: exactly one tile is a tab stop; arrow keys move focus among the rest
                const tabStop = i === Math.max(0, Math.min(gridFocus, filtered.length - 1));
                return (
                  <button key={b.id} data-tile={i} tabIndex={tabStop ? 0 : -1} onFocus={() => setGridFocus(i)}
                    onClick={() => place(b.id)} disabled={tileDisabled(b)}
                    className="pick-tile group relative disabled:cursor-not-allowed"
                    title={underPower ? `${b.name} · needs Power ${powerFloor} in ${bracket}` : restricted ? `${b.name} · not owned` : isBoosted ? `${b.name} (${b.cls}) · free this season` : `${b.name} (${b.cls})`}>
                    <span className="thumb block"
                      style={cssVars({ "--tc": RARITY_COLOR[b.rarity] || "#26303f", outline: isTop ? "2px solid var(--accent)" : undefined, outlineOffset: isTop ? "1px" : undefined })}>
                      <Avatar b={b} fluid dim={isUsed || restricted} />
                    </span>
                    {isTop && <span className="mono absolute top-0 right-0 text-[8px] px-1 leading-tight" style={{ background: "var(--accent)", color: "#0a0a0c" }}>⏎</span>}
                    {underPower && !isUsed && (
                      <span className="mono absolute top-0 right-0 text-[7px] px-0.5 leading-tight font-bold"
                        style={{ background: "var(--muted)", color: "#0a0a0c" }} title={`not Power ${powerFloor} — unselectable in ${bracket}`}>P{powerFloor}</span>
                    )}
                    {myTurn && isBoosted && !isTop && !isUsed && (
                      <span className="mono absolute top-0 right-0 text-[7px] px-0.5 leading-tight font-bold"
                        style={{ background: "var(--gold)", color: "#0a0a0c" }} title="free maxed brawler this Ranked season">FREE</span>
                    )}
                    <span className="mono block text-[8px] truncate w-full text-center mt-0.5 tracking-tight"
                      style={{ color: restricted ? "var(--dim)" : "var(--muted)" }}>{b.name}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* RIGHT — THE CALL + ranked + top meta (sticky; first on mobile) */}
        <div className="space-y-3 lg:sticky lg:top-3 h-fit order-1 lg:order-2">
          <div className="panel">
            <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--line)]">
              <span className="label" style={{ color: callAccent }}>◆ {recTitle}</span>
              {(loading || personalLoading) && <span className="mono text-[10px] text-[var(--muted)] caret">ANALYZING</span>}
            </div>

            {warnings.length > 0 && (
              <div className="px-3 py-2 border-b border-[var(--line)] space-y-1 anim-fade">
                {warnings.map((w, i) => (
                  <div key={i} className="flex items-start gap-2 mono text-[10.5px]">
                    <span style={{ color: SEV_COLOR[w.severity] || "#888" }}>▸</span>
                    <span className="text-[var(--muted)]">{w.text}</span>
                  </div>
                ))}
              </div>
            )}

            {step.kind === "done" && (
              <div className="p-4 mono text-[12px] text-[var(--muted)]">✓ DRAFT SET · SEE YOUR GAME PLAN BELOW.</div>
            )}

            {/* BLIND-PICK DUAL COLUMNS — general meta vs. your own roster, side by side */}
            {dualCols && (
              <PickColumns general={pickList} generalReady={!!recs}
                personal={personalRecs ? (personalRecs.picks || []) : null}
                personalError={roster?.error ? rosterFailReason(roster.error) : personalErr}
                name={roster?.name} byId={byId} onPlace={place} />
            )}

            {/* THE CALL — the dominant #1 read (single-list mode) */}
            {!dualCols && step.kind !== "done" && phase === "pick" && topPick && (
              <TheCall kind="pick" r={topPick} b={byId.get(topPick.brawler_id)} accent={callAccent} onPlace={() => place(topPick.brawler_id)} />
            )}
            {step.kind !== "done" && phase === "ban" && topBan && (
              <TheCall kind="ban" r={topBan} b={byId.get(topBan.brawler_id)} accent={callAccent} onPlace={() => place(topBan.brawler_id)} />
            )}
            {!recs && step.kind !== "done" && !dualCols && <CallSkeleton />}

            {/* ranked runners-up */}
            <div>
              {!dualCols && step.kind !== "done" && phase === "pick" &&
                pickList.slice(1).map((r, i) => <RankedPick key={r.brawler_id} r={r} i={i + 2} b={byId.get(r.brawler_id)} onClick={() => place(r.brawler_id)} />)}
              {step.kind !== "done" && phase === "ban" &&
                banList.slice(1).map((r, i) => <RankedBan key={r.brawler_id} r={r} i={i + 2} b={byId.get(r.brawler_id)} onClick={() => place(r.brawler_id)} />)}
              {!recs && step.kind !== "done" && !dualCols && [0, 1, 2].map((i) => <RowSkeleton key={i} />)}
            </div>
          </div>

          {railOk && <TopMetaStrip picks={topPicks} byId={byId} used={used} onPick={place} disabled={step.kind === "done"} />}
        </div>
      </div>

      {recs?.game_plan && our.some((x) => x != null) && <GamePlanPanel gp={recs.game_plan} blind={blindPick} />}
      <AdSlot name="footer" />
    </div>
  );
}

// ---- THE CALL: the large, unmistakable #1 recommendation ----
function TheCall({ kind, r, b, accent, onPlace }: {
  kind: "pick" | "ban"; r: PickRec | BanRec; b?: Brawler; accent: string; onPlace: () => void;
}) {
  const isBan = kind === "ban";
  const pr = r as PickRec, br = r as BanRec;
  // Shown instantly (no count-up): under a 20s clock the headline number must read true on the
  // first glance — an animated ramp from 0 briefly shows a misleadingly low value.
  const score = isBan ? br.threat : pr.score;
  // A ban leads with its projected swing when the backend could compute one; threat is the
  // fallback, and stays visible below either way as the raw read on the brawler.
  const swing = isBan && br.ban_value != null ? br.ban_value : null;
  const headline = swing != null ? swingLabel(swing) : pct(score);
  const scoreLabel = swing != null ? "WIN SWING" : isBan ? "THREAT" : "SCORE";
  const col = isBan ? "var(--red)" : scoreColor(score);
  const reason = isBan ? banReason(br) : pickReason(pr);
  const cls = b?.cls || pr.cls;
  return (
    <button onClick={onPlace} className="card-rec block w-full text-left anim-snap" style={cssVars({ "--glow": accent })}>
      <div className="flex items-center justify-between px-3 pt-2.5 pb-2">
        <span className="mono text-[10px] tracking-[0.15em]" style={{ color: accent }}>▸ THE CALL</span>
        <span className="mono text-[10px] font-bold px-2 py-1" style={{ background: accent, color: "#0a0a0c" }}>{isBan ? "BAN" : "PICK"} ⏎</span>
      </div>
      <div className="px-3 pb-2 flex gap-3">
        <div className="relative shrink-0">
          <Avatar b={b} size={72} />
          <span className="mono absolute -top-1.5 -left-1.5 text-[9px] font-bold px-1 leading-tight" style={{ background: accent, color: "#0a0a0c" }}>01</span>
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="display text-[19px] truncate">{r.name}</div>
              <div className="mono text-[10px] tracking-[0.1em]" style={{ color: CLASS_COLOR[cls] || "#aaa" }}>{(CLASS_SHORT[cls] || cls).toUpperCase()}</div>
            </div>
            <div className="text-right shrink-0" title={swing != null ? SWING_HINT : undefined}>
              <div className="mono font-bold text-[30px] leading-none tabular-nums" style={{ color: col }}>{headline}</div>
              <div className="mono text-[9px] tracking-[0.12em] text-[var(--dim)] mt-1">{scoreLabel}</div>
            </div>
          </div>
          <div className="mono text-[11px] text-[var(--muted)] mt-2 leading-snug">▸ {reason}</div>
        </div>
      </div>
      <div className="px-3 pb-3 space-y-1.5">
        {isBan ? (
          <>
            <Meter k="WINRATE" v={br.map_winrate} />
            <div className="flex items-center gap-2">
              <span className="mono w-9 shrink-0 text-[9px] tracking-[0.12em] text-[var(--dim)]">USE</span>
              <div className="meter flex-1"><i style={{ width: pct(Math.min(1, br.use_rate * 2)), background: "var(--gold)" }} /></div>
              <span className="mono w-6 text-right text-[10px] tabular-nums text-[var(--muted)]">{two(br.use_rate)}</span>
            </div>
          </>
        ) : (
          pickSignals(pr).slice(0, 5).map((s) => <Meter key={s.k} k={s.k} v={s.v} />)
        )}
        <div className="flex items-center justify-between pt-1.5 border-t border-[var(--line)]">
          <span className="label">CONFIDENCE {pct((r as PickRec).confidence ?? (r as BanRec).confidence)}</span>
          {!isBan && pr.personal_games != null && (
            <span className="mono text-[10px] px-1.5 py-0.5" style={{ background: "#3b82f622", color: "#7fb4ff" }}
              title="your recent ranked games with this brawler">YOU · {Math.round(pr.personal_games)}G</span>
          )}
        </div>
        {!isBan && pr.gaps && pr.gaps.length > 0 && (
          <div className="flex flex-wrap gap-1 pt-1.5">
            {pr.gaps.map((g) => (
              <span key={g} className="mono text-[9px] px-1.5 py-0.5 border uppercase tracking-[0.06em]"
                style={{ borderColor: "#e8843a66", color: "#e8a24a" }} title="missing from your loadout">{g}</span>
            ))}
          </div>
        )}
      </div>
    </button>
  );
}

function RankedPick({ r, i, b, onClick }: { r: PickRec; i: number; b?: Brawler; onClick: () => void }) {
  const score = r.score;
  const sig = pickSignals(r).sort((a, b2) => Math.abs(b2.v - 0.5) - Math.abs(a.v - 0.5)).slice(0, 3);
  return (
    <button onClick={onClick} className="card-rec flex items-center gap-2.5 w-full text-left px-3 py-2 border-t border-[var(--line)]"
      style={cssVars({ "--glow": "var(--blue)" })}>
      <span className="mono text-[11px] text-[var(--dim)] w-4 text-right tabular-nums">{i}</span>
      <Avatar b={b} size={34} />
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2">
          <span className="font-semibold text-[13px] truncate">{r.name}</span>
          <span className="mono text-[9px] tracking-[0.08em] shrink-0" style={{ color: CLASS_COLOR[r.cls] || "#aaa" }}>{CLASS_SHORT[r.cls] || r.cls}</span>
        </div>
        <SigLine sig={sig} />
      </div>
      <span className="mono font-bold text-[16px] tabular-nums shrink-0" style={{ color: scoreColor(score) }}>{pct(score)}</span>
    </button>
  );
}

function RankedBan({ r, i, b, onClick }: { r: BanRec; i: number; b?: Brawler; onClick: () => void }) {
  return (
    <button onClick={onClick} className="card-rec flex items-center gap-2.5 w-full text-left px-3 py-2 border-t border-[var(--line)]"
      style={cssVars({ "--glow": "var(--red)" })}>
      <span className="mono text-[11px] text-[var(--dim)] w-4 text-right tabular-nums">{i}</span>
      <Avatar b={b} size={34} />
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2">
          <span className="font-semibold text-[13px] truncate">{r.name}</span>
          <span className="mono text-[9px] tracking-[0.08em] shrink-0" style={{ color: CLASS_COLOR[r.cls] || "#aaa" }}>{CLASS_SHORT[r.cls] || r.cls}</span>
        </div>
        <span className="mono text-[10px] tabular-nums text-[var(--dim)]">
          MAP <span style={{ color: r.map_winrate >= 0.5 ? "var(--green)" : "var(--muted)" }}>{two(r.map_winrate)}</span>
          <span className="text-[var(--line-strong)]"> · </span>
          USE <span style={{ color: "var(--gold)" }}>{two(r.use_rate)}</span>
          {r.self_deny && <>
            <span className="text-[var(--line-strong)]"> · </span>
            <span style={{ color: "var(--blue)" }} title="the draft projects this onto your side — pick it rather than ban it">YOURS</span>
          </>}
        </span>
      </div>
      <span className="mono font-bold text-[16px] tabular-nums shrink-0"
        style={{ color: r.self_deny ? "var(--dim)" : "var(--red)" }}
        title={r.ban_value != null ? SWING_HINT : undefined}>
        {r.ban_value != null ? swingLabel(r.ban_value) : pct(r.threat)}
      </span>
    </button>
  );
}

// ---- Blind-pick dual columns: META (anyone's best pick — advise teammates from it) beside the
// personalized list (owned + fieldable + boosted, your mastery and win-rates). Replaces the single
// TheCall+runners list at Diamond and below once a tag is loaded: with the whole team picking at
// once there's no seat to personalize "in turn", so both reads stay visible the entire phase.
const COL_PICKS = 8; // rows per column — keeps the rail scannable under a 20s pick window

function PickColumns({ general, generalReady, personal, personalError, name, byId, onPlace }: {
  general: PickRec[]; generalReady: boolean;
  personal: PickRec[] | null;          // null → roster or personal scoring still loading
  personalError: string | null; name?: string;
  byId: Map<number, Brawler>; onPlace: (id: number) => void;
}) {
  // The ⏎ badge marks what Enter places: your #1 once it's live, else the meta #1 (matches topPick).
  const personalLive = !!personal && personal.length > 0;
  return (
    <div className="grid grid-cols-2 anim-fade">
      <div className="min-w-0 border-r border-[var(--line)]">
        <div className="px-2.5 pt-2 label cursor-help"
          title="the strongest picks for anyone, scored from ranked data that is ~97% Power 11 with a full loadout — advise your teammates from this side">◆ META</div>
        <div className="px-2.5 pb-1 mono text-[8px] tracking-[0.08em] text-[var(--dim)]">ANYONE · P11 BASELINE</div>
        {generalReady
          ? general.slice(0, COL_PICKS).map((r, i) => (
              <MiniPick key={r.brawler_id} r={r} b={byId.get(r.brawler_id)} top={i === 0}
                accent="var(--blue)" enterHint={i === 0 && !personalLive} onClick={() => onPlace(r.brawler_id)} />
            ))
          : [0, 1, 2, 3].map((i) => <MiniRowSkeleton key={i} />)}
      </div>
      <div className="min-w-0">
        <div className="px-2.5 pt-2 label truncate cursor-help" style={{ color: "var(--gold)" }}
          title="only brawlers you own that clear this bracket's power floor — a different list from META, not a re-ranking of it. The % assumes a Power 11 brawler on a full loadout, so it does not yet dock for power level or a missing hypercharge.">
          ◈ {(name || "YOU").toUpperCase()}
        </div>
        <div className="px-2.5 pb-1 mono text-[8px] tracking-[0.08em] text-[var(--dim)]">YOUR ROSTER · P11 BASELINE</div>
        {personalError ? (
          <div className="mono text-[10px] px-2.5 py-3 leading-snug" style={{ color: "var(--red)" }}>⚠ {personalError}</div>
        ) : personal == null ? (
          [0, 1, 2, 3].map((i) => <MiniRowSkeleton key={i} />)
        ) : personal.length === 0 ? (
          <div className="mono text-[10px] text-[var(--dim)] px-2.5 py-3 leading-snug">◦ nothing you own scores here — lean on the meta column</div>
        ) : (
          personal.slice(0, COL_PICKS).map((r, i) => (
            <MiniPick key={r.brawler_id} r={r} b={byId.get(r.brawler_id)} top={i === 0}
              accent="var(--gold)" enterHint={i === 0} onClick={() => onPlace(r.brawler_id)} />
          ))
        )}
      </div>
    </div>
  );
}

// Compact half-width row: no rank number (order implies it), the single strongest signal, and the
// score. The full signal line + rationale live in the tooltip — a ~190px column can't fit them.
function MiniPick({ r, b, top, accent, enterHint, onClick }: {
  r: PickRec; b?: Brawler; top: boolean; accent: string; enterHint: boolean; onClick: () => void;
}) {
  const sig = pickSignals(r).sort((a, b2) => Math.abs(b2.v - 0.5) - Math.abs(a.v - 0.5))[0];
  const detail = pickSignals(r).map((s) => `${s.k} ${two(s.v)}`).join(" · ");
  const gaps = gapTags(r.gaps);
  // The score now prices power/loadout deficits (readiness), so the % already reflects these gaps
  // (hypercharge excepted — no estimator exists). This compact row doesn't render the reason chips,
  // so the note still names what the player's copy is missing.
  const gapNote = gaps.length > 0 ? `\n⊘ ${r.gaps.join(" · ")} — missing from your copy` : "";
  return (
    <button onClick={onClick}
      className="card-rec flex items-center gap-2 w-full text-left px-2 py-1.5 border-t border-[var(--line)]"
      style={cssVars({ "--glow": accent })}
      title={`${r.name} · ${pct(r.score)} — ${pickReason(r)}\n${detail}${gapNote}`}>
      <Avatar b={b} size={top ? 40 : 30} />
      <div className="flex-1 min-w-0">
        <div className={`font-semibold truncate ${top ? "text-[13px]" : "text-[12px]"}`}>{r.name}</div>
        <div className="flex items-baseline gap-1.5 min-w-0">
          {sig && (
            <span className="mono text-[9px] tabular-nums text-[var(--dim)] shrink-0">
              {sig.k} <span style={{ color: sig.v >= 0.5 ? "var(--green)" : "var(--muted)" }}>{two(sig.v)}</span>
            </span>
          )}
          {gaps.length > 0 && (
            <span className="mono text-[9px] tracking-[0.06em] truncate" style={{ color: "#e8a24a" }}>
              ⊘ {gaps.join(" ")}
            </span>
          )}
        </div>
      </div>
      <div className="text-right shrink-0">
        <div className={`mono font-bold tabular-nums ${top ? "text-[15px]" : "text-[13px]"}`}
          style={{ color: scoreColor(r.score) }}>{pct(r.score)}</div>
        {enterHint && <div className="mono text-[8px] px-1 leading-tight inline-block" style={{ background: accent, color: "#0a0a0c" }}>⏎</div>}
      </div>
    </button>
  );
}

function MiniRowSkeleton() {
  return (
    <div className="flex items-center gap-2 px-2 py-1.5 border-t border-[var(--line)]">
      <div className="w-[30px] h-[30px] skeleton" />
      <div className="flex-1 space-y-1"><div className="h-2.5 w-16 skeleton" /><div className="h-2 w-10 skeleton" /></div>
    </div>
  );
}

function CallSkeleton() {
  return (
    <div className="p-3">
      <div className="flex gap-3">
        <div className="w-[72px] h-[72px] skeleton" />
        <div className="flex-1 space-y-2 pt-1">
          <div className="h-4 w-28 skeleton" />
          <div className="h-2.5 w-16 skeleton" />
          <div className="h-2 w-40 skeleton mt-3" />
        </div>
      </div>
      <div className="mt-3 space-y-1.5">
        {[0, 1, 2, 3].map((i) => <div key={i} className="h-1.5 w-full skeleton" />)}
      </div>
    </div>
  );
}
function RowSkeleton() {
  return (
    <div className="flex items-center gap-2.5 px-3 py-2 border-t border-[var(--line)]">
      <div className="w-[34px] h-[34px] skeleton" />
      <div className="flex-1 space-y-1.5"><div className="h-2.5 w-20 skeleton" /><div className="h-2 w-28 skeleton" /></div>
      <div className="h-4 w-9 skeleton" />
    </div>
  );
}

// Skinny horizontal strip of the map's strongest brawlers at a full loadout — the pure meta,
// stable across the draft. Icons only; a constant "who's generally strong here" reference.
function TopMetaStrip({ picks, byId, used, onPick, disabled }: {
  picks: TopPick[]; byId: Map<number, Brawler>; used: Set<number>;
  onPick: (id: number) => void; disabled: boolean;
}) {
  const blurb =
    "The strongest picks right now if you owned every brawler at a full loadout (all gadgets, " +
    "gears & star powers). Updates as the draft fills in, but ignores your roster, the pure meta.";
  return (
    <div className="panel">
      <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--line)] cursor-help" title={blurb}>
        <span className="label" style={{ color: "var(--gold)" }}>◆ TOP META</span>
        <span className="label">FULL LOADOUT</span>
      </div>
      <div className="p-2 flex flex-wrap gap-1.5">
        {picks.length === 0
          ? [0, 1, 2, 3, 4, 5, 6, 7].map((i) => <div key={i} className="w-11 h-11 skeleton" />)
          : picks.map((p, i) => {
              const b = byId.get(p.brawler_id);
              const isUsed = used.has(p.brawler_id);
              return (
                <button key={p.brawler_id} onClick={() => onPick(p.brawler_id)} disabled={isUsed || disabled}
                  className="group relative disabled:cursor-not-allowed anim-snap"
                  title={`#${i + 1}  ${p.name}\n${pct(p.score)} pick score · ${pct(p.map_winrate)} map win rate\nassumes a full loadout`}>
                  <span className="thumb block" style={cssVars({ "--tc": (b && RARITY_COLOR[b.rarity]) || "#26303f" })}>
                    <Avatar b={b} size={44} dim={isUsed} />
                  </span>
                  <span className="mono absolute -top-1 -left-1 text-[8px] px-0.5 leading-tight"
                    style={{ background: i === 0 ? "var(--gold)" : "var(--panel3)", color: i === 0 ? "#0a0a0c" : "var(--muted)", border: "1px solid var(--line)" }}>{i + 1}</span>
                </button>
              );
            })}
      </div>
    </div>
  );
}

// Verdict colors for a head-to-head / pair rate. `favored` and `unfavored` are the same hues as
// `strong`/`losing` at lower emphasis — the bands are wide on purpose (see `_edge` in
// engine/gameplan.py), so the palette shouldn't imply more precision than the samples support.
const EDGE_COLOR: Record<string, string> = {
  strong: "var(--green)", favored: "var(--green)", even: "var(--muted)",
  unfavored: "var(--gold)", losing: "var(--red)", unknown: "var(--dim)",
};
const EDGE_DIM: Record<string, number> = { strong: 1, favored: 0.72, even: 1, unfavored: 0.72, losing: 1 };

// Win rates live in a narrow band around 50%, so a 0-100% bar is all fill and no signal. This
// plots the *deviation* from even on a centered axis instead, saturating at ±`span`.
function DeltaBar({ v, span = 0.1 }: { v: number; span?: number }) {
  const d = Math.max(-1, Math.min(1, (v - 0.5) / span));
  const up = d >= 0;
  return (
    <div className="relative h-[5px] w-full" style={{ background: "var(--panel3)" }}>
      <div className="absolute inset-y-0" style={{ left: "50%", width: 1, background: "var(--line-strong)" }} />
      <div className="absolute inset-y-0" style={{
        left: up ? "50%" : `${50 + d * 50}%`, width: `${Math.abs(d) * 50}%`,
        background: up ? "var(--green)" : "var(--red)",
      }} />
    </div>
  );
}

function GamePlanPanel({ gp, blind }: { gp: GamePlan; blind?: boolean }) {
  const Section = ({ label, color, mark, items }: { label: string; color: string; mark: string; items: string[] }) =>
    items.length === 0 ? null : (
      <div>
        <div className="label mb-2" style={{ color }}>{label}</div>
        <ul className="space-y-1.5">
          {items.map((t, i) => <li key={i} className="mono text-[11px] text-[var(--muted)] leading-snug flex gap-1.5"><span style={{ color }}>{mark}</span>{t}</li>)}
        </ul>
      </div>
    );
  // `?? []` throughout: an older API (see GamePlan in lib/api.ts) sends none of these.
  const h2h = gp.head_to_head;
  const mapRead = gp.map_read ?? [];
  const pairs = gp.pairs ?? [];
  const ourNames = h2h?.grid[0]?.vs.map((c) => c.name) ?? [];
  const hasData = !!(h2h || mapRead.length > 0 || pairs.length > 0);
  // Every chip prints the sample behind it — the panel claims its numbers are measured, so a
  // headline without an n would be the one place that claim isn't kept.
  const Chip = ({ mark, label, body, n, color }: { mark: string; label: string; body: string; n?: number | null; color: string }) => (
    <div className="flex items-baseline gap-1.5 px-2 py-1" style={{ background: "var(--panel2)", borderLeft: `2px solid ${color}` }}>
      <span className="mono text-[9px] tracking-[0.1em]" style={{ color }}>{mark} {label}</span>
      <span className="mono text-[11px] text-[var(--muted)]">{body}</span>
      {n != null && <span className="mono text-[10px] text-[var(--dim)]">n≈{Math.round(n)}</span>}
    </div>
  );
  return (
    <div className="panel mt-3 p-4 anim-fade">
      <div className="flex items-center gap-2 mb-3 pb-3 border-b border-[var(--line)] flex-wrap">
        <span className="label" style={{ color: "var(--accent)" }}>▣ Game plan</span>
        <span className="mono text-[10px] px-2 py-0.5 border border-[var(--line)] text-[var(--muted)] uppercase tracking-wide">{gp.archetype}</span>
        {gp.enemy && (
          <>
            <span className="mono text-[10px] text-[var(--dim)]">vs</span>
            <span className="mono text-[10px] px-2 py-0.5 border uppercase tracking-wide"
              style={{ borderColor: "var(--red)66", color: "var(--red)" }}>{gp.enemy.archetype}</span>
          </>
        )}
      </div>

      <div className="grid md:grid-cols-2 gap-3 mb-4">
        <div className="p-3" style={{ background: "#3b82f610", border: "1px solid #3b82f640" }}>
          <div className="label mb-1" style={{ color: "var(--blue)" }}>◎ Win condition</div>
          <div className="text-[13px]">{gp.win_condition}</div>
          {gp.objective && <div className="mono text-[10px] text-[var(--muted)] mt-1.5 uppercase tracking-wide">OBJECTIVE · {gp.objective}</div>}
        </div>
        {gp.model_read && (
          <div className="p-3" style={{ background: "var(--panel2)", border: "1px solid var(--line)" }}>
            <div className="flex items-baseline justify-between mb-1">
              <span className="label">◆ Model read on the draft</span>
              <span className="mono text-[15px]" style={{ color: scoreColor(gp.model_read.win_prob) }}>{pct(gp.model_read.win_prob)}</span>
            </div>
            <DeltaBar v={gp.model_read.win_prob} />
            <div className="text-[12px] mt-1.5">{gp.model_read.verdict}</div>
            <div className="mono text-[10px] text-[var(--dim)] mt-1 leading-snug">{gp.model_read.note}</div>
          </div>
        )}
      </div>

      {gp.enemy?.clash && (
        <div className="mb-4 pl-2.5 py-1 text-[12px] text-[var(--muted)] leading-snug" style={{ borderLeft: "2px solid var(--line-strong)" }}>
          <span className="mono text-[10px] tracking-[0.1em] text-[var(--dim)]">STYLE CLASH · </span>{gp.enemy.clash}
        </div>
      )}

      <div className="grid md:grid-cols-2 gap-x-6 gap-y-4">
        <div>
          <div className="label mb-2">Your roles</div>
          <div className="space-y-1.5">
            {gp.roles.map((r) => (
              <div key={r.name} className="text-[12px]">
                <span className="font-semibold" style={{ color: CLASS_COLOR[r.cls] || "#aaa" }}>{r.name}</span>
                <span className="mono text-[11px] text-[var(--muted)]"> / {r.role}</span>
              </div>
            ))}
          </div>
          <div className="mono text-[11px] text-[var(--muted)] mt-2 italic">{gp.playstyle}</div>
        </div>
        <div>
          <div className="label mb-2" style={{ color: "var(--red)" }}>Vs their threats</div>
          <div className="space-y-1.5">
            {gp.threats.length === 0 && <div className="mono text-[11px] text-[var(--muted)]">{blind ? "ENEMY HIDDEN IN BLIND PICK — FOCUS YOUR OWN COMP." : "NO ENEMY PICKS ON THE BOARD YET."}</div>}
            {gp.threats.map((t) => (
              <div key={t.name} className="text-[12px]">
                <span className="font-semibold" style={{ color: CLASS_COLOR[t.cls] || "#aaa" }}>{t.name}</span>
                <span className="mono text-[11px] text-[var(--muted)]"> / {t.tip}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-[var(--line)]">
        <div className="label mb-1">◆ Standard mode &amp; role strategy</div>
        <div className="mono text-[10px] text-[var(--dim)] mb-3 leading-snug">
          RULE-BASED, NOT LEARNED — THE MATCH DATA IS DRAFT-TO-OUTCOME ONLY, WITH NO POSITIONS OR TIMINGS,
          SO NOTHING IN IT COULD TEACH A MODEL HOW TO PLAY THE MODE. THE ROLES AND THREAT TIPS ABOVE ARE THIS HALF TOO.
        </div>
        <div className="grid md:grid-cols-3 gap-x-6 gap-y-4">
          <Section label="Do" color="var(--green)" mark="✓" items={gp.tips} />
          <Section label="Avoid" color="var(--red)" mark="✕" items={gp.avoid} />
          <Section label="Compensate" color="var(--gold)" mark="⚠" items={gp.compensate} />
        </div>
      </div>
      {hasData && (
        <div className="mt-4 pt-3 border-t border-[var(--line)]">
          <div className="label mb-1" style={{ color: "var(--gold)" }}>◆ From the collected matches</div>
          <div className="mono text-[10px] text-[var(--dim)] mb-3 leading-snug">
            EVERYTHING BELOW IS MEASURED, NOT ADVICE — SAME MATCH DATA THAT RANKS THE PICK BOARD. N = EFFECTIVE
            SAMPLE AFTER RECENCY WEIGHTING; THIN CELLS ARE LEFT BLANK RATHER THAN SHOWN AT LOW CONFIDENCE.
          </div>

          {h2h && (
            <div className="mb-4">
              <div className="label mb-1.5">Head-to-head</div>
              <div className="overflow-x-auto">
                <table className="border-collapse min-w-full">
                  <thead>
                    <tr>
                      <th className="mono text-[9px] tracking-[0.1em] text-[var(--dim)] font-normal text-left pb-1.5 pr-3">THEIR PICK</th>
                      {ourNames.map((n) => (
                        <th key={n} className="mono text-[10px] font-normal text-right pb-1.5 px-2 whitespace-nowrap" style={{ color: "var(--blue)" }}>{n}</th>
                      ))}
                      <th className="mono text-[9px] tracking-[0.1em] text-[var(--dim)] font-normal text-right pb-1.5 pl-3 whitespace-nowrap">AVG</th>
                    </tr>
                  </thead>
                  <tbody>
                    {h2h.grid.map((row) => (
                      <tr key={row.enemy} className="border-t border-[var(--line)]">
                        <td className="py-1.5 pr-3 whitespace-nowrap">
                          <span className="text-[12px] font-semibold" style={{ color: CLASS_COLOR[row.enemy_cls] || "#aaa" }}>{row.enemy}</span>
                          <span className="mono text-[9px] tracking-[0.08em] ml-1.5" style={{ color: CLASS_COLOR[row.enemy_cls] || "#aaa", opacity: 0.7 }}>{CLASS_SHORT[row.enemy_cls] || row.enemy_cls}</span>
                        </td>
                        {row.vs.map((c, i) => (
                          <td key={i} className="py-1.5 px-2 text-right mono text-[12px] tabular-nums"
                            style={{ color: EDGE_COLOR[c.edge] || "var(--muted)", opacity: EDGE_DIM[c.edge] ?? 1 }}
                            title={c.winrate == null ? "too thin a sample with both on the board"
                              : `n≈${Math.round(c.games)} effective (recency-weighted) sample`}>
                            {c.winrate == null ? "·" : pct(c.winrate)}
                          </td>
                        ))}
                        <td className="py-1.5 pl-3 text-right mono text-[12px] tabular-nums text-[var(--muted)] whitespace-nowrap"
                          title={row.mean == null ? "no cell in this row cleared the sample floor"
                            : `mean of ${row.mean_cells ?? 0} of ${row.vs.length} cells · n≈${Math.round(row.mean_games ?? 0)}`}>
                          {row.mean == null ? "·" : pct(row.mean)}
                          {row.mean != null && (
                            <span className="text-[var(--dim)] text-[10px]"> /{row.mean_cells ?? 0}</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mono text-[10px] text-[var(--dim)] mt-1.5 leading-snug">
                YOUR SIDE&apos;S WIN RATE IN MATCHES WITH BOTH BRAWLERS ON THE BOARD — A TEAM RESULT, NOT A 1V1 DUEL.
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                {h2h.best?.winrate != null && (
                  <Chip mark="▲" label="LEAN ON" color="var(--green)" n={h2h.best.games}
                    body={`${h2h.best.ours} into ${h2h.best.theirs} · ${pct(h2h.best.winrate)}`} />
                )}
                {h2h.danger?.winrate != null && (
                  <Chip mark="▼" label="RISK" color="var(--red)" n={h2h.danger.games}
                    body={`${h2h.danger.ours} into ${h2h.danger.theirs} · ${pct(h2h.danger.winrate)}`} />
                )}
                {h2h.focus?.winrate != null && (
                  <Chip mark="◎" label="FOCUS" color="var(--gold)" n={h2h.focus.games}
                    body={`${h2h.focus.enemy} — weakest overall, across ${h2h.focus.cells ?? 0} matchups · ${pct(h2h.focus.winrate)}`} />
                )}
              </div>
            </div>
          )}

          <div className="grid md:grid-cols-2 gap-x-6 gap-y-4">
            {mapRead.length > 0 && (
              <div>
                <div className="label mb-2">Form on this map</div>
                <div className="space-y-2">
                  {mapRead.map((m) => (
                    <div key={m.name}>
                      <div className="flex items-baseline justify-between gap-2">
                        <div className="min-w-0">
                          <span className="text-[12px] font-semibold" style={{ color: CLASS_COLOR[m.cls] || "#aaa" }}>{m.name}</span>
                          {m.tag !== "solid" && (
                            <span className="mono text-[9px] tracking-[0.1em] ml-1.5"
                              style={{ color: m.tag === "anchor" ? "var(--green)" : "var(--gold)" }}>
                              {m.tag === "anchor" ? "PLAY THROUGH" : "WEAK LINK"}
                            </span>
                          )}
                        </div>
                        <span className="mono text-[11px] tabular-nums shrink-0" style={{ color: scoreColor(m.winrate) }}>
                          {pct(m.winrate)} <span className="text-[var(--dim)]">n≈{Math.round(m.games)}</span>
                        </span>
                      </div>
                      <div className="mt-1"><DeltaBar v={m.winrate} /></div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {pairs.length > 0 && (
              <div>
                <div className="label mb-2">Your pairs</div>
                <div className="space-y-2">
                  {pairs.map((p) => (
                    <div key={`${p.a}-${p.b}`}>
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="text-[12px] truncate">{p.a} <span className="text-[var(--dim)]">+</span> {p.b}</span>
                        <span className="mono text-[11px] tabular-nums shrink-0" style={{ color: scoreColor(p.winrate) }}>
                          {pct(p.winrate)} <span className="text-[var(--dim)]">n≈{Math.round(p.games)}</span>
                        </span>
                      </div>
                      <div className="mt-1"><DeltaBar v={p.winrate} /></div>
                    </div>
                  ))}
                </div>
                <div className="mono text-[10px] text-[var(--dim)] mt-2 leading-snug">
                  WIN RATE WHEN THESE TWO WERE DRAFTED TOGETHER — KEEP THE STRONGEST PAIR PLAYING OFF EACH OTHER.
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
