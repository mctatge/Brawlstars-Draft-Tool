// Build-time access to the bundled Brawlify reference snapshots in ../data/reference.
// Server-only (uses fs): imported by the statically-generated guide pages and the sitemap
// during `next build`, so the exported HTML carries real reference data with no runtime
// fetch. Do not import from client components.
//
// What this yields per mode is the *competitive map catalog* — every not-retired map for the
// mode, ~13-23 each. Ranked only rotates a handful of those per mode per season, and the only
// rotation signal is collected match volume, which lives server-side (see the reference
// endpoint in backend/bsdraft/api/main.py). Static pages built from this file must therefore
// present the catalog as the set Ranked draws from, never as the live rotation.
import fs from "node:fs";
import path from "node:path";

// Mirrors RANKED_MODES in backend/bsdraft/constants.py — update both when the
// ranked rotation changes.
export const RANKED_MODES = [
  "Gem Grab",
  "Brawl Ball",
  "Knockout",
  "Hot Zone",
  "Heist",
  "Bounty",
] as const;

export type RankedModeName = (typeof RANKED_MODES)[number];

// Mirrors RANKED_MAP_ENABLE_OVERRIDES in backend/bsdraft/data/reference.py: maps live in
// ranked whose upstream `disabled` flag lags behind (the upstream feed tracks the casual
// rotation). Update both together.
const MAP_ENABLE_OVERRIDES = new Set([15000886]); // Safe(r) Zone (Heist)

export type GuideMap = { name: string; environment: string };

export type ModeRef = {
  name: RankedModeName;
  slug: string;
  color: string;
  imageUrl: string;
  maps: GuideMap[];
};

type RawGameMode = { name?: string; color?: string; imageUrl?: string };
type RawMap = {
  id?: number;
  name: string;
  disabled?: boolean;
  gameMode?: { name?: string };
  environment?: { name?: string };
};

// process.cwd() is frontend/ during `next build`; the reference JSONs live at the repo root.
const REFERENCE_DIR = path.join(process.cwd(), "..", "data", "reference");

function loadJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(path.join(REFERENCE_DIR, file), "utf8"));
}

export const modeSlug = (name: string) => name.toLowerCase().replace(/\s+/g, "-");

let cache: ModeRef[] | null = null;

/** The six ranked modes with their competitive map catalogs, in canonical order. */
export function rankedModes(): ModeRef[] {
  if (cache) return cache;
  const modes = loadJson<{ list: RawGameMode[] }>("gamemodes.json").list;
  const maps = loadJson<{ list: RawMap[] }>("maps.json").list;
  cache = RANKED_MODES.map((name) => {
    const meta = modes.find((m) => m.name === name);
    const pool = maps
      .filter(
        (x) =>
          (!x.disabled || (x.id !== undefined && MAP_ENABLE_OVERRIDES.has(x.id))) &&
          x.gameMode?.name === name,
      )
      .map((x) => ({ name: x.name, environment: x.environment?.name ?? "" }))
      .sort((a, b) => a.name.localeCompare(b.name));
    return {
      name,
      slug: modeSlug(name),
      color: meta?.color ?? "#3b82f6",
      imageUrl: meta?.imageUrl ?? "",
      maps: pool,
    };
  });
  return cache;
}

export function modeBySlug(slug: string): ModeRef | undefined {
  return rankedModes().find((m) => m.slug === slug);
}
