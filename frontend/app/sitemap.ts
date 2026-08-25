import type { MetadataRoute } from "next";
import { rankedModes } from "@/lib/reference";

// Statically generated at build time (next.config output: "export") -> out/sitemap.xml.
// Enumerates the real routes under app/; keep in sync when a page is added or removed.
// The /guides/<mode> routes are derived from the same loader that generates them.
export const dynamic = "force-static";

const BASE = "https://brawldraft.com";

export default function sitemap(): MetadataRoute.Sitemap {
  const pages = [
    "", "/how-it-works", "/faq", "/guide", "/guides",
    ...rankedModes().map((m) => `/guides/${m.slug}`),
    "/purchases", "/model", "/privacy",
  ];
  return pages.map((p) => ({
    url: `${BASE}${p}`,
    // The homepage tracks a dataset that re-syncs continuously; the written pages are stable.
    changeFrequency: p === "" ? "daily" : "monthly",
    priority: p === "" ? 1 : 0.7,
  }));
}
