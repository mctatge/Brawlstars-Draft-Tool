import type { Metadata } from "next";
import DocNav from "@/components/DocNav";
import { DocFooter } from "@/components/ContentPage";
import { rankedModes } from "@/lib/reference";
import { MODE_GUIDES } from "@/lib/guide-content";

export const metadata: Metadata = {
  title: "Brawl Stars Ranked Mode Guides — Brawl Draft",
  description:
    "Drafting guides for all six Brawl Stars Ranked modes — Gem Grab, Brawl Ball, Knockout, Hot Zone, Heist, and Bounty — what wins each mode, how to draft it, who to ban, and how its map pool splits into archetypes.",
  alternates: { canonical: "/guides" },
};

export default function GuidesIndex() {
  const modes = rankedModes();

  return (
    <div className="min-h-screen p-3 md:p-5 max-w-3xl mx-auto">
      <DocNav current="/guides" />

      <header className="mb-9">
        <div className="label mb-3" style={{ color: "var(--accent)" }}>▸ FIELD MANUAL</div>
        <h1 className="display text-[clamp(1.9rem,5vw,3rem)] mb-4">Ranked mode guides</h1>
        <div className="h-px w-full bg-[var(--line)] mb-5" />
        <div className="text-[15px] leading-relaxed text-[var(--muted)]">
          <p className="mb-3">
            Brawl Stars Ranked rotates across six modes, and each one rewards a different draft.
            These guides cover the fundamentals that don&rsquo;t change with the patch: what actually
            wins each mode, how to structure a draft for it, and how its maps split into archetypes.
          </p>
          <p>
            For live, patch-aware numbers on any specific map — win rates, synergies, counters, and
            suggested bans — bring the map to the{" "}
            <a href="/" className="text-[var(--blue)] underline underline-offset-2 decoration-[var(--line-strong)] hover:decoration-[var(--blue)] hover:text-[var(--text)]">draft board</a>.
            For the draft format itself — bans, blind pick vs the snake, seat position — see the{" "}
            <a href="/guide" className="text-[var(--blue)] underline underline-offset-2 decoration-[var(--line-strong)] hover:decoration-[var(--blue)] hover:text-[var(--text)]">draft guide</a>.
          </p>
        </div>
      </header>

      <div className="grid gap-2 sm:grid-cols-2">
        {modes.map((m) => (
          <a
            key={m.slug}
            href={`/guides/${m.slug}`}
            className="panel card-rec block p-4"
            style={{ "--glow": m.color } as React.CSSProperties}
          >
            <div className="flex items-center gap-2.5 mb-2">
              {m.imageUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={m.imageUrl} alt="" width={26} height={26} className="shrink-0" />
              )}
              <h2 className="font-bold tracking-tight text-[var(--text)]">{m.name}</h2>
            </div>
            <p className="text-[var(--muted)] text-xs leading-relaxed mb-3">{MODE_GUIDES[m.name].tagline}</p>
            <p className="mono text-[10px] uppercase tracking-[0.08em]" style={{ color: m.color }}>
              {m.maps.length} competitive maps →
            </p>
          </a>
        ))}
      </div>

      <section className="mt-9">
        <h2 className="text-lg font-bold tracking-tight text-[var(--text)] mb-3">How these guides fit the tool</h2>
        <div className="text-[15px] leading-relaxed text-[var(--muted)]">
          <p>
            The guides are the stable layer: draft principles that hold across balance patches. The
            draft board is the live layer — a win-probability model and per-map statistics rebuilt
            continuously from over a million real ranked matches, which is where you should look for
            &ldquo;who do I actually pick on this map, today.&rdquo; The principles tell you why the
            numbers look the way they do; the numbers tell you which brawler executes the principle
            best this patch. See{" "}
            <a href="/how-it-works" className="text-[var(--blue)] underline underline-offset-2 decoration-[var(--line-strong)] hover:decoration-[var(--blue)] hover:text-[var(--text)]">how it works</a>{" "}
            for what&rsquo;s behind the numbers.
          </p>
        </div>
      </section>

      <DocFooter current="/guides" />
    </div>
  );
}
