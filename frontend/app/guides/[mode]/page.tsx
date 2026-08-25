import type { Metadata } from "next";
import { notFound } from "next/navigation";
import DocNav from "@/components/DocNav";
import { DocFooter } from "@/components/ContentPage";
import { rankedModes, modeBySlug } from "@/lib/reference";
import { MODE_GUIDES } from "@/lib/guide-content";

// Statically generate exactly the six ranked-mode guides at build time (output: "export").
export const dynamicParams = false;

export function generateStaticParams() {
  return rankedModes().map((m) => ({ mode: m.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ mode: string }> }): Promise<Metadata> {
  const { mode } = await params;
  const ref = modeBySlug(mode);
  if (!ref) return {};
  return {
    title: `Drafting ${ref.name} in Brawl Stars Ranked — Brawl Draft`,
    description: `How to draft ${ref.name} in Brawl Stars Ranked: what wins the mode, how to structure picks and bans, and the ${ref.maps.length} competitive ${ref.name} maps Ranked draws its rotation from.`,
    alternates: { canonical: `/guides/${ref.slug}` },
  };
}

function SectionHead({ n, title }: { n: string; title: string }) {
  return (
    <div className="flex items-baseline gap-2.5 mb-3">
      <span className="mono text-[11px] tabular-nums text-[var(--dim)] shrink-0 pt-0.5">{n}</span>
      <h2 className="text-lg font-bold tracking-tight text-[var(--text)]">{title}</h2>
    </div>
  );
}

export default async function ModeGuidePage({ params }: { params: Promise<{ mode: string }> }) {
  const { mode } = await params;
  const ref = modeBySlug(mode);
  if (!ref) notFound();
  const guide = MODE_GUIDES[ref.name];
  const others = rankedModes().filter((m) => m.slug !== ref.slug);
  const indent = "pl-[calc(11px+0.625rem)]";

  return (
    <div className="min-h-screen p-3 md:p-5 max-w-3xl mx-auto">
      <DocNav current="/guides" />

      <header className="mb-9">
        <nav aria-label="Breadcrumb" className="label mb-3">
          <a href="/guides" className="hover:text-[var(--text)]">Mode guides</a>
          {" / "}
          <span style={{ color: ref.color }}>{ref.name}</span>
        </nav>
        <div className="flex items-center gap-3 mb-4">
          {ref.imageUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={ref.imageUrl} alt="" width={40} height={40} className="shrink-0" />
          )}
          <h1 className="display text-[clamp(1.9rem,5vw,3rem)]">Drafting {ref.name}</h1>
        </div>
        <div className="h-px w-full bg-[var(--line)] mb-5" />
        <p className="text-[15px] leading-relaxed font-medium" style={{ color: ref.color }}>
          {guide.tagline}
        </p>
      </header>

      <div className="space-y-9">
        <section>
          <SectionHead n="01" title="The objective" />
          <div className={`space-y-3 text-[15px] leading-relaxed text-[var(--muted)] ${indent}`}>
            {guide.objective.map((p, i) => (
              <p key={i}>{p}</p>
            ))}
          </div>
        </section>

        <section>
          <SectionHead n="02" title="How to draft it" />
          <div className={`space-y-2 ${indent}`}>
            {guide.draft.map((point, i) => (
              <div key={point.title} className="panel p-4">
                <h3 className="font-bold tracking-tight mb-1.5 text-[var(--text)]">
                  <span className="mono text-[11px] tabular-nums mr-2" style={{ color: ref.color }}>
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {point.title}
                </h3>
                <p className="text-sm leading-relaxed text-[var(--muted)]">{point.body}</p>
              </div>
            ))}
          </div>
        </section>

        <section>
          <SectionHead n="03" title="Who to ban" />
          <div className={`text-[15px] leading-relaxed text-[var(--muted)] ${indent}`}>
            <p>{guide.bans}</p>
            <p className="mt-3">
              The ban phase runs from Diamond upward — below that, drafts go straight to picks. Bans
              are also the most patch-sensitive call in the draft: for today&rsquo;s data-backed ban
              suggestions on a specific map, use the{" "}
              <a href="/" className="text-[var(--blue)] underline underline-offset-2 decoration-[var(--line-strong)] hover:decoration-[var(--blue)] hover:text-[var(--text)]">draft board</a>.
            </p>
          </div>
        </section>

        <section>
          <SectionHead n="04" title={`The ${ref.name} maps`} />
          <div className={`text-[15px] leading-relaxed text-[var(--muted)] ${indent}`}>
            <p className="mb-4">{guide.mapNotes}</p>
            <ul className="flex flex-wrap gap-1.5">
              {ref.maps.map((m) => (
                <li
                  key={m.name}
                  className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] text-[var(--muted)]"
                >
                  {m.name}
                </li>
              ))}
            </ul>
            <p className="text-xs mt-3 text-[var(--dim)]">
              {`All ${ref.maps.length} competitive ${ref.name} maps — Ranked rotates a handful of these each season, and the draft board’s map picker tracks the live rotation, with per-map win rates, synergies, and counters for each.`}
            </p>
          </div>
        </section>

        <section>
          <SectionHead n="05" title="Other modes" />
          <div className={`flex flex-wrap gap-1.5 ${indent}`}>
            {others.map((m) => (
              <a
                key={m.slug}
                href={`/guides/${m.slug}`}
                className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] text-[var(--muted)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl"
              >
                {m.name}
              </a>
            ))}
          </div>
        </section>
      </div>

      <DocFooter current="/guides" />
    </div>
  );
}
