import DocNav, { NAV } from "@/components/DocNav";

export type Section = { heading: string; body: string; bullets?: string[] };
export type Content = { title: string; intro: string; sections: Section[] };

// Minimal inline formatter: the drafted copy only ever uses **bold** and [links](/href), so a
// full markdown dependency would be dead weight in a static export. One alternation walks both.
const INLINE = /\*\*(.+?)\*\*|\[(.+?)\]\(([^)]+)\)/g;

function inline(text: string) {
  const out: React.ReactNode[] = [];
  let last = 0, key = 0, m: RegExpExecArray | null;
  INLINE.lastIndex = 0;
  while ((m = INLINE.exec(text)) !== null) {
    if (m.index > last) out.push(<span key={key++}>{text.slice(last, m.index)}</span>);
    if (m[1] !== undefined) {
      out.push(<strong key={key++} className="font-semibold text-[var(--text)]">{m[1]}</strong>);
    } else {
      out.push(
        <a key={key++} href={m[3]} className="text-[var(--blue)] underline underline-offset-2 decoration-[var(--line-strong)] hover:decoration-[var(--blue)] hover:text-[var(--text)]">{m[2]}</a>
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(<span key={key++}>{text.slice(last)}</span>);
  return out;
}

function Prose({ text }: { text: string }) {
  return (
    <>
      {text.split(/\n\n+/).map((para, i) => (
        <p key={i} className="mb-3 last:mb-0">{inline(para)}</p>
      ))}
    </>
  );
}

// Shared by ContentPage and the hand-laid-out docs pages (the /guides section), so the nav
// chips and the Supercell notice stay identical everywhere.
export function DocFooter({ current }: { current: string }) {
  return (
    <footer className="mt-12 pt-5 border-t border-[var(--line)] text-xs text-[var(--muted)]">
      <div className="flex flex-wrap gap-x-1 gap-y-1.5 mb-3">
        {NAV.filter((n) => n.href !== current).map((n) => (
          <a key={n.href} href={n.href} className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl">{n.label}</a>
        ))}
        <a href="/privacy" className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl">Privacy</a>
      </div>
      <p className="mono text-[10px] leading-relaxed text-[var(--dim)]">
        This content is not affiliated with, endorsed, sponsored, or specifically approved by Supercell and Supercell is not
        responsible for it (
        <a href="https://supercell.com/en/fan-content-policy/" className="underline hover:text-[var(--text)]"
          target="_blank" rel="noopener noreferrer">Fan Content Policy</a>
        ).
      </p>
    </footer>
  );
}

export default function ContentPage({ content, current }: { content: Content; current: string }) {
  return (
    <div className="min-h-screen p-3 md:p-5 max-w-3xl mx-auto">
      <DocNav current={current} />

      <header className="mb-9">
        <div className="label mb-3" style={{ color: "var(--accent)" }}>▸ FIELD MANUAL</div>
        <h1 className="display text-[clamp(1.9rem,5vw,3rem)] mb-4">{content.title}</h1>
        <div className="h-px w-full bg-[var(--line)] mb-5" />
        <div className="text-[15px] leading-relaxed text-[var(--muted)]">
          <Prose text={content.intro} />
        </div>
      </header>

      <div className="space-y-9">
        {content.sections.map((s, i) => (
          <section key={s.heading}>
            <div className="flex items-baseline gap-2.5 mb-3">
              <span className="mono text-[11px] tabular-nums text-[var(--dim)] shrink-0 pt-0.5">{String(i + 1).padStart(2, "0")}</span>
              <h2 className="text-lg font-bold tracking-tight text-[var(--text)]">{s.heading}</h2>
            </div>
            <div className="text-[15px] leading-relaxed text-[var(--muted)] pl-[calc(11px+0.625rem)]">
              <Prose text={s.body} />
              {s.bullets && s.bullets.length > 0 && (
                <ul className="mt-3 space-y-2">
                  {s.bullets.map((b, j) => (
                    <li key={j} className="pl-4 relative">
                      <span className="mono absolute left-0 top-0" style={{ color: "var(--accent)" }}>▸</span>
                      {inline(b)}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        ))}
      </div>

      <DocFooter current={current} />
    </div>
  );
}
