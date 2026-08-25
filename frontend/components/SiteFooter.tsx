// Server-rendered on purpose. DraftBoard is a client component that returns a BootScreen until
// the API answers, so anything inside it is absent from the exported HTML — which means crawlers
// (and the AdSense reviewer) would see the homepage as an empty shell with no links to the
// written pages. Keeping the footer out here guarantees the nav and the Supercell notice are in
// the static markup, and visible even if the backend is cold or down.
export default function SiteFooter({ blurb }: { blurb?: string }) {
  return (
    <footer className="max-w-[1240px] mx-auto px-3 md:px-5 pb-6 mt-6 pt-4 border-t border-[var(--line)] text-center text-xs text-[var(--muted)]">
      <nav className="flex flex-wrap justify-center gap-x-1.5 gap-y-1.5 mb-3">
        <a href="/purchases" className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl">Upgrades</a>
        <a href="/guide" className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl">Draft guide</a>
        <a href="/guides" className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl">Mode guides</a>
        <a href="/how-it-works" className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl">How it works</a>
        <a href="/model" className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl">The model</a>
        <a href="/faq" className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl">FAQ</a>
        <a href="/privacy" className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl">Privacy</a>
      </nav>
      <p className="mono text-[10px] leading-relaxed text-[var(--dim)] max-w-2xl mx-auto">
        {blurb && <>{blurb} · </>}
        This content is not affiliated with, endorsed, sponsored, or specifically approved by Supercell and Supercell is not
        responsible for it (
        <a href="https://supercell.com/en/fan-content-policy/" className="underline hover:text-[var(--text)]"
          target="_blank" rel="noopener noreferrer">Fan Content Policy</a>
        )
      </p>
    </footer>
  );
}
