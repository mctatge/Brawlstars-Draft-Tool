import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy — Brawl Draft",
  description: "What Brawl Draft stores and why: a player tag in your browser, public match data, and (if ads are enabled) Google AdSense.",
};

// Required for AdSense approval, and honest on its own terms: the site has no accounts and only
// Cloudflare's cookieless Web Analytics, so there is genuinely little to disclose. Keep this page in sync with what the
// board actually does — an inaccurate privacy policy is worse than none.
export default function Privacy() {
  return (
    <div className="min-h-screen p-3 md:p-5 max-w-2xl mx-auto">
      <nav className="panel flex flex-wrap items-center gap-x-1 gap-y-1 px-3 py-2 mb-8">
        <span className="brand-gradient text-[14px] mr-2">BRAWL DRAFT</span>
        <div className="flex flex-wrap gap-x-1 ml-auto mono text-[11px] uppercase tracking-[0.06em]">
          <a href="/" className="px-2.5 py-1.5 border border-transparent text-[var(--muted)] hover:text-[var(--text)] ctl">Draft board</a>
          <a href="/purchases" className="px-2.5 py-1.5 border border-transparent text-[var(--muted)] hover:text-[var(--text)] ctl">Upgrades</a>
          <a href="/guide" className="px-2.5 py-1.5 border border-transparent text-[var(--muted)] hover:text-[var(--text)] ctl">Draft guide</a>
          <a href="/how-it-works" className="px-2.5 py-1.5 border border-transparent text-[var(--muted)] hover:text-[var(--text)] ctl">How it works</a>
          <a href="/faq" className="px-2.5 py-1.5 border border-transparent text-[var(--muted)] hover:text-[var(--text)] ctl">FAQ</a>
          <span className="px-2.5 py-1.5 border text-[var(--text)]" style={{ borderColor: "var(--accent)" }} aria-current="page">Privacy</span>
        </div>
      </nav>
      <header className="mb-7">
        <div className="label mb-3" style={{ color: "var(--accent)" }}>▸ PRIVACY</div>
        <h1 className="display text-[clamp(1.6rem,4vw,2.4rem)]">Privacy</h1>
        <p className="mono text-[10px] text-[var(--dim)] mt-2 uppercase tracking-[0.1em]">Effective September 17, 2026</p>
        <div className="h-px w-full bg-[var(--line)] mt-5" />
      </header>

      <div className="space-y-5 text-sm leading-relaxed text-[var(--text)]">
        <section>
          <h2 className="font-semibold mb-1.5">The short version</h2>
          <p className="text-[var(--muted)]">
            Brawl Draft has no accounts, no sign-in, and no tracking of who you are. The only thing you can
            give the site is a Brawl Stars player tag, and the only place it is remembered is your own browser.
          </p>
        </section>

        <section>
          <h2 className="font-semibold mb-1.5">Your player tag</h2>
          <p className="text-[var(--muted)]">
            Entering a tag is optional. When you do, it is sent to our server to look up your rank, owned
            brawlers, and personal win rates so recommendations can be personalized, and it is saved in your
            browser&rsquo;s local storage so the board remembers you next visit. The ✕ button next to the tag field
            deletes it from your browser. Tags are public identifiers in Brawl Stars; we do not link them to
            names, emails, or any other personal information.
          </p>
        </section>

        <section>
          <h2 className="font-semibold mb-1.5">Match data</h2>
          <p className="text-[var(--muted)]">
            Recommendations are built from public ranked-match data collected through the official Brawl Stars
            API. This is the same information visible in the in-game battle log.
          </p>
        </section>

        <section>
          <h2 className="font-semibold mb-1.5">Hosting</h2>
          <p className="text-[var(--muted)]">
            The site is served by Cloudflare Pages and the API runs on Render. Like nearly all web hosts, they
            keep standard server logs (IP address, request time) to operate the service. Cloudflare Web Analytics
            also counts page views and load times in aggregate; it does not use cookies or local storage and
            does not identify you.
          </p>
        </section>

        <section>
          <h2 className="font-semibold mb-1.5">Advertising</h2>
          <p className="text-[var(--muted)]">
            The site may show ads served by Google AdSense. Google and its partners may use cookies or device
            identifiers to serve ads based on your visits to this and other sites; where required (for example
            in the EEA and UK), you will be asked for consent first, and you can choose non-personalized ads.
            You can manage ad personalization at{" "}
            <a href="https://adssettings.google.com" className="underline hover:text-[var(--text)]" target="_blank" rel="noopener noreferrer">
              adssettings.google.com
            </a>{" "}
            and read Google&rsquo;s policy at{" "}
            <a href="https://policies.google.com/technologies/ads" className="underline hover:text-[var(--text)]" target="_blank" rel="noopener noreferrer">
              policies.google.com/technologies/ads
            </a>.
          </p>
        </section>

        <section>
          <h2 className="font-semibold mb-1.5">Supercell</h2>
          <p className="text-[var(--muted)]">
            This site is unofficial fan content, not affiliated with, endorsed, sponsored, or specifically
            approved by Supercell, per the{" "}
            <a href="https://supercell.com/en/fan-content-policy/" className="underline hover:text-[var(--text)]" target="_blank" rel="noopener noreferrer">
              Supercell Fan Content Policy
            </a>.
          </p>
        </section>

        <section>
          <h2 className="font-semibold mb-1.5">Changes</h2>
          <p className="text-[var(--muted)]">
            If what the site collects ever changes, this page changes with it, along with the date above.
          </p>
        </section>
      </div>
    </div>
  );
}
