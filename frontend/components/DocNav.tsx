import Logo from "@/components/Logo";

// Site-wide nav for the written pages. The board lives at "/", so it leads.
export const NAV = [
  { href: "/", label: "Draft board" },
  { href: "/purchases", label: "Upgrades" },
  { href: "/guide", label: "Draft guide" },
  { href: "/guides", label: "Mode guides" },
  { href: "/how-it-works", label: "How it works" },
  { href: "/model", label: "The model" },
  { href: "/faq", label: "FAQ" },
];

// Tactical top bar shared by the written pages, so the docs read as the same console as the
// board: mono nav, hairline rules, sharp corners. Long-form prose stays in the readable sans.
export default function DocNav({ current }: { current: string }) {
  return (
    <nav className="panel flex flex-wrap items-center gap-x-1 gap-y-1 px-3 py-2 mb-8">
      <a href="/" className="flex items-center gap-2 mr-3" aria-label="Brawl Draft home">
        <Logo size={22} />
        <span className="brand-gradient text-[14px]">BRAWL DRAFT</span>
      </a>
      <span className="label hidden sm:inline mr-2">// DOCS</span>
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
