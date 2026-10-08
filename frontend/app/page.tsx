import DraftBoard from "@/components/DraftBoard";
import SiteFooter from "@/components/SiteFooter";

// min-h-screen lives here rather than on the board: the footer is a sibling of DraftBoard (so it
// stays server-rendered), and if the board kept the full-height rule the footer would be pushed
// a whole viewport below the content on tall displays.
export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <aside className="mx-auto w-full max-w-[1240px] px-4 pt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-[var(--muted)]">
        <span>Experimental draft assistant · live beta</span>
        <a href="/demo" className="seg px-3 py-2">Try an example — no account needed →</a>
      </aside>
      <div className="flex-1">
        <DraftBoard />
      </div>
      <SiteFooter blurb="Recommendations fuse a trained win-prob model with empirical map stats" />
    </div>
  );
}
