# AdSense Go-Live — enabling the env-gated ad slots

The frontend ships with AdSense wired but dark. This is the checklist for turning ads on
without breaking ad serving. Read before touching `frontend/components/AdSlot.tsx`,
`frontend/app/layout.tsx`'s ad loader, or `frontend/public/ads.txt`.

## How the gating works

`components/AdSlot.tsx` + the script loader in `app/layout.tsx` render **nothing** until
both env vars are set in the Cloudflare Pages build environment:

- `NEXT_PUBLIC_ADSENSE_CLIENT` — the `ca-pub-…` publisher id.
- `NEXT_PUBLIC_ADSENSE_SLOT_FOOTER` — the footer slot id.

Both are inlined at build time (static export), so enabling ads requires a Pages rebuild,
not just an env change.

## Go-live checklist

1. Complete AdSense console setup (site added, review passed).
2. Set the two env vars above in the Cloudflare Pages build env.
3. **In the SAME commit**, add `frontend/public/ads.txt` with the real
   `google.com, pub-…, DIRECT, …` line. A served `ads.txt` **without** the publisher line
   halts ad serving ("Unauthorized").

## Review requirements: About and Contact pages

Google's review of the application (submitted 2026-08-10) asked for an "About Us" and a
"Contact Us" page. They live at `/about` and `/contact` (`frontend/app/about/page.tsx`,
`frontend/app/contact/page.tsx`, both `ContentPage` prose pages), are linked from every footer
through `SITE_LINKS` in `frontend/components/DocNav.tsx`, and are in `app/sitemap.ts`. Keep
them reachable from the footer — the reviewer crawls the exported HTML.

Site identity (repo URL, author, contact address) lives in `frontend/lib/site.ts`. No contact
address is published anywhere in the repo, so `CONTACT_EMAIL` there is a `null` placeholder and
the Contact page shows only the GitHub issue tracker until it is set. Set it to a dedicated
mailbox before resubmitting if the reviewer wants an email on the page.

## Brand assets

AdSense uploads (logo rasters, 5:1 light-theme PNG) live under `frontend/public/brand/` —
see the README there.
