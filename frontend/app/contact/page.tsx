import type { Metadata } from "next";
import ContentPage, { type Content, type Section } from "@/components/ContentPage";
import { CONTACT_EMAIL, GITHUB_ISSUES_URL, GITHUB_REPO_URL } from "@/lib/site";

export const metadata: Metadata = {
  title: "Contact — Brawl Draft",
  description: "How to reach the person behind Brawl Draft with bug reports, feature requests, corrections and questions.",
  alternates: { canonical: "/contact" },
};

// The email section only renders once CONTACT_EMAIL is set in lib/site.ts — a placeholder address
// on a live contact page would be worse than none.
const email: Section[] = CONTACT_EMAIL
  ? [{ heading: "Email", body: `For anything that does not belong on a public tracker (a privacy question, a correction to something on this site, a takedown request, or just a note), email [${CONTACT_EMAIL}](mailto:${CONTACT_EMAIL}).` }]
  : [];

const content: Content = {
  label: "CONTACT",
  title: "Contact",
  intro: "Brawl Draft is a one-person project, so there is no support desk. Everything that comes in does get read. Pick whichever channel fits.",
  sections: [
    ...email,
    { heading: "Bugs and feature requests", body: `The best place for anything about the tool itself is the public issue tracker: [open an issue on GitHub](${GITHUB_ISSUES_URL}). A recommendation that looks wrong, a brawler or map that is missing or misnamed, a page that will not load, or something you wish the board did. Filing it there keeps it visible until it is fixed, and existing issues are public too, so you can check whether yours is already known.\n\nThe [source code](${GITHUB_REPO_URL}) is open under the MIT license if you would rather send a fix than a report.` },
    { heading: "What helps", body: "For a recommendation you think is wrong, the map, the mode and the full state of the board (bans and picks, in order) is enough to reproduce it. For anything to do with your roster or rank, include your player tag. It is public game data and the only way to see what the site saw. A screenshot never hurts." },
    { heading: "Privacy and data", body: "The [Privacy](/privacy) page lists what the site stores and why. It is short, because there are no accounts and no tracking cookies. Anything it does not answer is welcome through the channels above." },
    { heading: "About the game itself", body: "Brawl Draft is unofficial fan content with no connection to Supercell. Questions about your account, purchases, in-game bans, or anything else inside Brawl Stars need to go to Supercell's own support, not here." },
  ],
};

export default function Contact() {
  return <ContentPage content={content} current="/contact" />;
}
