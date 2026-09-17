import type { Metadata } from "next";
import ContentPage, { type Content } from "@/components/ContentPage";
import { AUTHOR_GITHUB_URL, AUTHOR_NAME, GITHUB_REPO_URL } from "@/lib/site";

export const metadata: Metadata = {
  title: "About — Brawl Draft",
  description: "Who builds Brawl Draft and why: an independent, open-source, unofficial fan project. An AI ranked-draft assistant for Brawl Stars, not affiliated with Supercell.",
  alternates: { canonical: "/about" },
};

// Prose lives here as data so ContentPage owns all the layout and typography. Keep this factual:
// it exists for transparency (who runs the site, where the data comes from, who it is not).
const content: Content = {
  label: "ABOUT",
  title: "About Brawl Draft",
  intro: "Brawl Draft is a free draft assistant for Brawl Stars Ranked. You set the map, tap the bans and picks as they happen, and it ranks what to take next using real ranked match data and a trained win-probability model.\n\nIt is an independent, unofficial fan project. It is not affiliated with, endorsed by, or sponsored by Supercell.",
  sections: [
    { heading: "Who builds it", body: `Brawl Draft is built and maintained by [${AUTHOR_NAME}](${AUTHOR_GITHUB_URL}), an independent developer working on it alone. There is no company behind it and no account to sign up for.\n\nThe whole project is open source under the MIT license (the data pipeline, the model, the draft engine and this site) at [github.com/mctatge/Brawlstars-Draft-Tool](${GITHUB_REPO_URL}). If you want to check how a number was produced, the code that produced it is public.` },
    { heading: "Why it exists", body: "Most draft tools stop at a win rate, a synergy number and a counter number. Brawl Draft started as an attempt to do the whole thing properly: collect real ranked matches at scale, train a model that reads an unfinished draft natively, fuse it with empirical map statistics, and then show every component of every score so you can judge the advice rather than take it on faith.\n\nIt was built to be both a practical tool for ranked drafting and an end-to-end machine-learning project: data pipeline, trained model, product. It is only useful while the data keeps flowing, so the collector runs continuously and the model is retrained as the meta moves." },
    { heading: "Where the data comes from", body: "Every number traces back to real ranked matches collected through the official Brawl Stars API, deduplicated so each match counts once. The current model was trained on more than a million of them, with a held-out set that was never shown to it during training.\n\nBrawler and map reference data and images come from Brawlify / BrawlAPI. [How it works](/how-it-works) is the plain-English walkthrough of the pipeline and the model's real accuracy; [the model dossier](/model) has the architecture, the equations and the held-out numbers." },
    { heading: "What it does not do", body: "It cannot see how well anyone plays. Ranked outcomes are mostly decided by aim, positioning and rotations, none of which a draft can see, so the model's edge over a coin flip is real but modest. It is used to rank picks relative to each other, not to predict your game. The [FAQ](/faq) goes into the limits in more detail.\n\nThe site has no accounts and no tracking cookies. Cloudflare counts page views without cookies, and the [Privacy](/privacy) page lists exactly what is stored and why." },
    { heading: "Supercell and the Fan Content Policy", body: "Brawl Draft is unofficial fan content published under [Supercell's Fan Content Policy](https://supercell.com/en/fan-content-policy/). It is not affiliated with, endorsed, sponsored, or specifically approved by Supercell, and Supercell is not responsible for it. Brawl Stars is a trademark of Supercell Oy, and the brawler and map art shown on the board is Supercell's." },
    { heading: "Get in touch", body: "Bug reports, feature requests, corrections and questions are all welcome. The [Contact](/contact) page has the details." },
  ],
};

export default function About() {
  return <ContentPage content={content} current="/about" />;
}
