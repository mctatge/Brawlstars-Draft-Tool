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
  intro: "Brawl Draft is a free, experimental public beta for Brawl Stars Ranked. You enter a map, bans and picks, and it ranks options using collected match data, a win-probability model and explicit heuristics. It is also an end-to-end machine-learning project, from collection and evaluation to publication and a usable draft board.\n\nIt has limited live capacity and no demonstrated win-rate lift for its users. The [offline demonstration](/demo) offers a saved first example without waiting for the API. This is an independent, unofficial fan project, not affiliated with, endorsed by, or sponsored by Supercell.",
  sections: [
    { heading: "Who builds it", body: `Brawl Draft is built and maintained by [${AUTHOR_NAME}](${AUTHOR_GITHUB_URL}), an independent developer working on it alone. There is no company behind it and no account to sign up for.\n\nThe whole project is open source under the MIT license (the data pipeline, the model, the draft engine and this site) at [github.com/mctatge/Brawlstars-Draft-Tool](${GITHUB_REPO_URL}). If you want to check how a number was produced, the code that produced it is public.` },
    { heading: "Why it exists", body: "Most draft tools stop at a win rate, a synergy number and a counter number. Brawl Draft started as an experiment: collect real ranked matches at scale, train a model that reads an unfinished draft natively, fuse it with empirical map statistics, and then show every component of every score so you can judge the advice rather than take it on faith.\n\nIt was built to be both a practical tool for ranked drafting and an end-to-end machine-learning project: data pipeline, trained model, product. Collection is scheduled on a home machine, and drift checks can request retraining as the meta moves. Collection, review, evaluation and publication are separate operations: an outage or failed check can leave the previous accepted release in service. The public beta does not promise availability or capacity for a large userbase." },
    { heading: "Where the data comes from", body: "Population statistics and model training use ranked matches collected through the official Brawl Stars API, deduplicated so each match counts once. Live analysis starts at a reviewed balance-era boundary, with recency weighting within that era. Earlier games stay in the archive for research. Other inputs, including composition warnings and parts of the account adjustments, are explicit heuristics.\n\nThe collector is selective, and published model and stats artifacts may have different samples or update times. [The model dossier](/model) reports the served model's training provenance and held-out metrics. [How it works](/how-it-works) explains what those measurements establish and their limits. Brawler and map reference data and images come from Brawlify / BrawlAPI." },
    { heading: "What it does not do", body: "The tool cannot see aim, positioning, rotations or player coordination. Prediction on observed matches does not prove that changing a pick would change the result, and no controlled study has demonstrated a win-rate increase from following its recommendations. The blended pick score is not a calibrated probability that you will win. The [FAQ](/faq) explains the limits.\n\nThe API and home roster service have resource limits. Failed initial roster lookups leave personalization unavailable; failed refreshes preserve only the same account's last successful roster with a visible warning. Initial live loading is bounded and offers a retry, while the [offline demonstration](/demo) stays independent of those services.\n\nThe site has no accounts and no tracking cookies. Cloudflare counts page views without cookies, and the [Privacy](/privacy) page lists what is stored and why." },
    { heading: "Supercell and the Fan Content Policy", body: "Brawl Draft is unofficial fan content published under [Supercell's Fan Content Policy](https://supercell.com/en/fan-content-policy/). It is not affiliated with, endorsed, sponsored, or specifically approved by Supercell, and Supercell is not responsible for it. Brawl Stars is a trademark of Supercell Oy, and the brawler and map art shown on the board is Supercell's." },
    { heading: "Get in touch", body: "Bug reports, feature requests, corrections and questions are all welcome. The [Contact](/contact) page has the details." },
  ],
};

export default function About() {
  return <ContentPage content={content} current="/about" />;
}
