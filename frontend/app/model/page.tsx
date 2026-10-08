import type { Metadata } from "next";
import "katex/dist/katex.min.css";
import "./model.css";
import DocNav from "@/components/DocNav";
import { DocFooter } from "@/components/ContentPage";
import { Equation } from "@/components/model/Tex";
import MirrorTool from "@/components/model/MirrorTool";
import ModelStatus from "@/components/model/ModelStatus";
import { StrengthPath, CounterPath } from "@/components/model/figures";

export const metadata: Metadata = {
  title: "The Model — Method and Live Evaluation | Brawl Draft",
  description: "How Brawl Draft scores partial teams, evaluates new models, and verifies the statistics for the exact served weights. An experimental drafting assistant.",
  alternates: { canonical: "/model" },
  openGraph: { type: "website", siteName: "Brawl Draft", title: "The Model — Method and Live Evaluation",
    description: "Equations, evaluation safeguards and limits of the experimental draft assistant.",
    url: "/model", images: ["/opengraph-image.png"] },
};

export default function Page() {
  return <>
    <div style={{ maxWidth: 1180, margin: "0 auto", padding: "20px clamp(16px, 4vw, 56px)" }}><DocNav current="/model" /></div>
    <main className="dossier"><div className="wrap">
      <header className="py-10 md:py-16 narrow">
        <p className="eyebrow">Experimental draft assistant</p>
        <h1 className="display text-4xl md:text-6xl mb-5">What the model knows.</h1>
        <p className="lede">A small neural model reads the map and the known picks. The board blends its estimate with empirical matchup statistics. Neither the estimate nor the blended draft score measures how much following a suggestion will improve your win rate.</p>
        <p><a href="/demo">Try a recorded draft without an account or a live connection →</a></p>
      </header>
      <section><h2 className="text-2xl mb-5">The artifact being served</h2>
        <p>These figures come from an evaluation report bound by SHA-256 to the loaded weights. If the API is asleep or a legacy model has no verified report, the page leaves current metrics unavailable. Historical experiments remain in the repository.</p>
        <ModelStatus />
        <p className="mt-5">Log loss measures probability error; lower is better. AUC measures ordering of winners and losers, where 0.5 is chance. ECE summarizes calibration in bins. Good average calibration does not establish accuracy for every map, rank, partial board or new balance patch.</p>
      </section>
      <section><h2 className="text-2xl mb-5">One score for each side</h2>
        <p>Each brawler has learned strength and counter vectors. Map and mode embeddings supply context. A shared network scores team strength, a directed counter term compares the teams, and an optional class-pair term captures within-team composition.</p>
        <Equation tex={String.raw`z(A,B\mid c)=S(A,c)-S(B,c)+P_A^\top Q_B-P_B^\top Q_A+T(A)-T(B)`} />
        <Equation tex={String.raw`S(T,c)=\operatorname{MLP}\!\left(\left[\frac{1}{3}\sum_{b\in T}e_b,\ c\right]\right),\quad T(T)=\sum_{i<j}M_{\operatorname{class}(i),\operatorname{class}(j)}`} />
        <div className="grid md:grid-cols-2 gap-5 my-6"><figure className="figwrap overflow-x-auto"><StrengthPath /><figcaption>The same strength network scores both teams.</figcaption></figure><figure className="figwrap overflow-x-auto"><CounterPath /><figcaption>Directed interactions describe the matchup.</figcaption></figure></div>
        <p>Swapping teams negates every term. Applying the sigmoid therefore makes the two predicted win probabilities add to one. This is an architectural identity; it is not evidence that the predictions are correct.</p>
        <Equation tex={String.raw`p(A,B)=\sigma(z),\qquad p(A,B)+p(B,A)=\sigma(z)+\sigma(-z)=1`} />
        <MirrorTool />
      </section>
      <section className="narrow"><h2 className="text-2xl mb-5">Unknown picks stay unknown</h2>
        <p>During training, slots are hidden behind a learned unknown-slot embedding. At inference, the model pads partial teams to three slots with that embedding. It estimates outcomes associated with similar partial teams in historical play, rather than simulating a perfect opponent.</p>
        <p>The match log contains completed teams, not actual pick order or bans. Masking creates partial-board training examples; it cannot recover which player picked first, what alternatives were available, or why a team chose a brawler. Unseen brawlers and maps use fallback embeddings until a compatible retrain is published.</p>
      </section>
      <section className="narrow"><h2 className="text-2xl mb-5">The recommendation score</h2>
        <p>The engine averages the active model, map, counter, synergy and role signals, then renormalizes their weights. Counters need revealed enemies and synergy needs known allies. A loaded account adds fieldability checks and bounded readiness and personal-history adjustments. The displayed draft score is a ranking signal, not a calibrated win probability.</p>
        <Equation tex={String.raw`\operatorname{base}(b)=\frac{\sum_{k\in\mathcal A}\omega_k v_k(b)}{\sum_{k\in\mathcal A}\omega_k}`} />
        <p>Empirical statistics use the reviewed balance-era cutoff and recency decay within that era. Older history remains useful for research but does not enter current live statistics. A model from an older era is disabled until a compatible model passes publication checks.</p>
      </section>
      <section className="narrow"><h2 className="text-2xl mb-5">Evaluation and publication</h2>
        <p>The retraining workflow downloads the actual released NumPy model as its incumbent. It splits eligible matches chronologically into training, selection and final test windows, keeping equal timestamps together. Training and candidate choice use the first two windows; final metrics and the paired regression check use the last.</p>
        <p>Publication fails if the incumbent is missing, invalid or incompatible; if fresh test data is insufficient; if the allowed full-team log-loss regression is exceeded; or if NumPy export and capability checks fail. Calibration and first-pick comparison against empirical rates are reported as diagnostics. They are not enforced performance thresholds.</p>
        <p>The first migration from legacy weights needs an explicit bootstrap because their training-window metadata is missing. Its overlap is labeled unknown. Later test windows exclude the incumbent snapshot and previously reserved evaluation attempts.</p>
        <p>Weights and metrics are uploaded to an immutable release and verified before a small pointer is promoted. Serving validates both files and keeps its last usable bundle if an update fails. Tests run in pull-request CI; a full retrain runs only in the separate retraining workflow.</p>
      </section>
      <section className="narrow"><h2 className="text-2xl mb-5">What remains unproven</h2>
        <p>The crawler starts from leaderboard players, so its sample differs from the whole player population. Skill, inventory, map geometry, positioning and execution are only partly observed or absent. Observed win-rate associations do not establish that changing a pick causes the predicted improvement.</p>
        <p>The public beta runs on a small Render instance with an optional home-hosted roster service. Capacity and availability are limited. The recorded example remains usable when those services are unavailable; broad concurrent usage has not been validated.</p>
        <p><a href="https://github.com/mctatge/Brawlstars-Draft-Tool/blob/main/docs/MODEL_CARD.md">Model card</a> · <a href="https://github.com/mctatge/Brawlstars-Draft-Tool/blob/main/docs/model-evaluation.md">Historical ablations</a></p>
      </section>
    </div></main>
    <div className="max-w-4xl mx-auto px-5 pb-8"><DocFooter current="/model" /></div>
  </>;
}
