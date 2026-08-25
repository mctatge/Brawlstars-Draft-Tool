/**
 * /model — the technical companion to /how-it-works: the same system, but with the equations,
 * the architecture diagrams and the held-out numbers instead of the prose.
 *
 * Every figure quoted here traces to a source of truth in the repo — docs/metrics.json for the
 * evaluation numbers, backend/bsdraft/engine/scoring.py for the blend weights (the signal-share
 * chart derives its geometry from them), and the shipped winprob.npz for the parameter table.
 * When any of those move, this page moves with them. It descends from a standalone HTML dossier
 * (docs/winprob-visual.html), retired on 2026-08-21 once this page superseded it — recover it from
 * git history rather than reviving it, since two copies of this story will drift.
 *
 * Math is typeset by KaTeX at build time (see components/model/Tex.tsx), so the exported HTML
 * already holds the equations and the page ships no math JavaScript.
 *
 * Note on prose: Turbopack drops the leading space of a JSX text node that follows an inline
 * element when that node wraps across source lines — hence the explicit {" "} in a few places.
 */
import type { Metadata } from "next";
import "katex/dist/katex.min.css";
import "./model.css";
import DocNav from "@/components/DocNav";
import { Tex, Equation } from "@/components/model/Tex";
import MirrorTool from "@/components/model/MirrorTool";
import StateGrid from "@/components/model/StateGrid";
import {
  StrengthPath, CounterPath, LogitJoin, ExtraRows, DecayClocks, TrainLoop,
  EdgeCalibration, SignalShares, StackingSplit, WEIGHTS,
} from "@/components/model/figures";

export const metadata: Metadata = {
  title: "The Model — 12,505 Parameters | Brawl Draft",
  description:
    "The machine learning behind Brawl Draft, in full: an antisymmetric embedding network trained on 1,615,154 ranked matches, with the equations, the architecture diagrams, the held-out numbers, and the limits.",
  alternates: { canonical: "/model" },
  openGraph: {
    // A route-level openGraph REPLACES the root layout's rather than merging with it, so the
    // card's type/siteName/image have to be restated here or /model unfurls as a bare link.
    type: "website",
    siteName: "Brawl Draft",
    title: "The Model — 12,505 Parameters",
    description: "Every equation, diagram and held-out number behind the win-probability net that ranks picks on the draft board.",
    url: "/model",
    images: ["/opengraph-image.png"],
  },
};

// Team colours reused inside the TeX so the math carries the same functional encoding as the
// rest of the site: blue = your team, red = the enemy. (Kept in sync with globals.css.)
const A = "\\textcolor{#3b82f6}{A}";
const B = "\\textcolor{#ff3b30}{B}";

function SecHead({ n, title, lede }: { n: string; title: string; lede: React.ReactNode }) {
  return (
    <div className="sec-head">
      <div className="sec-num">{n}</div>
      <div>
        <h2>{title}</h2>
        <p className="lede">{lede}</p>
      </div>
    </div>
  );
}

function Fig({ children, caption }: { children: React.ReactNode; caption: React.ReactNode }) {
  return (
    <figure className="figwrap">
      <div className="figscroll">{children}</div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export default function Page() {
  // Shares of a personalized board's score that come from the three hand-set heuristics.
  const divisorPersonal = Object.values(WEIGHTS).reduce((a, b) => a + b, 0);
  const heuristicShare = (WEIGHTS.role + WEIGHTS.mastery + WEIGHTS.personal) / divisorPersonal;
  const modelPersonal = WEIGHTS.model / divisorPersonal;

  return (
    <>
      {/* DocNav is shared site chrome and stays OUTSIDE .dossier: inside it, the dossier's
          unlayered `.panel` padding and `a` styling would beat the nav's Tailwind utilities
          (unlayered CSS wins over @layer utilities), rendering it taller here than on every
          other doc page. The inline box reproduces .wrap so it still lines up with the page. */}
      <div style={{ maxWidth: 1180, margin: "0 auto", padding: "20px clamp(16px, 4vw, 56px) 0" }}>
        <DocNav current="/model" />
      </div>

    <div className="dossier">
      <div className="spine" aria-hidden="true" />

      <nav className="rail" aria-label="Sections">
        <a href="#eq"><span>01</span> The equation</a>
        <a href="#mirror"><span>02</span> The mirror</a>
        <a href="#rows"><span>03</span> Three +1 rows</a>
        <a href="#mask"><span>04</span> Unknown slots</a>
        <a href="#train"><span>05</span> Training</a>
        <a href="#worth"><span>06</span> What it&apos;s worth</a>
        <a href="#twice"><span>07</span> Written twice</a>
        <a href="#blend"><span>08</span> One of seven</a>
        <a href="#limits"><span>09</span> What it can&apos;t do</a>
      </nav>

      <div className="page">
        {/* ============================ MASTHEAD ============================ */}
        <header className="wrap masthead" style={{ paddingTop: 0 }}>
          <div className="eyebrow">Brawl Draft · model dossier · winprob.npz</div>
          <h1>Yes, it&apos;s a neural network.<br />The whole thing fits in 12,505 numbers.</h1>
          <p className="deck">
            The model ranking picks on the draft board is a genuine neural network — learned embeddings,
            a two-layer MLP, backprop over a million ranked matches. It is also small enough to print.
            Here is every parameter it has, every decision baked into its shape, and every number it earned.
          </p>

          <div className="stats masthead-stats">
            <div className="stat"><div className="k">Trained parameters</div><div className="v tA">12,505</div><div className="n">Nine tensors. 70% of the weights are embedding lookups.</div></div>
            <div className="stat"><div className="k">Ranked matches</div><div className="v">1,615,154</div><div className="n">Deduped, recency-weighted, real ranked play.</div></div>
            <div className="stat"><div className="k">Calibration error</div><div className="v green">0.011</div><div className="n">Says 60%, wins ~60%. The headline result.</div></div>
            <div className="stat"><div className="k">Held-out AUC</div><div className="v">0.630</div><div className="n">Full comps. The draft is not the whole game.</div></div>
            <div className="stat"><div className="k">Empty board</div><div className="v gold">0.5000</div><div className="n">Not learned. True by construction.</div></div>
          </div>

          <div className="hero-eq">
            <Equation
              title="The entire model"
              tex={[
                `\\ell(${A}, ${B} \\mid c) \\;=\\; \\underbrace{\\big(S(${A}, c) - S(${B}, c)\\big)}_{\\text{team strength}} \\;+\\; \\underbrace{\\big(P_{${A}} \\cdot Q_{${B}} - P_{${B}} \\cdot Q_{${A}}\\big)}_{\\text{directed counters}}`,
                `P(${A} \\text{ beats } ${B} \\mid c) \\;=\\; \\sigma\\big(\\ell(${A}, ${B} \\mid c)\\big) \\;=\\; \\frac{1}{1 + e^{-\\ell}}`,
              ]}
            >
              <b>c</b> is the map and mode. <b>S</b> scores a team in that context. <b>P</b> and <b>Q</b>{" "}are each
              brawler&apos;s attacker and defender vectors. Every term flips sign when you swap the teams — which is
              the reason the two probabilities always sum to exactly one.
            </Equation>
          </div>
        </header>

        {/* ============================ 01 THE EQUATION ============================ */}
        <section id="eq" className="wrap">
          <SecHead n="01" title="One equation, two halves"
            lede="The first half asks how good each team is here. The second asks who beats whom. Neither half can express a preference for going first." />

          <div className="stack">
            <Equation
              title="The two halves, written out"
              tex={[
                `c = \\big[\\, e_{\\text{map}},\\; e_{\\text{mode}} \\,\\big] \\in \\mathbb{R}^{16+8}`,
                `S(T, c) = \\mathrm{MLP}\\!\\left(\\left[\\; \\frac{1}{|T|}\\sum_{b \\in T} E_b, \\;\\; c \\;\\right]\\right), \\qquad P_T = \\sum_{b \\in T} p_b, \\qquad Q_T = \\sum_{b \\in T} q_b`,
                `\\mathrm{MLP}(x) = W_2\\,\\mathrm{ReLU}(W_1 x + b_1) + b_2, \\qquad W_1 \\in \\mathbb{R}^{64 \\times 56},\\;\\; W_2 \\in \\mathbb{R}^{1 \\times 64}`,
              ]}
            >
              <b>E<sub>b</sub></b> ∈ ℝ<sup>32</sup> is a brawler&apos;s learned embedding; <b>p<sub>b</sub></b>,{" "}
              <b>q<sub>b</sub></b> ∈ ℝ<sup>16</sup> are its attacker and defender vectors. There is no bias term
              on the logit, and no calibration layer after the sigmoid — every number the board shows comes out
              of these three lines.
            </Equation>

            <Fig caption={<>
              <b>The strength half.</b> Both teams are pushed through the same embedding table and the same
              two-layer network — the dashed ties are shared weights, not copies. Because the score is read off
              the <b>mean</b>{" "}of a team&apos;s embeddings, pick order cannot matter; because only the difference
              survives, a global &ldquo;team A is better&rdquo; offset has nowhere to live.
            </>}><StrengthPath /></Fig>

            <Fig caption={<>
              <b>The counter half.</b> Every brawler carries two 16-dimensional vectors: what it does to others,
              and what others do to it. Only <b>cross-team</b>{" "}products are ever formed — our attack against their
              defence, minus theirs against ours. The X across the centreline is the mechanism: it is what lets
              the model say &ldquo;this brawler beats that one&rdquo; rather than merely &ldquo;this brawler is
              good.&rdquo;
            </>}><CounterPath /></Fig>

            <Fig caption={<>
              <b>The join.</b> No bias term, no offset, no calibration layer bolted on afterwards — the logit is
              the sum of two antisymmetric quantities, and the sigmoid is the only nonlinearity between it and
              the number shown on the board.
            </>}><LogitJoin /></Fig>

            <div className="callout">
              <div className="kicker">Two poolings, deliberately different</div>
              <p>The strength path takes the <b className="gold">mean</b>{" "}of a team&apos;s three embeddings; the
              counter path takes the <b className="gold">sum</b> of them. Same tensor shapes, same arrow on a
              diagram, opposite meaning: the mean makes strength a property of the team as a unit (and lets an
              unknown slot average in), while the sum makes each brawler contribute its own matchup vector. This
              asymmetry is hand-duplicated in the NumPy server — change one side only and nothing breaks loudly.
              It just quietly computes a different function.</p>
            </div>

            <div className="grid-2">
              <div className="tablewrap">
                <table>
                  <caption className="tcap">Every parameter in the shipped artifact</caption>
                  <thead><tr><th>Tensor</th><th>Shape</th><th>Params</th></tr></thead>
                  <tbody>
                    <tr><td>brawler.weight</td><td>108 × 32</td><td>3,456</td></tr>
                    <tr><td>counter_p.weight</td><td>108 × 16</td><td>1,728</td></tr>
                    <tr><td>counter_q.weight</td><td>108 × 16</td><td>1,728</td></tr>
                    <tr><td>map_emb.weight</td><td>114 × 16</td><td>1,824</td></tr>
                    <tr><td>mode_emb.weight</td><td>7 × 8</td><td>56</td></tr>
                    <tr><td>strength.0.weight</td><td>64 × 56</td><td>3,584</td></tr>
                    <tr><td>strength.0.bias</td><td>64</td><td>64</td></tr>
                    <tr><td>strength.3.weight</td><td>1 × 64</td><td>64</td></tr>
                    <tr><td>strength.3.bias</td><td>1</td><td>1</td></tr>
                    <tr className="hi"><td>total</td><td>9 tensors</td><td>12,505</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="panel panel-2">
                <div className="kicker">What the budget says</div>
                <p><b>8,792 of 12,505 parameters — 70% — are embedding lookups</b>, 6,912 of them per-brawler.
                The network proper, the part that does the arithmetic, is the other 3,713: one 56×64 layer, one
                64×1 layer, and their biases.</p>
                <p>That ratio is the honest description of this model. It is mostly a learned table of what each
                brawler is and who it beats, with a small opinion attached about how to combine them given the
                map. Nothing here needs a GPU, and inference is nine matrix operations on arrays that fit in a
                phone&apos;s L2 cache.</p>
                <p className="dim">The nine tensors come from six parameter-bearing modules: five embedding
                tables and one two-layer MLP.</p>
              </div>
            </div>
          </div>
        </section>

        {/* ============================ 02 THE MIRROR ============================ */}
        <section id="mirror" className="wrap">
          <SecHead n="02" title="The mirror is not trained. It is built in."
            lede="Swap the two teams and every term in the logit changes sign. That one property removes a whole category of things the model would otherwise have to learn." />

          <div className="grid-2">
            <div className="panel">
              <div className="kicker">Drag the logit</div>
              <MirrorTool />
            </div>
            <div className="stack">
              <Equation
                tex={[
                  `\\ell(${B}, ${A} \\mid c) = -\\,\\ell(${A}, ${B} \\mid c) \\qquad\\text{and}\\qquad \\sigma(-z) = 1 - \\sigma(z)`,
                  `\\Longrightarrow\\quad P(${A} \\text{ wins}) + P(${B} \\text{ wins}) = \\sigma(z) + \\sigma(-z) = 1`,
                ]}
              >
                Holds identically, for every set of weights the optimizer could ever reach — including the random
                ones at initialization.
              </Equation>
              <ul className="clean">
                <li><b>No team-order bias to unlearn.</b> The raw data has team A winning 51% of the time, an
                artifact of how the battle log is read. A model with a bias term would spend capacity fitting
                that. This one cannot represent it.</li>
                <li><b>No swap augmentation.</b> The usual trick — feed every match twice, once with the teams
                flipped — buys nothing here, so training does half the work.</li>
                <li><b>An empty board returns exactly 0.5.</b> Both sides become three identical mask rows, so the
                strength difference is zero and the crossed counter terms cancel term for term. Verified
                numerically on the shipped artifact: <code>0.5</code>, not <code>0.4998</code>.</li>
                <li><b>The empty board is excluded from training</b> for the same reason — its gradient is
                identically zero. It is correct for free, so spending compute on it would be waste.</li>
              </ul>
            </div>
          </div>
        </section>

        {/* ============================ 03 THREE +1 ROWS ============================ */}
        <section id="rows" className="wrap">
          <SecHead n="03" title="Three tables have an extra row. Every one means something different."
            lede="108 is not 107+1 in the same way that 114 is 113+1. Conflating them is the easiest way to misread this model." />

          <Fig caption={<>
            <b>Four extra rows, four different jobs.</b>{" "}The brawler table&apos;s extra row is the model&apos;s
            vocabulary for &ldquo;not picked yet.&rdquo; The map and mode tables reserve row 0 for an unknown
            bucket that training never touches and the pinned vocabulary no longer routes to — dead weight, 24 of
            12,505 parameters. The fourth row does not exist in the artifact at all; it is computed when the file
            is loaded.
          </>}><ExtraRows /></Fig>
        </section>

        {/* ============================ 04 UNKNOWN SLOTS ============================ */}
        <section id="mask" className="wrap">
          <SecHead n="04" title="A half-empty board is an input, not a gap"
            lede="A draft assistant is always asked about incomplete teams. The model was taught to read them rather than to guess the missing picks." />

          <StateGrid />

          <div className="grid-2 mt">
            <div className="panel">
              <div className="kicker">What masking actually does</div>
              <p>Every epoch, every match is re-masked from scratch. With probability 0.7 it keeps its full 3v3;
              otherwise one of the 14 partial states is drawn uniformly, and a random subset of each team&apos;s
              slots is overwritten with the mask row.</p>
              <p><b>The label never changes.</b>{" "}A masked row still carries the real outcome, so the model is not
              learning &ldquo;who wins from here&rdquo; — it is learning</p>
              <div style={{ margin: "14px 0" }}>
                <Equation tex={`\\hat p = \\Pr\\big(\\text{win} \\mid \\text{the known picks are on the final teams}\\big)`} />
              </div>
              <p>marginalized over every way real drafts continued.</p>
              <p className="dim">Fresh masks each epoch make this free augmentation: a million matches become a
              different million every pass.</p>
            </div>
            <div className="stack">
              <Equation
                title="Why masking cannot break the mirror"
                tex={`\\underbrace{(3-k_{${A}})(3-k_{${B}})\\,(p_\\varnothing \\cdot q_\\varnothing)}_{\\text{in } P_{${A}} \\cdot Q_{${B}}} \\;-\\; \\underbrace{(3-k_{${B}})(3-k_{${A}})\\,(p_\\varnothing \\cdot q_\\varnothing)}_{\\text{in } P_{${B}} \\cdot Q_{${A}}} \\;=\\; 0`}
              >
                Mask-versus-mask contributions appear identically on both sides of the difference and cancel
                exactly, whatever the mask row learned. So antisymmetry — and the exact 0.5 on an empty board —
                survives partial drafts for free.
              </Equation>
              <div className="panel panel-2">
                <div className="kicker">The mismatch it cannot fix</div>
                <p>Training masks a <b>random subset</b> of the final team. At inference, the picks you know are
                the <b>early</b> picks — and early picks are not a random sample of final teams.</p>
                <p>Measuring that gap needs pick order, which the battle log never records. It is a known,
                unquantified conditioning error, and it is the reason the partial-draft numbers below are
                described as population averages rather than as forecasts.</p>
              </div>
            </div>
          </div>
        </section>

        {/* ============================ 05 TRAINING ============================ */}
        <section id="train" className="wrap">
          <SecHead n="05" title="Fit to a meta that keeps moving"
            lede="Recency-weighted cross-entropy, two validation heads that answer different questions, and a gate that refuses to publish a regression." />

          <div className="stack">
            <Equation
              title="Objective"
              tex={[
                `\\mathcal{L}(\\theta) = -\\sum_i w_i \\Big[\\, y_i \\log \\hat p_i + (1 - y_i)\\log(1 - \\hat p_i) \\,\\Big]`,
                `w_i = 2^{-(t_{\\max} - t_i)/\\tau}, \\qquad \\tau = 30 \\text{ days}, \\qquad \\textstyle\\frac{1}{N}\\sum_i w_i = 1`,
              ]}
            >
              Ordinary binary cross-entropy, with every match discounted by how old it is. A balance patch does
              not invalidate the past so much as demote it. <b>t<sub>max</sub></b> is the newest match in the
              dataset, not the wall clock.
            </Equation>

            <Fig caption={<>
              <b>Two clocks, on purpose.</b> The net leans on a 30-day half-life; the count-based win-rate tables
              it is blended with use 21 days. The reference point is the newest match <em>in the dataset</em>, not
              the wall clock — so if the crawler stalls, weights freeze rather than silently sliding toward
              uniform.
            </>}><DecayClocks /></Fig>

            <Fig caption={<>
              <b>Chosen by one metric, gated by another.</b>{" "}Early stopping watches the masked mixture — the
              model&apos;s actual job. The publish gate watches unmasked full comps against the <em>previous
              checkpoint on the same rows</em>, the only comparison that is free of data drift. Fail it and
              training exits without writing the model, the metrics, or the charts.
            </>}><TrainLoop /></Fig>

            <div className="grid-2">
              <div className="tablewrap">
                <table>
                  <caption className="tcap">Defaults — and the unattended retrain runs with every one of them</caption>
                  <thead><tr><th>Knob</th><th>Value</th><th>What it governs</th></tr></thead>
                  <tbody>
                    <tr><td>--p-full</td><td>0.70</td><td>Share of rows kept as full 3v3; the rest are masked</td></tr>
                    <tr><td>--halflife-days</td><td>30.0</td><td>Recency decay on the loss</td></tr>
                    <tr><td>--max-full-delta</td><td>0.002</td><td>Hard publish gate vs the previous checkpoint</td></tr>
                    <tr><td>--epochs / patience</td><td>40 / 6</td><td>Ceiling and early stop on masked val log-loss</td></tr>
                    <tr><td>--batch / --lr</td><td>256 / 1e-3</td><td>AdamW, weight decay 1e-4, no schedule</td></tr>
                    <tr><td>--val-frac / --seed</td><td>0.15 / 0</td><td>Seeded random split; three separate mask RNG streams</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="panel panel-2">
                <div className="kicker">The gate cuts both ways</div>
                <p>A no-regression gate stops a bad retrain from shipping. It can also lock a good one out — and
                it did. On <b>20 Aug 2026</b> the gate was found to have refused <b>38 consecutive</b> retrains
                (deltas +0.0022 to +0.0031), freezing the served model at 12 Aug while the meta report kept
                announcing a shifted meta.</p>
                <p>Two things made it invisible: the only trace was a line in a crawl log, and the crawler had
                been IP-blocked for part of that window, so those retrains were refitting a dataset that never
                grew. The collector now counts consecutive failures across restarts and files an issue after
                three.</p>
                <p className="dim">Before moving the threshold, a sweep over <code>--p-full</code> × seeds runs
                with the gate disabled, to separate a frozen dataset from a threshold below the run-to-run noise
                floor from a real ratchet against a lucky incumbent.</p>
              </div>
            </div>
          </div>
        </section>

        {/* ============================ 06 WHAT IT'S WORTH ============================ */}
        <section id="worth" className="wrap">
          <SecHead n="06" title="What 12,505 parameters actually buy"
            lede="Held out: 242,273 matches the model never saw. The interesting result is not the accuracy. It is that the probabilities can be trusted." />

          <div className="stack">
            <div className="tablewrap">
              <table>
                <caption className="tcap">Full comps, held-out split</caption>
                <thead><tr><th>Predictor</th><th>Log-loss ↓</th><th>Accuracy ↑</th><th>AUC ↑</th><th>ECE ↓</th></tr></thead>
                <tbody>
                  <tr><td>Always 0.5</td><td>0.6931</td><td>0.500</td><td>—</td><td>—</td></tr>
                  <tr><td>Logistic regression on presence</td><td>0.6841</td><td>0.552</td><td>0.574</td><td>—</td></tr>
                  <tr><td>Previous unmasked checkpoint</td><td>0.6622</td><td>0.595</td><td>0.636</td><td>0.011</td></tr>
                  <tr className="hi"><td>Shipped net (masked, p-full 0.7)</td><td>0.6650</td><td>0.590</td><td>0.630</td><td>0.011</td></tr>
                </tbody>
              </table>
            </div>
            <p className="note-line">Partial-draft support cost <span className="mono">+0.0028</span> log-loss and{" "}
              <span className="mono">−0.0062</span> AUC against the previous, unmasked checkpoint scored on the same
              held-out rows. Read that as the going rate for reading half-empty boards, not as a clean ablation:
              the earlier checkpoint was fit to an earlier snapshot of the crawl, so data growth is folded into the
              same delta. A 50/50 masked mixture cost three times as much and bought nothing extra.</p>

            <Equation
              title="The two metrics that matter, defined"
              tex={[
                `\\mathrm{ECE} = \\sum_{m=1}^{M} \\frac{|B_m|}{N}\\,\\Big|\\, \\mathrm{acc}(B_m) - \\mathrm{conf}(B_m) \\,\\Big|`,
                `\\text{edge} = \\frac{1}{N}\\sum_{i=1}^{N} \\big|\\, \\hat p_i - 0.5 \\,\\big|`,
              ]}
            >
              <b>ECE</b>{" "}bins predictions by confidence and asks how far each bin&apos;s realised win rate sits
              from what was promised — 0.011 means the promise is kept to within a percentage point.{" "}
              <b>Edge</b> is not a quality metric at all: it is how much the model is willing to claim, which is
              only meaningful next to an ECE that stays flat.
            </Equation>

            <Fig caption={<>
              <b>Confidence grows with information; calibration stays put.</b> Bars are the average edge the model
              claims, <Tex>{"|\\hat p - 0.5|"}</Tex>, which roughly triples from a single known pick to a full
              board. The line is expected calibration error, which stays inside a percentage point at every state —
              0.011 to 0.014, with the one bump at 1v1. A model that got louder as it learned more without staying
              honest would climb with the bars.
            </>}><EdgeCalibration /></Fig>

            <div className="grid-2">
              <div className="tablewrap">
                <table>
                  <caption className="tcap">Per draft state, whole val split masked to each</caption>
                  <thead><tr><th>State</th><th>Log-loss ↓</th><th>AUC ↑</th><th>ECE ↓</th><th>edge</th></tr></thead>
                  <tbody>
                    <tr><td>1v0</td><td>0.6910</td><td>0.538</td><td>0.013</td><td>0.034</td></tr>
                    <tr><td>1v1</td><td>0.6865</td><td>0.564</td><td>0.014</td><td>0.050</td></tr>
                    <tr><td>2v1</td><td>0.6828</td><td>0.579</td><td>0.013</td><td>0.064</td></tr>
                    <tr><td>2v2</td><td>0.6769</td><td>0.599</td><td>0.013</td><td>0.077</td></tr>
                    <tr><td>3v2</td><td>0.6723</td><td>0.612</td><td>0.011</td><td>0.088</td></tr>
                    <tr className="hi"><td>3v3</td><td>0.6650</td><td>0.630</td><td>0.011</td><td>0.099</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="panel panel-2">
                <div className="kicker">The admission in the middle of the table</div>
                <p>At <span className="mono">1v0</span> — one pick known, nothing else — the net scores{" "}
                <b>0.6910</b>. A shrunk per-map win-rate marginal, arithmetic anyone could do in a spreadsheet,
                scores <b>0.6906</b>.</p>
                <p>The net loses, by 0.0004. At the lowest-information state it adds nothing beyond the raw
                statistic, which the blend already carries separately. It is also not washed out, which is what
                the masking design was checked against — and the training script prints{" "}
                <span className="mono">NET LOSES to the empirical marginal</span> if that ever stops being
                true.</p>
              </div>
            </div>
          </div>
        </section>

        {/* ============================ 07 WRITTEN TWICE ============================ */}
        <section id="twice" className="wrap">
          <SecHead n="07" title="The same model, written twice"
            lede="PyTorch trains it. Nothing resembling PyTorch is allowed near the server, so the forward pass exists a second time, by hand, in NumPy." />

          <div className="panel constraint">
            <div className="kicker">Why</div>
            <p>The API runs on a 512&nbsp;MB instance. The serve requirements file installs exactly seven
            packages — <span className="mono">fastapi, uvicorn, pydantic, pydantic-settings, httpx, tenacity,
            numpy</span> — and torch, sklearn, pandas and matplotlib are deliberately absent. A torch import in
            any serve-path module does not slow the deploy down; it breaks the build.</p>
          </div>

          <div className="twocol">
            <div className="col">
              <div className="col-head"><span className="tag">TRAINING</span> <span className="mono dim">models/winprob.py</span></div>
              <pre className="code"><span className="c">{"// nn.Module, autograd, GPU-optional"}</span>{`
team_vec = brawler(team).mean(dim=1)
h  = cat([team_vec, ctx], -1)
S  = strength(h).squeeze(-1)

pa = counter_p(team_a).sum(dim=1)
qb = counter_q(team_b).sum(dim=1)
counter = (pa*qb).sum(-1) - (pb*qa).sum(-1)

logit = S_a - S_b + counter`}</pre>
            </div>
            <div className="col">
              <div className="col-head"><span className="tag on">SERVING</span> <span className="mono dim">models/serve.py</span></div>
              <pre className="code"><span className="c">{"// pure NumPy, no autograd, no dropout"}</span>{`
team_vec = W["brawler.weight"][team].mean(1)
h  = concat([team_vec, ctx], 1)
h  = maximum(h @ W0.T + b0, 0.0)
S  = (h @ W3.T + b3)[:, 0]

pa = W["counter_p.weight"][a].sum(1)
qb = W["counter_q.weight"][b].sum(1)
logit = S_a - S_b + (pa*qb).sum(1) - (pb*qa).sum(1)`}</pre>
            </div>
          </div>

          <div className="grid-2 mt">
            <div className="panel">
              <div className="kicker">Where the two can drift</div>
              <ul className="clean tight">
                <li><b className="tB">Loud.</b> Serving addresses the MLP by position —{" "}
                <span className="mono">strength.0.*</span> and <span className="mono">strength.3.*</span>. Insert
                a layer in the Sequential and every key renumbers; inference raises{" "}
                <span className="mono">KeyError</span> on the first request.</li>
                <li><b className="gold">Silent.</b> The activation is hardcoded as{" "}
                <span className="mono">maximum(h, 0)</span>. Swap ReLU for GELU in training and the export
                succeeds, the shapes match, the keys match — and the server computes a different function
                forever.</li>
                <li><b className="gold">Silent.</b> Mean-pool for strength, sum-pool for counters, duplicated by
                hand on both sides. Identical shapes either way.</li>
                <li><b className="gold">Silent.</b> The list of which tables are embedding-indexed is maintained
                by hand. A new embedding table would export cleanly, load cleanly, and then index out of bounds
                on the first unknown id.</li>
              </ul>
            </div>
            <div className="panel panel-2">
              <div className="kicker">What guards it</div>
              <p>A parity test walks all 16 draft states and asserts the two implementations agree to{" "}
              <span className="mono">5e-5</span> — tolerance rather than equality, because NumPy&apos;s{" "}
              <span className="mono">h @ W.T + b</span> and torch&apos;s fused <span className="mono">addmm</span>{" "}
              round differently in float32.</p>
              <p><b>It skips silently wherever torch is absent</b> — which includes the deploy environment and any
              checkout that only installed the serve requirements. The test is real, and it only runs where the
              second implementation isn&apos;t.</p>
              <div className="rule" />
              <p className="dim">The export also pins the trained vocabulary — brawler ids, map ids and mode
              names, in row order — into the artifact, and refuses to write if the live catalog has diverged.
              Without that check, a brawler added after training would land exactly on the mask row: in range,
              silently scored as &ldquo;not picked yet.&rdquo;</p>
            </div>
          </div>
        </section>

        {/* ============================ 08 ONE OF SEVEN ============================ */}
        <section id="blend" className="wrap">
          <SecHead n="08" title="One signal out of seven"
            lede="The net never speaks alone. It is blended with count-based statistics, and its share of the answer changes with every pick made." />

          <div className="stack">
            <Equation
              title="Shrinkage, then a renormalized weighted average"
              tex={[
                `\\widehat{w} = \\frac{\\text{wins} + \\kappa\\,\\pi}{\\text{games} + \\kappa}, \\qquad \\mathrm{conf} = \\frac{\\text{games}}{\\text{games} + \\kappa}, \\qquad \\kappa = 20`,
                `\\mathrm{score}(b) = \\frac{\\sum_{k \\in \\mathcal{A}} \\omega_k\\, v_k(b)}{\\sum_{k \\in \\mathcal{A}} \\omega_k}`,
              ]}
            >
              Every raw win rate is Bayesian-shrunk toward a prior <b>π</b> with a pseudo-count of 20 games, and
              &ldquo;wins&rdquo; and &ldquo;games&rdquo; are themselves recency-weighted counts{" "}
              <Tex>{"\\sum_i w_i"}</Tex> on a 21-day half-life. <b>𝒜</b> is the set of signals <em>active</em> at
              this draft state — an inactive one is dropped from the average entirely rather than defaulted to a
              neutral 0.5 that would drag every candidate toward the middle.
            </Equation>

            <Fig caption={<>
              <b>A weight means nothing without its divisor.</b> Inactive signals are dropped from the average
              entirely, not defaulted, so the remaining weights rescale. Saturated bars are the four signals
              fitted against 995,135 held-out matches; the grey ones marked <span className="mono">*</span> are
              hand-set heuristics that have never been ablated — and even on a fully personalized board they take
              at most {pct(heuristicShare)} of the answer, against {pct(modelPersonal)} for the net. At most, because
              the personal weight is itself scaled by how much you have played the brawler: the bottom row is drawn
              at the limit where that confidence reaches 1, and every real board sits below it.
            </>}><SignalShares /></Fig>

            <div className="grid-2">
              <div className="tablewrap">
                <table>
                  <caption className="tcap">The seven signals</caption>
                  <thead><tr><th>Signal</th><th>Weight</th><th>Status</th><th>Active when</th></tr></thead>
                  <tbody>
                    <tr className="hi"><td>model</td><td>0.40</td><td>fitted</td><td>artifact loaded</td></tr>
                    <tr><td>map rate</td><td>0.25</td><td>fitted</td><td>always</td></tr>
                    <tr><td>counter</td><td>0.20</td><td>fitted</td><td>enemies revealed</td></tr>
                    <tr><td>synergy</td><td>0.05</td><td>fitted</td><td>allies picked</td></tr>
                    <tr><td>role fit</td><td>0.10</td><td>heuristic</td><td>always</td></tr>
                    <tr><td>mastery</td><td>0.10</td><td>heuristic</td><td>roster supplied</td></tr>
                    <tr><td>personal</td><td>0.08 × conf</td><td>heuristic</td><td>you&apos;ve played it</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="panel panel-2">
                <div className="kicker">Why 0.40</div>
                <p>A sweep over candidate weightings on 995k matches put the model at 0.40 — it beat the previous
                shipped weights in 200 of 200 paired bootstrap resamples and landed within 0.0005 AUC of the
                linear ceiling. The curve plateaus between 0.40 and 0.50 and falls away toward model-only.</p>
                <p>Synergy survives at 0.05 rather than 0 on a judgement call: conditional on map and counter it
                adds nothing measurable, because the net already encodes it — but mid-draft it is the only signal
                that answers &ldquo;does this fit what we already have.&rdquo;</p>
                <p className="dim">Mastery and personal were cut by more than half in Aug 2026. At their old
                values the two &ldquo;how good is <em>this player</em>{" "}on it&rdquo; signals were about 31% of a
                personalized pick&apos;s score, out-driving the net and burying meta picks the player could grow
                into.</p>
              </div>
            </div>

            <Equation
              title="A heuristic that yields to data"
              tex={`\\text{role}_{\\text{eff}} = 0.5 + \\big(1 - \\mathrm{conf}\\big)\\big(\\text{role fit} - 0.5\\big)`}
            >
              Role fit is a hand-set mode-and-class <em>prior</em>, and its 0.5–0.9 spread punched far above its
              0.10 weight next to the compressed ~0.47–0.59 band real map win rates live in. So it is shrunk
              toward neutral by that brawler&apos;s own per-map confidence: on a well-sampled map every candidate
              gets the same 0.5 and the term drops out of the ranking entirely; on a freshly rotated map with no
              data, the archetype prior speaks at full volume.
            </Equation>

            <Fig caption={<>
              <b>The model ate the count tables.</b> A stacking regression asked how much to trust the net versus
              the empirical win-rate tables. In June, on 40k matches, it answered 31/69. In August, on 995k, the
              same question answered 78/22. Nothing about the architecture changed — only how much it had seen.
            </>}><StackingSplit /></Fig>
          </div>
        </section>

        {/* ============================ 09 LIMITS ============================ */}
        <section id="limits" className="wrap">
          <SecHead n="09" title="What it can't do"
            lede="A 0.630 AUC is a real edge and a modest one. Most of what decides a ranked match is not in the draft at all." />

          <div className="grid-2">
            <ul className="clean">
              <li><b>Skill dominates.</b>{" "}Both teams usually draft competently, so the draft explains only a slice
              of the outcome. Matchmaking equalizes the rest — the base team-A win rate is 0.511. No weighting
              scheme pulls a pooled AUC far past 0.63. The honest claim is a small, real edge, not
              &ldquo;predicts winners.&rdquo;</li>
              <li><b>It is seat-blind.</b> The battle log records final teams and never pick order, so the model
              cannot condition on who picks next. The same board scores identically whether the next pick is
              yours or theirs.</li>
              <li><b>It has never seen a ban.</b> The API does not expose the ban phase at all. Ban value is
              inferred separately from win rate and contest rate, and unlike the pick weights, none of it is fit
              to held-out matches — there is no ground truth to fit to.</li>
              <li><b>Partial boards are averages, not worst cases.</b> An unfinished board marginalizes over how
              real opponents continued such drafts. It does not simulate an opponent finding the sharpest answer,
              so a pick with a rare but devastating counter is overvalued against a strong one.</li>
              <li><b>One population, and it is a mixture.</b> The crawler seeds from leaderboards but expands
              through battle-log tags, which diffuses down-ladder: in a 200,000-match sample about <b>61% of
              matches are Diamond-or-below</b>, and Pro never appeared. A single un-conditioned net is therefore
              fit to a blend of brackets and fits neither tail well. Rank-bracket stat tables mitigate that on the
              empirical side; the net itself is not bracket-conditioned.</li>
              <li><b>It cannot see how built your brawler is.</b> Power level is collected but dropped before
              training, and the equipped star power, gadget, gears and hypercharge are never in a battle log at
              all. In practice that is closer to a definition than a defect — 97.2% of observed player-slots are
              Power 11 — so read the number as near-max-power play.</li>
              <li><b>It goes stale.</b> Brawler strength moves with every balance patch. Recency weighting slows
              the decay; only retraining stops it, which is why a detected meta shift retrains and republishes
              without anyone asking.</li>
            </ul>

            <div className="stack">
              <div className="panel">
                <div className="kicker">The number that matters most</div>
                <p className="big-num green">0.011</p>
                <p>For a tool that <em>consumes</em> probabilities — blending them, ranking with them, projecting
                ban swings from them — calibration matters more than accuracy. A well-calibrated 0.58-accuracy
                model is useful. An overconfident 0.62 one is worse than nothing, because every downstream number
                inherits the lie.</p>
              </div>
              <div className="panel panel-2">
                <div className="kicker">So, is it a neural network?</div>
                <p>Yes: learned embeddings, a nonlinear hidden layer, trained end to end by backpropagation. Also
                12,505 parameters, no attention, no GPU, and a forward pass you can follow with a finger.</p>
                <p>Both of those are true at once, and the second one is the point. The architecture is small
                because the inductive bias — antisymmetry, shared weights, a trained token for
                &ldquo;unknown&rdquo; — does the work that capacity would otherwise have to.</p>
              </div>
            </div>
          </div>
        </section>

        <footer className="wrap">
          <div className="foot">
            <div>
              <div className="eyebrow">Sources</div>
              <p className="dim small">Every figure is read from the shipped artifact{" "}
                <span className="mono">winprob.npz</span> and the code that writes it — the training model, the
                NumPy server, the training script and the scoring engine. Evaluation numbers come from the
                model card and the held-out ablation suite, cross-checked against the metrics the training run
                emits. The prose version of all of this, without the math, is{" "}
                <a href="/how-it-works">How Brawl Draft works</a>.</p>
            </div>
            <div>
              <div className="eyebrow">Standing</div>
              <p className="dim small">Model retrained 24 Aug 2026 on 1,615,154 matches. Weights last swept
                10 Aug 2026 on 995,135. Not affiliated with, endorsed, sponsored or specifically approved by
                Supercell; built on the public Brawl Stars API (
                <a href="https://supercell.com/en/fan-content-policy/" target="_blank" rel="noopener noreferrer">Fan
                Content Policy</a>).</p>
            </div>
          </div>
        </footer>
      </div>
    </div>
    </>
  );
}
