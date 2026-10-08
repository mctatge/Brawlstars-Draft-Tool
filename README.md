# Brawl Draft — experimental ranked draft assistant

Brawl Draft combines a win-probability model with empirical map statistics, composition
warnings and optional account personalization for **Brawl Stars Ranked**. It is also an
end-to-end ML engineering project: collection → evaluation → publication → serving.

**Status: experimental public beta.** The live API runs on a small Render instance; roster
lookup depends on a home service. Availability and capacity are limited. Broad concurrent
usage and an improvement in player win rate have not been demonstrated.

Start with the **[recorded example](https://brawldraft.com/demo)** — no account or live API
required after the page loads. Then try the [live beta](https://brawldraft.com).

## What it does

- Recommends bans and picks for the selected map and current board.
- Scores partial teams using a learned unknown-slot embedding.
- Blends model, map, counter, synergy and role signals; shows the components.
- Checks ownership and power requirements for the user's seat, with bounded readiness and
  personal-history adjustments.
- Hides superseded recommendations during map, pick, slot and account changes.

The draft score ranks candidates. It is not a calibrated probability of winning. Predicting
historical outcomes does not prove a causal benefit from following recommendations.

## First demonstration

Node.js 22 is enough for the saved walkthrough; no credentials, dataset or Python setup:

```bash
npm --prefix frontend ci
npm --prefix frontend run dev
# Open http://localhost:3000/demo
```

The three-step example contains dated, recorded outputs from an identified model and a
bounded current-era aggregate sample. It makes no recommendation or roster requests.
Its artifact hashes and sample size are shown under “Example source.”

## How it works

A player-centric crawler collects ranked battle logs through the official Brawl Stars API.
Matches are deduplicated because a match can appear in six players' logs. Live analysis
uses a reviewed balance-era cutoff, with recency decay within that era; raw history remains
available for research.

A small antisymmetric embedding network estimates win probability from teams, map and mode:

$$
z(A,B\mid c)=S(A,c)-S(B,c)+P_A^\top Q_B-P_B^\top Q_A+T(A)-T(B),
\qquad p(A,B)=\sigma(z).
$$

Swapping teams negates the logit, so their probabilities sum to one by construction.
Masked training supports unfinished drafts. The match log contains completed teams,
not actual pick order or bans, so the model cannot reconstruct those decisions.

PyTorch trains the network; pure NumPy serves it through FastAPI. A Next.js static frontend
calls the API. The cloud serve environment excludes torch, sklearn and pandas.

See the [model card](docs/MODEL_CARD.md) for architecture and limitations, and
[historical ablations](docs/model-evaluation.md) for signal-weight experiments.

## Evaluation and release safeguards

The [model page](https://brawldraft.com/model) reads `/api/model` from the configured API.
It displays current metrics only when a report is verified against the exact served
weights. Legacy weights without a bound report show evaluation as unavailable.
`docs/metrics.json` and the model-card numerical tables are historical experiments.

The retraining workflow downloads the **actual released NumPy incumbent**, reserves a fresh
evaluation snapshot, and uses separate chronological training, selection and final test
windows. Publication fails on missing/invalid incumbent, insufficient fresh data, excess
full-team log-loss regression, or failed export/capability checks. Calibration and empirical
single-pick comparisons are diagnostics, not enforced performance thresholds.

Weights and metrics are uploaded to an immutable release and verified before a small pointer
moves. Serving validates the bundle and retains its last usable copy on failure.
Pull-request CI runs backend regressions, Torch/NumPy parity, frontend async tests, typecheck
and static build. Full retraining runs separately.

Read [reliability and publication](docs/reliability-and-publication.md) for the exact
checks, limits and first migration procedure.

## Development

For a live local backend and your own training data (Python 3.11+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cp .env.example .env
# Add your IP-locked Brawl Stars API key before collecting.
PYTHONPATH=backend python backend/scripts/collect.py --target 30000
PYTHONPATH=backend python backend/scripts/train.py
PYTHONPATH=backend python backend/scripts/export_model.py
PYTHONPATH=backend uvicorn bsdraft.api.main:app --port 8000
```

In another terminal:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000 NEXT_PUBLIC_ROSTER_BASE=http://localhost:8000 npm --prefix frontend run dev
```

Plain training/export commands are local research tools. Publishing a serving bundle requires
the additional incumbent, reservation and evaluation checks in the retraining workflow.
See [dev commands](docs/dev-commands.md) for collection, exports and tests.

## Deployment and operational limits

The IP-locked key stays on the home machine. The crawler publishes dataset/statistics artifacts
to GitHub Releases. Render pulls these periodically; Cloudflare Pages serves the frontend.
An optional tunnel exposes the home roster API. Its availability can fail independently of
the draft API.

The single API process bounds personal-history scans to two, shares and limits live
profile requests, and omits cold history while warming. These controls reduce avoidable work;
they do not establish large-userbase capacity. Keepwarm does not provide an uptime guarantee.
The header's archive count is not the current-era model training count.

`/api/health` reports readiness, statistics source, model identity, data age and per-artifact
sync attempts/errors. No usable empirical data returns HTTP 503. The recorded example remains
usable when live services are unavailable.

See [deployment topology](docs/deployment-topology.md), [render.yaml](render.yaml) and
[roster tunnel setup](deploy/roster-tunnel.md). Merge/deployment and the first model migration
are separate rollout steps.

## Credits and license

MIT. Unofficial Supercell fan content; not affiliated with or endorsed by Supercell.
Data: [Brawl Stars API](https://developer.brawlstars.com) and
[Brawlify](https://brawlapi.com). Prior inspiration:
[DraftStars](https://github.com/mcmckinley/DraftStars).
