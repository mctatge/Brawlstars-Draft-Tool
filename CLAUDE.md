# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AI ranked-draft assistant for Brawl Stars: a Python win-probability model + draft engine (`backend/`, package `bsdraft`) behind a FastAPI API, with a Next.js draft board (`frontend/`). Deployed as a live site; the Supercell data key is IP-locked to a home machine, which shapes most of the deployment design.

## Universal rules

- Everything backend needs `PYTHONPATH=backend` (the `bsdraft` package lives under `backend/`; scripts run from the repo root).
- The deployed API installs `requirements-serve.txt` only — **no torch/sklearn/pandas** in any serve-path module, or the deploy build breaks (512 MB Render free tier).
- The model has two implementations that must stay in sync: torch training (`models/winprob.py`) and pure-NumPy serving (`models/serve.py`). Change one → change both.
- `backend/bsdraft/constants.py` and `data/reference.py` stay pure stdlib — no third-party imports.
- Frontend `npm run dev` hits the **deployed** API by default (`frontend/.env.local`); override `NEXT_PUBLIC_API_BASE=http://localhost:8000` to test local backend changes in-browser. Roster/personalization only works locally on the home machine via the `backend-local` + `frontend-local-api` launch configs (local API with the key, open CORS); elsewhere verify on the deployed site — see [docs/dev-commands.md](docs/dev-commands.md).

## Where things live

| Doc | What it covers | When to read it |
| --- | --- | --- |
| [docs/dev-commands.md](docs/dev-commands.md) | Every command: setup, run the API, collect→train→export pipeline, tests, frontend dev/build | Before running, training, or testing anything |
| [docs/backend-architecture.md](docs/backend-architecture.md) | Backend layers, data flow, the four cross-cutting design constraints, the two recommend endpoints | Before changing anything under `backend/bsdraft/` |
| [docs/deployment-topology.md](docs/deployment-topology.md) | Home crawler → GitHub Release artifacts → Render hot-swap; keepwarm/drift Actions; roster tunnel | Before touching `deploy/`, `data/sync.py`, `render.yaml`, `.github/workflows/`, artifact export scripts, or any `*_URL` env var — and when debugging the live site |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | The win-probability model: math, training data, calibration, limitations | Before changing model architecture, features, or training |
| [docs/model-evaluation.md](docs/model-evaluation.md) | Held-out ablations behind `DEFAULT_WEIGHTS`; why signal weights are global, not per-map | Before changing `engine/scoring.py` weights or adding/removing a scoring signal |
| [docs/item-winrate.md](docs/item-winrate.md) | Data-driven `/api/loadout` picks: single-item-owner inference (estimator, gate, biases) and the profiles→build→publish→serve pipeline | Before changing `engine/loadout.py`, `data/itemstats_build.py`, `engine/itemstats.py`, `collect/profiles.py`, or the `itemstats.json` artifact |
| [docs/readiness.md](docs/readiness.md) | What an under-leveled brawler costs: the within-player power-deficit estimator, its placebo gate, and the conservative shipping rule behind `readiness.json` | Before changing `data/readiness_build.py`, `scripts/export_readiness.py`, `data/reference/readiness.json`, or anything that prices power level into a score |
| [docs/purchase-advisor.md](docs/purchase-advisor.md) | The `/purchases` "what to upgrade next" page: data reality, scoring, the curated `economy.json` table, and the `/api/purchases` endpoint | Before changing `engine/purchases.py`, `data/reference/economy.json`, the `/api/purchases` endpoint, or `frontend/components/PurchaseAdvisor.tsx` |
| [docs/adsense-go-live.md](docs/adsense-go-live.md) | Enabling the env-gated ad slots without halting ad serving | Before touching `AdSlot.tsx`, `ads.txt`, or enabling AdSense |
| [deploy/roster-tunnel.md](deploy/roster-tunnel.md) | Cloudflare Tunnel setup for per-visitor roster (`roster.brawldraft.com`) | When touching roster personalization or tunnel/launchd config |
| [PLAN.md](PLAN.md) | Goals, competitive landscape, phased roadmap | For scope/why questions or when planning new features |
| [README.md](README.md) | Public-facing overview, feature summary, quickstart | For the product pitch or user-visible behavior |
| [CHANGELOG.md](CHANGELOG.md) | Dated log of notable user-visible changes | After shipping one — add an entry; or when summarizing what shipped |

`Notes/` holds personal planning scratch notes — not maintained documentation.
