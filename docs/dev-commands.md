# Dev Commands — setup, run, train, test, frontend

Every command needed to develop this repo: Python venv setup, running the FastAPI backend,
the data-collection → training → export pipeline, tests, and the Next.js frontend.

## Backend setup (Python 3.11+)

Everything in the backend needs `PYTHONPATH=backend` — the `bsdraft` package lives under
`backend/`, and scripts are run from the repo root.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt          # full stack: torch, sklearn, pandas, fastapi
cp .env.example .env                              # add BRAWLSTARS_API_TOKEN + PLAYER_TAG
```

There are three requirements files — see [backend-architecture.md](backend-architecture.md)
for why. `requirements.txt` is the full dev stack; `requirements-serve.txt` is what the
deployed API installs (no torch/sklearn/pandas).

## Run the API

Serves the model from `data/processed/winprob.npz`; reads `data/raw/matches.jsonl` locally.

```bash
PYTHONPATH=backend uvicorn bsdraft.api.main:app --reload --port 8000
```

## Data → model pipeline

One-time / retrain path. `collect.py` needs the IP-locked Supercell key (home machine only —
see [deployment-topology.md](deployment-topology.md)).

```bash
PYTHONPATH=backend python backend/scripts/collect.py --target 30000   # snowball crawl → data/raw/
PYTHONPATH=backend python backend/scripts/train.py                    # torch train → winprob.pt + docs/ charts
PYTHONPATH=backend python backend/scripts/export_model.py             # winprob.pt → winprob.npz (commit this)
PYTHONPATH=backend python backend/scripts/export_stats.py             # precomputed stats artifact (published next to matches.jsonl.gz)
PYTHONPATH=backend python backend/scripts/export_rank_index.py        # precomputed tag→tier rank index (rank_index.npz; the cloud LOADS it — ~66 MB peak vs ~200 MB+ building in RAM)
```

Other scripts under `backend/scripts/`:

- `smoke_test.py` — verify the API key works + inspect real response shapes.
- `ablate_components.py` / `ablate_context.py` — held-out ablations → `docs/ablation*.json`
  (methodology + results in [model-evaluation.md](model-evaluation.md)).
- `refresh_reference.py` — re-pull the Brawlify reference JSONs. **Careful:** refreshing
  `maps.json` without a retrain silently re-maps trained map embedding rows; brawlers are
  safe (id-sorted, append-only).

## Tests

Lightweight by design; each test file also runs standalone via `__main__`.

```bash
PYTHONPATH=backend python -m pytest backend/tests/
PYTHONPATH=backend python backend/tests/test_personal.py
```

`pytest` / `ruff` / `mypy` are optional dev tools (not pinned in requirements).

## Frontend (Next.js)

```bash
npm --prefix frontend install
npm --prefix frontend run dev      # http://localhost:3000
npm --prefix frontend run build    # static export → frontend/out/ (output: "export")
```

**Frontend dev points at the deployed API by default.** `frontend/.env.local` sets
`NEXT_PUBLIC_API_BASE` to the live Render URL, so `npm run dev` hits production, not your
local uvicorn. To test local backend changes in-browser, override
`NEXT_PUBLIC_API_BASE=http://localhost:8000`. The var is inlined at build time for the
static export.

### Roster/personalization in local dev — needs the local-only API (home machine)

Anything gated on a loaded roster — the "I'm pick" seat checkboxes (Mythic+), the blind-pick
personal pick column (Diamond and below), owned-item filtering in the
loadout popover, mastery weighting, the `/purchases` advisor — is **dead on a plain
`npm run dev`** (`⚠ roster service is down — personalization is off`). On the home machine it
works with the two `.claude/launch.json` configs (verified 2026-08-19): start `backend-local`
(uvicorn on 127.0.0.1:8099 with `CORS_ORIGINS=*`; it reads the key from `.env`, builds stats in
~3 min) and `frontend-local-api` (`NEXT_PUBLIC_API_BASE=http://127.0.0.1:8099`; `ROSTER_BASE`
falls back to it, so roster, rank and purchases all resolve locally). Elsewhere, verify on the
deployed site.

Why: `getRoster`/`getRank` go to `NEXT_PUBLIC_ROSTER_BASE`, which falls back to
`NEXT_PUBLIC_API_BASE` (Render) when unset — and Render has no Supercell token, so it can only
ever answer `loaded:false`. Setting `NEXT_PUBLIC_ROSTER_BASE=https://roster.brawldraft.com`
does **not** fix it: `CORS_ORIGINS` in `com.bsdraft.api.plist` is deliberately locked to the
three prod origins, so the browser gets no `access-control-allow-origin` and the fetch throws.
Verify with:

```bash
curl -s -D - -o /dev/null -H "Origin: http://localhost:3000" "https://roster.brawldraft.com/api/roster?tag=YOURTAG" | grep -i access-control
```

If you later want this working locally, in preference order:

1. **Second local-only API** (keeps the public surface untouched — this is what the
   `backend-local` + `frontend-local-api` launch configs do, and it works). By hand:
   ```bash
   CORS_ORIGINS='*' PYTHONPATH=backend .venv/bin/uvicorn bsdraft.api.main:app --host 127.0.0.1 --port 8099
   ```
   then `NEXT_PUBLIC_API_BASE=http://127.0.0.1:8099` for the frontend (`ROSTER_BASE` falls back
   to it). Same machine, same key, so rosters load fully. Don't expose 8099 through the tunnel.
2. **Add `http://localhost:3000` to `CORS_ORIGINS`** in `com.bsdraft.api.plist`, then
   `launchctl bootout` + `bootstrap` (a plist edit — `kickstart -k` won't re-read it; see
   [../deploy/roster-tunnel.md](../deploy/roster-tunnel.md)). One line, no extra process, but it
   widens a *public* endpoint: any page served from `localhost:3000` on anyone's machine could
   then read rosters from the browser.
3. **Next.js dev rewrite** proxying `/roster-api/*` → the tunnel, making it same-origin. Avoids
   both, but puts a dev-only path in committed `next.config`.

Ads are shipped dark (env-gated) — read [adsense-go-live.md](adsense-go-live.md) before
touching `AdSlot.tsx`, `ads.txt`, or enabling AdSense.
