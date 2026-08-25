# Deployment Topology — home crawler, GitHub Release artifacts, Render hot-swap, Cloudflare Tunnel

How the system is deployed and why: the Supercell API key is IP-locked to the home machine,
so collection runs at home and the public cloud API consumes published artifacts. Read this
before touching `deploy/`, `data/sync.py`, `render.yaml`, artifact export scripts, the
keepwarm workflow, or any `*_URL` / `REFRESH_SECONDS` env var — and when debugging the
live site.

## Home machine (has the key)

The crawler runs on a home machine via four launchd plists under `deploy/`: crawler
(`com.bsdraft.crawler`), API (`com.bsdraft.api`), tunnel (`com.bsdraft.tunnel`), and the
IP-lockout watchdog (`com.bsdraft.watchdog`, below). It publishes `matches.jsonl.gz` +
`winprob.npz` + precomputed stats, rank-index, and meta-drift-report artifacts to a
GitHub Release.

The crawler agent runs with `--retrain-on-shift`, so a detected meta shift auto-retrains and
republishes the model. The one manual path left is a **new brawler**: run
`backend/scripts/refresh_reference.py` + retrain + a commit (the reference JSONs are bundled
into the repo).

### IP-rotation watchdog (the recurring outage)

Comcast rotates the home IP every 1–2 weeks (2026-07-11, 08-10, 08-13, 08-24); each
rotation makes the IP-locked key 403 (`accessDenied.invalidIp`), which kills the crawler
*silently* and takes down the roster tunnel — and the cloud-side keepwarm Action has never
caught one in time. `com.bsdraft.watchdog` runs `backend/scripts/watchdog.py` every
10 minutes (stdlib-only, on system `python3` — deliberately independent of the venv). It
probes one cheap API endpoint with the `.env` token; on lockout it fires a macOS
notification (once per outage, with a reminder every `WATCHDOG_REMIND_HOURS`, default 6)
and writes `data/raw/watchdog_status.json` containing the new public IP and the exact fix:

1. Mint a key allowing that IP at developer.brawlstars.com, paste it into `.env`.
2. `launchctl kickstart -k gui/$UID/com.bsdraft.api` and `.../com.bsdraft.crawler` —
   both hold the old token in memory until restarted (the tunnel agent has no key).

The watchdog re-reads `.env` each cycle, so it confirms the fix with a one-shot
"recovered" notification. On first install, run `python3 backend/scripts/watchdog.py
--test-notify` once and approve the notification permission so real alerts get through.

## Cloud API (Render, no key)

The Render API (`render.yaml`) pulls via `DATA_URL` / `MODEL_URL` / `STATS_URL` /
`RANK_INDEX_URL` / `META_REPORT_URL` every `REFRESH_SECONDS` and **hot-swaps rebuilt stats
and a reloaded model with no restart** (see `data/sync.py` and the `_refresh_loop` /
`lifespan` in `api/main.py`).

The precomputed artifacts exist to fit Render's 512 MB free tier:

- **Stats artifact** — lets the cloud API skip a full dataset replay at boot;
  `STATS_MAX_MATCHES` only bounds the fallback rebuild if the artifact can't load.
- **Rank-index artifact** (`rank_index.npz` — the serve arrays themselves) — loaded as a
  compact NumPy `tag→tier` lookup for `/api/rank`: ~66 MB peak / ~0.3 s at 3.0M tags, vs the
  ~263 MB decode peak of the legacy `rank_index.json.gz` (which OOM-killed the box on
  2026-08-23; the crawler still dual-publishes it as the rollback, and the loader reads either
  by magic bytes). If the artifact can't load the API **serves an empty index** (ranks read as
  unknown until the next sync) — it never falls back to the ~200 MB in-memory build, which is
  the OOM the artifact exists to prevent.
- **Meta-report artifact** (`meta_report.json`, a few KB, written by the crawler's per-cycle
  drift check) — what `/api/meta` serves. Recomputing drift streams the full dataset twice,
  which takes minutes on the free tier's CPU sliver and times out the frontend's meta banner.

## Keepwarm + drift alerts (GitHub Actions)

A scheduled Action (`.github/workflows/keepwarm.yml`) pings `/api/health` to keep Render's
free tier out of cold-sleep, and once a day checks `/api/meta` (the drift detector in
`engine/drift.py`) — filing a `meta-alert` GitHub issue when the meta shifts or a new
brawler appears.

## Per-visitor roster via Cloudflare Tunnel

Consequence of the IP lock: the public backend can't fetch a roster itself, so
personalization is wired around it — the frontend pulls the player's roster from the home
machine over a **Cloudflare Tunnel** (`roster.brawldraft.com` → the `com.bsdraft.api` agent;
setup in [../deploy/roster-tunnel.md](../deploy/roster-tunnel.md) and
`deploy/cloudflared.yml`) and **forwards it in the `/api/recommend` body**
(`RecommendRequest.roster`), which drives the owned-filter + mastery/loadout scoring there.
`/api/rank` likewise resolves from the collected data when no key is present.

## Deploy triggers

Push to `main` auto-deploys both halves: Render rebuilds the API, Cloudflare Pages rebuilds
the static frontend export.
