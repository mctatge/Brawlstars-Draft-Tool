"""Runtime configuration loaded from .env (API token, player tag, crawler tuning).

Importing this requires `pydantic-settings` (see backend/requirements.txt). The
pure-data reference layer deliberately does NOT import this, so it can run without
installing dependencies.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from bsdraft.constants import REPO_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets
    brawlstars_api_token: str = ""
    player_tag: str = ""

    # Crawler tuning
    crawl_rate_limit_per_sec: float = 5.0
    crawl_seed_countries: str = "global,US,DE,KR,BR,JP"
    crawl_ranked_only: bool = True
    # Re-scan a known player once their last fetch is older than this many hours, to pick up
    # ranked games played since (the API only exposes a player's last ~25 battles). 0 disables
    # re-scanning — each player is fetched at most once.
    crawl_revisit_hours: float = 12.0

    # Cloud serving / live refresh
    # URL of the published matches dataset (GitHub Release asset, gzipped). When set, the API
    # syncs it at startup and every `refresh_seconds`, rebuilding draft stats with no restart.
    data_url: str = ""
    # URL of the published win-prob model (winprob.npz Release asset). When set, the API syncs
    # it alongside the dataset and hot-swaps the reloaded model in without a restart — so a
    # retrain (e.g. after a balance shift) rolls out live instead of waiting for a redeploy.
    model_url: str = ""
    # URL of the published precomputed stats (stats.json.gz Release asset). When set, the API
    # LOADS the empirical stats from it instead of rebuilding them in memory from the full
    # match dataset — so it uses *all* matches with no 512 MB OOM (the home machine builds +
    # publishes them; see scripts/export_stats.py). Unset = rebuild locally (capped, below).
    stats_url: str = ""
    # URL of the published player-rank index (rank_index.npz Release asset; the legacy
    # rank_index.json.gz also loads — the loader dispatches on content). When set, the API only
    # ever LOADS the tag->Ranked-tier lookup (~66 MB peak at 3M tags for the npz), degrading to
    # an empty index on failure — it never falls back to building the ~3M-entry dict in memory
    # (~200 MB, the 512 MB OOM). The home machine builds + publishes it (see
    # scripts/export_rank_index.py). Unset = build locally from the matches (home/dev only).
    rank_index_url: str = ""
    # URL of the published meta-drift report (meta_report.json Release asset, a few KB). When
    # set, /api/meta SERVES it instead of recomputing drift over the full match dataset (two
    # streaming passes — minutes per data change on a small cloud CPU, which times out the
    # frontend's meta banner). The crawler writes + publishes it after each cycle's meta check
    # (see scripts/collect.py). Unset = compute locally from the matches.
    meta_report_url: str = ""
    # URL of the published per-item win-rate table (itemstats.json.gz Release asset). When set,
    # /api/loadout serves DATA-DRIVEN gadget/star-power picks (single-item-owner inference over the
    # matches x ownership-profiles join) instead of the effect heuristic, falling back to the
    # heuristic per item where the sample is thin. The home machine builds + publishes it (see
    # scripts/export_itemstats.py, which needs the collected profiles). Unset = heuristic only.
    itemstats_url: str = ""
    refresh_seconds: int = 600  # re-sync interval in seconds (0 disables the refresh loop)
    # Comma-separated allowed CORS origins. "*" allows any (fine for the read-only meta API).
    # Lock this to your site's origin on the roster host you expose via the tunnel, e.g.
    # CORS_ORIGINS=https://brawlstars-draft-tool.pages.dev
    cors_origins: str = "*"

    # Engine tuning
    stats_halflife_days: float = 21.0  # recency half-life (days) for empirical stats; <=0 disables decay
    # Cap the empirical-stats build to the most recent N matches (0 = all). Bounds peak RAM so
    # the build fits a small instance (e.g. Render's 512 MB) as the crawler grows the dataset;
    # recency weighting already makes older matches near-weightless, so this barely moves the
    # numbers. Lower it if the host still OOMs; raise/zero it on a bigger box.
    stats_max_matches: int = 60000
    # The frontend re-polls the roster so a long session picks up newly unlocked/upgraded
    # brawlers. Serve a cached roster for this many seconds so that polling (and multiple
    # tabs) don't each hit the live Supercell API. Keep it well below the poll interval so a
    # poll still refreshes. 0 disables caching (every request fetches live).
    roster_ttl_seconds: int = 90

    @property
    def cors_origin_list(self):
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()] or ["*"]

    @property
    def seed_countries(self):
        return [c.strip() for c in self.crawl_seed_countries.split(",") if c.strip()]

    @property
    def normalized_player_tag(self):
        """Player tag without a leading '#', uppercased (API path form)."""
        return self.player_tag.lstrip("#").strip().upper()


settings = Settings()
