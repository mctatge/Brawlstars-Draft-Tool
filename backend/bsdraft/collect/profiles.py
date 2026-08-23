"""Collect per-player *ownership* profiles — which star powers / gadgets / gears each player
owns on each brawler — the second ingredient (with the match log) for the single-item-owner
win-rate inference that powers data-driven loadout advice.

Battle logs never record the equipped item, so per-item win rates can't be measured directly.
The workaround (the technique Brawl Time Ninja documents) attributes a player's recent matches on
a brawler to their *one* owned item of a type, using the baseline of players who own *none*. That
needs an ownership snapshot per player, which ``/players/{tag}`` provides — IP-locked, so this runs
on the home machine alongside the crawler.

State mirrors the crawler: profiles are appended to ``data/raw/profiles.jsonl`` and a
``profiled_tags.txt`` records each tag's last fetch so a resumed run skips recently-profiled tags
and revisits stale ones (ownership changes slowly, so the revisit window is long). Tags are drawn
from the match log — exactly the population whose matches the inference scores — newest first, so a
capped run profiles the players who matter most.

    PYTHONPATH=backend python -m bsdraft.collect.profiles --limit 20000 --recent-days 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from tqdm import tqdm

from bsdraft.collect.client import BrawlStarsClient, BrawlStarsError, normalize_tag
from bsdraft.config import settings
from bsdraft.constants import RAW_DIR
from bsdraft.engine.mastery import parse_roster

MATCHES_PATH = RAW_DIR / "matches.jsonl"
PROFILES_PATH = RAW_DIR / "profiles.jsonl"
PROFILED_PATH = RAW_DIR / "profiled_tags.txt"

# Ownership changes slowly (you unlock an item once), so re-profiling is far less urgent than
# re-scanning battle logs — a long default window keeps the crawl budget on new players.
DEFAULT_REVISIT_DAYS = 21.0


def owned_summary(player: dict) -> dict:
    """Compact per-brawler ownership for one player: ``{brawler_id: {sp:[ids], gd:[ids],
    gr:[[id,name,level],...], hc:bool, ht:int}}``. Reuses :func:`parse_roster` (the same parser the
    live roster uses) so the owned-item ids stay defined in one place. Brawlers the player owns but
    has no items on are still included — a zero-count is the inference's baseline, not missing data."""
    out: Dict[str, dict] = {}
    for bid, m in parse_roster(player).items():
        out[str(bid)] = {
            "sp": list(m.owned_star_powers),
            "gd": list(m.owned_gadgets),
            # gears carry the name too (no catalog id exists for gears — the id<->name map that lets
            # the build restrict to the six universal gears and the serve path resolve them is learned
            # from these live entries).
            "gr": [[g["id"], g.get("name", ""), g.get("level", 0)] for g in m.owned_gears],
            # Hypercharge ownership and lifetime trophies. Nothing consumes these yet; they are
            # recorded because they are only observable *live*. Battle logs carry no hypercharge
            # field, so a hypercharge contrast can never be reconstructed from the match log
            # retroactively — every un-profiled day is unrecoverable. Hypercharge is also the
            # cleanest ownership contrast available (one binary, and investment-matched: players
            # who own one on brawler X but not Y), which is what the power-deficit estimator in
            # :mod:`bsdraft.data.readiness_build` cannot measure from matches alone.
            "hc": bool(m.has_hypercharge),
            "ht": int(m.highest_trophies or 0),
        }
    return out


def iter_match_tags(recent_days: float = 0.0) -> Iterator[str]:
    """Yield player tags seen in the match log, most-recent matches first, de-duplicated. With
    ``recent_days > 0`` only tags from matches within that window are yielded (their current
    ownership best matches those matches — the inference uses a recent window too)."""
    if not MATCHES_PATH.exists():
        return
    cutoff = (time.time() - recent_days * 86400) if recent_days > 0 else 0.0
    rows: List[dict] = []
    with open(MATCHES_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cutoff and rec.get("ts", 0) < cutoff:
                continue
            rows.append(rec)
    seen = set()
    for rec in sorted(rows, key=lambda r: r.get("ts", 0), reverse=True):
        for tag in rec.get("player_tags", ()):  # already normalized at parse time
            if tag and tag not in seen:
                seen.add(tag)
                yield tag


class ProfileCollector:
    def __init__(self, client: BrawlStarsClient, revisit_days: float = DEFAULT_REVISIT_DAYS):
        self.client = client
        self.revisit_after = revisit_days * 86400
        self.profiled: Dict[str, float] = {}   # tag -> epoch seconds of last profile fetch
        self._now = time.time()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self) -> None:
        if not PROFILED_PATH.exists():
            return
        with open(PROFILED_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                tag = parts[0].strip()
                if not tag:
                    continue
                ts = 0.0
                if len(parts) > 1:
                    try:
                        ts = float(parts[1])
                    except ValueError:
                        ts = 0.0
                self.profiled[tag] = max(self.profiled.get(tag, 0.0), ts)

    def _due(self, tag: str) -> bool:
        last = self.profiled.get(tag)
        if last is None:
            return True
        if self.revisit_after <= 0:
            return False
        return (self._now - last) >= self.revisit_after

    async def run(self, tags: Iterable[str], limit: int = 0) -> int:
        """Fetch + persist ownership profiles for ``tags`` that are new or due for a re-profile,
        stopping after ``limit`` new profiles (0 = no cap). Each profile is flushed immediately so
        a crash keeps the work done so far, mirroring the crawler."""
        self._now = time.time()
        new = 0
        out = open(PROFILES_PATH, "a", encoding="utf-8")
        log = open(PROFILED_PATH, "a", encoding="utf-8")
        pbar = tqdm(total=limit or None, desc="profiles", unit="player")
        try:
            for raw in tags:
                tag = normalize_tag(raw)
                if not tag or not self._due(tag):
                    continue
                try:
                    player = await self.client.get_player(tag)
                except BrawlStarsError:
                    continue   # transient (e.g. a 403 IP-rotation outage) — DON'T mark profiled, so
                               # a good tag isn't skipped for the whole revisit window; retry next run
                row = {"tag": tag, "ts": int(self._now), "b": owned_summary(player)}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                # Mark profiled only after a successful fetch + write.
                self.profiled[tag] = self._now
                log.write(f"{tag}\t{self._now}\n")
                log.flush()
                new += 1
                pbar.update(1)
                if limit and new >= limit:
                    break
        finally:
            pbar.close()
            out.close()
            log.close()
            self._compact_log()
        return new

    def _compact_log(self) -> None:
        """Rewrite profiled_tags.txt from the in-memory map (dedupe the append-only log) via a
        temp file + atomic replace, so a crash can't truncate it."""
        tmp = PROFILED_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for tag, ts in self.profiled.items():
                fh.write(f"{tag}\t{ts}\n")
        tmp.replace(PROFILED_PATH)


async def _amain(limit: int, recent_days: float, revisit_days: float) -> None:
    async with BrawlStarsClient() as client:
        collector = ProfileCollector(client, revisit_days=revisit_days)
        n = await collector.run(iter_match_tags(recent_days), limit=limit)
    print(f"collected {n} new profiles -> {PROFILES_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect per-player ownership profiles for item win-rate inference.")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new profiles (0 = no cap)")
    ap.add_argument("--recent-days", type=float, default=0.0,
                    help="only profile tags seen in matches within this many days (0 = all)")
    ap.add_argument("--revisit-days", type=float, default=DEFAULT_REVISIT_DAYS,
                    help="re-profile a tag once its last profile is older than this many days")
    args = ap.parse_args()
    _ = settings  # ensure .env (API token) is loaded before the client is built
    asyncio.run(_amain(args.limit, args.recent_days, args.revisit_days))


if __name__ == "__main__":
    main()
