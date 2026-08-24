"""Collect ranked matches via the snowball crawler.

    # one-shot crawl (local only):
    PYTHONPATH=backend python backend/scripts/collect.py --target 3000
    PYTHONPATH=backend python backend/scripts/collect.py --target 500 --countries global,US

    # home daemon for the live site: crawl a batch, publish, sleep, repeat:
    PYTHONPATH=backend python backend/scripts/collect.py --loop 3600 --target 800 --publish

    # …and auto-retrain the model whenever the meta shifts (balance change / new brawler):
    PYTHONPATH=backend python backend/scripts/collect.py --loop 3600 --target 800 --publish --retrain-on-shift

Resumable: re-running continues from the existing matches/visited state in data/raw/. Pass
--publish to upload matches.jsonl.gz to a GitHub Release (see collect/publish.py) so the
deployed API can pull it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from bsdraft.collect import publish as publisher
from bsdraft.collect.client import AuthError, BrawlStarsClient
from bsdraft.collect.crawler import MATCHES_PATH, Crawler
from bsdraft.config import settings
from bsdraft.constants import PROCESSED_DIR, RAW_DIR, REPO_ROOT
from bsdraft.engine.drift import detect_drift, save_report

# Written when every request fails auth (key/IP broken); cleared on the next good crawl.
# The keep-warm workflow's data-staleness issue tells the human to look for this file.
STALLED_SENTINEL = RAW_DIR / "COLLECT_STALLED"


async def _run(target: int, countries: list, revisit_after: float) -> int:
    async with BrawlStarsClient() as client:
        crawler = Crawler(client, revisit_after=revisit_after)
        seeds = [settings.player_tag] if settings.player_tag else []
        backlog = len(crawler.frontier)  # players recovered from prior state, before seeding
        await crawler.seed(countries, seed_tags=seeds)
        queued = len(crawler.frontier)
        print(f"Resuming: {len(crawler.seen_matches)} matches, "
              f"{len(crawler.visited)} players scanned.")
        print(f"Queued {queued} players to scan ({backlog} recovered backlog + "
              f"{queued - backlog} new from rankings ({', '.join(countries)})).")
        new = await crawler.run(target_matches=target)
        print(f"\nDone. +{new} new matches  ->  {MATCHES_PATH}")
        print(f"Total unique matches: {len(crawler.seen_matches)}")
        return new


def _alert_stalled(err: Exception) -> None:
    """Fail LOUDLY on an auth/IP error: every request 403s until a human fixes the key's
    allow-list (the home IP rotated off it once and the crawler ran a month collecting
    nothing, silently). Sentinel file + stderr + best-effort macOS notification."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n!!! COLLECT STALLED at {now}: {err}", file=sys.stderr)
    print("!!! Every request is failing auth — fix the Supercell key's IP allow-list at "
          "https://developer.brawlstars.com (or mint a new key), then update .env.",
          file=sys.stderr)
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        STALLED_SENTINEL.write_text(f"{now}\n{err}\n", encoding="utf-8")
    except OSError as e:
        print(f"(could not write sentinel {STALLED_SENTINEL}: {e})", file=sys.stderr)
    if sys.platform == "darwin":  # the crawler lives on the home Mac
        try:
            subprocess.run(
                ["osascript", "-e",
                 'display notification "Auth/IP error — every request failing. '
                 'See data/raw/COLLECT_STALLED." with title "Brawl Stars crawler stalled"'],
                check=False, capture_output=True, timeout=10,
            )
        except Exception:  # noqa: BLE001 — the notification is a bonus, never a failure
            pass


def _clear_stalled() -> None:
    if STALLED_SENTINEL.exists():
        print("auth recovered — clearing the COLLECT_STALLED sentinel.")
        STALLED_SENTINEL.unlink(missing_ok=True)


def _try_publish() -> None:
    try:
        publisher.publish()
    except Exception as e:  # noqa: BLE001 — a publish hiccup shouldn't kill a long crawl loop
        print(f"publish failed: {e}")


def _publish_stats() -> None:
    """Build the full empirical stats and publish stats.json.gz so the live API LOADS them
    instead of rebuilding from the whole dataset (which OOMs the 512 MB free tier). Best-effort
    — never kills the crawl loop."""
    try:
        from bsdraft.engine.stats import build_bracketed
        from bsdraft.engine.stats_store import save_stats
        g, br = build_bracketed(halflife_days=settings.stats_halflife_days)  # all matches
        save_stats(g, br, publisher.STATS_PATH)
        publisher.publish_stats()
    except Exception as e:  # noqa: BLE001 — a stats hiccup shouldn't kill a long crawl loop
        print(f"stats publish failed: {e}")


def _publish_rank_index() -> None:
    """Build the full tag->Ranked-tier index and publish BOTH containers so the live API LOADS it
    instead of building a ~200 MB dict from the whole dataset (which OOMs the 512 MB free tier).

    Two assets on purpose, for the duration of the format migration: ``rank_index.npz`` is what
    RANK_INDEX_URL points at (~66 MB peak to load), and the legacy ``rank_index.json.gz`` keeps
    being refreshed so that reverting RANK_INDEX_URL lands on a *current* artifact instead of a
    frozen one. The expensive part — the full matches.jsonl scan — runs once and feeds both.

    Each upload is independent: a hiccup writing the new asset must not stop the legacy one the
    deployed API may still be reading, and vice versa. Best-effort — never kills the crawl loop."""
    try:
        from bsdraft.engine.playerrank import build_rank_index
        from bsdraft.engine.rank_store import save_rank_index
        idx = build_rank_index()
        save_rank_index(idx, publisher.RANK_INDEX_PATH)
        save_rank_index(idx, publisher.RANK_INDEX_NPZ_PATH)
    except Exception as e:  # noqa: BLE001 — a rank-index hiccup shouldn't kill a long crawl loop
        print(f"rank index build failed: {e}")
        return
    for path in (publisher.RANK_INDEX_PATH, publisher.RANK_INDEX_NPZ_PATH):   # legacy first
        try:
            publisher.publish_rank_index(path=path)
        except Exception as e:  # noqa: BLE001 — one asset failing must not block the other
            print(f"rank index publish failed ({path.name}): {e}")


def _check_meta(retrain_on_shift: bool, publish: bool = False) -> None:
    """Run the meta-drift detector on the freshly crawled data and print the report. The report
    is written to data/processed/meta_report.json and (with ``publish``) uploaded so the live
    API SERVES it instead of recomputing drift per data change — two streaming passes over the
    full dataset, minutes on the free tier's CPU sliver. When the meta has shifted and
    ``--retrain-on-shift`` is set, kick a model retrain so recommendations catch up. Never
    raises — a drift hiccup must not kill a long crawl loop."""
    try:
        report = detect_drift()
    except Exception as e:  # noqa: BLE001
        print(f"meta check failed: {e}")
        return
    print("\n--- meta drift ---")
    print(report.summary())
    try:
        save_report(report, publisher.META_REPORT_PATH)
        if publish:
            publisher.publish_meta_report()
    except Exception as e:  # noqa: BLE001 — a report hiccup shouldn't kill a long crawl loop
        print(f"meta report publish failed: {e}")
    if report.shifted and retrain_on_shift:
        _retrain()


# A retrain can fail the same way every hour — most often train.py's --max-full-delta gate
# refusing a checkpoint that is a hair worse on full comps. That is the gate doing its job, but a
# *run* of refusals means the served model is frozen while the meta moves, and the only trace is
# a line in crawl.out.log that nobody reads. (Found 2026-08-20 after 38 consecutive silent
# failures left the live model 8 days stale.) Track the streak across restarts and escalate it
# into a GitHub issue, the same channel the CI watchers use.
_RETRAIN_STATE_PATH = PROCESSED_DIR / "retrain_state.json"
_RETRAIN_ALERT_AFTER = 3       # ~3h at the default --loop 3600 — past a transient
_RETRAIN_REALERT_EVERY = 24    # and once a day after that, so a long stall keeps nagging

# Best-of-N seeds for the unattended retrain. The paired full-comp delta swings more between
# seeds (~0.0035, measured by gate_experiment.py 2026-08-23) than train.py's 0.002 gate, so a
# single-seed retrain passes or fails the gate by luck — the cause of the 2026-08 stall streak.
# Training 3 seeds and keeping the lowest full-comp val logloss reliably surfaces a candidate at
# or below the incumbent, so the gate passes on merit instead of coin-flip. Costs ~3x train time
# (minutes) per shifted cycle.
_RETRAIN_CANDIDATES = 3

# Full-comp regression gate for the unattended retrain, widened from train.py's strict 0.002
# default to the measured seed-to-seed noise floor (~0.0035, gate_experiment.py 2026-08-23). At
# 0.002 the gate sat BELOW that floor, so even the best of N candidates (e.g. +0.0021 on the first
# post-fix cycle) is within noise of the incumbent yet rejected — freezing the model on what is not
# a real regression. At the floor a genuine regression (>> 0.0035) is still blocked while best-of-N
# ships a fresher, noise-equivalent model. Manual `train.py` runs keep the strict 0.002 default.
_RETRAIN_MAX_FULL_DELTA = 0.0035


def _retrain_state() -> dict:
    try:
        return json.loads(_RETRAIN_STATE_PATH.read_text())
    except Exception:  # noqa: BLE001 — absent or corrupt state just starts a fresh streak
        return {}


def _record_retrain(ok: bool, detail: str = "") -> int:
    """Update the consecutive-failure streak and return it (0 on success)."""
    state = _retrain_state()
    streak = 0 if ok else int(state.get("consecutive_failures", 0)) + 1
    try:
        _RETRAIN_STATE_PATH.write_text(json.dumps(
            {"consecutive_failures": streak, "last_detail": detail[-2000:]}, indent=2))
    except Exception as e:  # noqa: BLE001 — bookkeeping must not kill the crawl loop
        print(f"retrain state write failed: {e}")
    return streak


def _alert_retrain_stalled(streak: int, detail: str) -> None:
    """File one issue per stalled streak (then daily), deduped on the open `model-stale` issue."""
    if streak < _RETRAIN_ALERT_AFTER:
        return
    if streak > _RETRAIN_ALERT_AFTER and (streak - _RETRAIN_ALERT_AFTER) % _RETRAIN_REALERT_EVERY:
        return
    body = (
        f"The auto-retrain has failed **{streak} cycles in a row**, so the model served by the "
        f"live API is frozen while `meta_report.json` keeps reporting a shifted meta.\n\n"
        f"Most often this is `train.py`'s full-comp regression gate refusing a checkpoint that is "
        f"marginally worse than the incumbent — see `docs/MODEL_CARD.md`. Check whether the "
        f"dataset is actually growing before touching the gate: retraining on a frozen dataset "
        f"reproduces the incumbent plus noise, and the noise alone can exceed the threshold.\n\n"
        f"Last failure:\n\n```\n{detail[-2000:] or '(no output captured)'}\n```\n"
    )
    try:
        existing = publisher._gh("issue", "list", "--label", "model-stale", "--state", "open",
                                 "--limit", "1", "--json", "number")
        if existing.returncode == 0 and json.loads(existing.stdout or "[]"):
            print(f"retrain stalled ({streak} cycles) — open model-stale issue already tracks it")
            return
        title = f"Model retrain stalled — {streak} consecutive failures"
        res = publisher._gh("issue", "create", "--title", title,
                            "--label", "model-stale", "--body", body)
        if res.returncode != 0:
            # `gh` refuses an unknown label outright. The alert matters more than the taxonomy,
            # so fall back to an unlabelled issue rather than losing it (the dedupe check above
            # is label-scoped, so an unlabelled fallback re-files daily — acceptable, and it
            # stops as soon as someone creates the label).
            res = publisher._gh("issue", "create", "--title", title, "--body", body)
        print(f"retrain stalled ({streak} cycles) -> filed issue"
              if res.returncode == 0 else f"model-stale issue create failed: {res.stderr.strip()}")
    except Exception as e:  # noqa: BLE001 — alerting is best-effort
        print(f"model-stale alert failed: {e}")


def _retrain() -> None:
    """Retrain, re-export, and publish the win-prob model so it reflects the shifted meta. The
    deployed API (with MODEL_URL set) hot-swaps the published model on its next refresh — no
    redeploy. Guarded so a failure can't take down the crawl loop."""
    print("meta shifted -> retraining win-prob model …")
    scripts = REPO_ROOT / "backend" / "scripts"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "backend")}
    try:
        # stderr is captured (and echoed) rather than inherited so the gate's own explanation can
        # ride along into the alert instead of being lost in crawl.err.log's progress-bar noise.
        # train.py trains best-of-N seeds (the gate is below the seed-noise floor); export_model.py
        # then serializes whichever candidate won.
        commands = [
            [sys.executable, str(scripts / "train.py"),
             "--candidates", str(_RETRAIN_CANDIDATES),
             "--max-full-delta", str(_RETRAIN_MAX_FULL_DELTA)],
            [sys.executable, str(scripts / "export_model.py")],
        ]
        for cmd in commands:
            res = subprocess.run(cmd, check=True, env=env, stderr=subprocess.PIPE, text=True)
            if res.stderr:
                sys.stderr.write(res.stderr)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip()
        if detail:
            sys.stderr.write(detail + "\n")
        streak = _record_retrain(False, detail)
        print(f"retrain failed ({streak} cycle(s) in a row): {e}")
        _alert_retrain_stalled(streak, detail)
        return
    except Exception as e:  # noqa: BLE001
        streak = _record_retrain(False, str(e))
        print(f"retrain failed ({streak} cycle(s) in a row): {e}")
        _alert_retrain_stalled(streak, str(e))
        return
    _record_retrain(True)
    try:
        publisher.publish_model()
        print("retrain complete — new model published; the live API hot-swaps it on its next refresh")
    except Exception as e:  # noqa: BLE001 — export succeeded; publishing is best-effort
        print(f"retrain complete, but model publish failed ({e}); "
              f"roll out manually: python -m bsdraft.collect.publish --only-model")


async def _loop(target: int, countries: list, interval: int, do_publish: bool,
                meta_check: bool, retrain_on_shift: bool, revisit_after: float) -> None:
    cycle = 0
    while True:
        cycle += 1
        print(f"\n=== crawl cycle {cycle} ===")
        try:
            await _run(target, countries, revisit_after)
            _clear_stalled()
            if do_publish:
                _try_publish()
                _publish_stats()
                _publish_rank_index()
            if meta_check:
                _check_meta(retrain_on_shift, publish=do_publish)
        except AuthError as e:
            # Alert but don't die (launchd would respawn us into the same wall) and don't
            # publish — retry next cycle in case the allow-list gets fixed meanwhile.
            _alert_stalled(e)
            print(f"sleeping {interval}s before retrying …  (Ctrl-C to stop)")
            await asyncio.sleep(interval)
            continue
        except Exception as e:  # noqa: BLE001 — one bad cycle (network drop, publish hiccup)
            # must never kill the daemon; log it and retry after the normal sleep. Without this
            # a transport error crashed the process, forcing a launchd restart + a full
            # matches.jsonl state reload, and reset the cycle counter to 1.
            print(f"cycle {cycle} aborted: {e!r} — retrying after the sleep")
        print(f"sleeping {interval}s …  (Ctrl-C to stop)")
        await asyncio.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="Crawl ranked Brawl Stars matches.")
    ap.add_argument("--target", type=int, default=2000, help="number of NEW matches to collect per run")
    ap.add_argument("--countries", default=None,
                    help="comma-separated country codes; defaults to .env CRAWL_SEED_COUNTRIES")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS",
                    help="run forever: crawl --target, publish, sleep SECONDS, repeat")
    ap.add_argument("--publish", action="store_true",
                    help="after each crawl, upload matches.jsonl.gz to the GitHub Release (live site)")
    ap.add_argument("--no-meta-check", dest="meta_check", action="store_false",
                    help="skip the meta-drift report after each crawl (on by default)")
    ap.add_argument("--retrain-on-shift", action="store_true",
                    help="when the meta-drift check trips, retrain + re-export the win-prob model")
    ap.add_argument("--revisit-hours", type=float, default=None,
                    help="re-scan a known player after this many hours to catch their newer "
                         "ranked games (default: .env CRAWL_REVISIT_HOURS; 0 disables)")
    ap.set_defaults(meta_check=True)
    args = ap.parse_args()
    raw = args.countries.split(",") if args.countries else settings.seed_countries
    countries = [c.strip() for c in raw if c.strip()]
    revisit_hours = args.revisit_hours if args.revisit_hours is not None else settings.crawl_revisit_hours
    revisit_after = max(0.0, revisit_hours) * 3600

    if args.loop > 0 and not args.publish:
        print("note: --loop without --publish — crawling locally only; the live site won't update.")

    try:
        if args.loop > 0:
            asyncio.run(_loop(args.target, countries, args.loop, args.publish,
                              args.meta_check, args.retrain_on_shift, revisit_after))
        else:
            asyncio.run(_run(args.target, countries, revisit_after))
            _clear_stalled()
            if args.publish:
                _try_publish()
            if args.meta_check:
                _check_meta(args.retrain_on_shift, publish=args.publish)
    except AuthError as e:  # loop mode handles this per-cycle; one-shot fails loudly
        _alert_stalled(e)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
