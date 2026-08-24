"""Publish the collected matches (and the trained model) to a GitHub Release for the API to pull.

Gzips ``data/raw/matches.jsonl`` and uploads it as the ``matches.jsonl.gz`` asset on a fixed
release tag (default ``data-latest``), replacing the previous asset; :func:`publish_model`
uploads ``winprob.npz`` alongside it. The cloud API's ``DATA_URL`` / ``MODEL_URL`` point at
those assets' stable download URLs:

    https://github.com/<owner>/<repo>/releases/download/data-latest/matches.jsonl.gz
    https://github.com/<owner>/<repo>/releases/download/data-latest/winprob.npz

Requires the GitHub CLI (`gh`), authenticated, run from inside the repo (gh infers the
owner/repo from the git remote). Keeping this on your machine is what lets the crawl keep
using the IP-locked Supercell key while the cloud stays free.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
from pathlib import Path

from bsdraft.constants import PROCESSED_DIR, RAW_DIR

MATCHES_PATH = RAW_DIR / "matches.jsonl"
GZ_PATH = RAW_DIR / "matches.jsonl.gz"
MODEL_PATH = PROCESSED_DIR / "winprob.npz"
STATS_PATH = PROCESSED_DIR / "stats.json.gz"
RANK_INDEX_PATH = PROCESSED_DIR / "rank_index.json.gz"       # legacy container (rollback target)
RANK_INDEX_NPZ_PATH = PROCESSED_DIR / "rank_index.npz"       # current container
META_REPORT_PATH = PROCESSED_DIR / "meta_report.json"
ITEMSTATS_PATH = PROCESSED_DIR / "itemstats.json.gz"
DEFAULT_TAG = "data-latest"


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def _ensure_release(tag: str) -> None:
    if _gh("release", "view", tag).returncode != 0:
        res = _gh(
            "release", "create", tag,
            "--title", "Latest dataset",
            "--notes", "Rolling ranked-match dataset powering the live draft API (updated by the crawler).",
        )
        if res.returncode != 0:
            raise RuntimeError(f"gh release create failed: {res.stderr.strip()}")


def gzip_matches(src: Path = MATCHES_PATH, dst: Path = GZ_PATH) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"No matches file at {src} — run the crawler first.")
    with open(src, "rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout)
    return dst


def publish(tag: str = DEFAULT_TAG) -> None:
    gz = gzip_matches()
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(gz), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload failed: {res.stderr.strip()}")
    print(f"published {gz.name} ({gz.stat().st_size / 1e6:.1f} MB) -> release '{tag}'")


def publish_model(tag: str = DEFAULT_TAG) -> None:
    """Upload winprob.npz to the release so an API with MODEL_URL set can hot-swap it. Run
    after export_model.py (the crawler does this automatically on a retrain-on-shift)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH} — export it first (scripts/export_model.py).")
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(MODEL_PATH), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload (model) failed: {res.stderr.strip()}")
    print(f"published {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.0f} KB) -> release '{tag}'")


def publish_stats(tag: str = DEFAULT_TAG) -> None:
    """Upload the precomputed stats (stats.json.gz) to the release so an API with STATS_URL set
    loads them instead of rebuilding from the full dataset. Run after scripts/export_stats.py
    (the crawler does this automatically each publish cycle)."""
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"No stats at {STATS_PATH} — build them first (scripts/export_stats.py).")
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(STATS_PATH), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload (stats) failed: {res.stderr.strip()}")
    print(f"published {STATS_PATH.name} ({STATS_PATH.stat().st_size / 1e6:.1f} MB) -> release '{tag}'")


def publish_rank_index(tag: str = DEFAULT_TAG, path: Path = RANK_INDEX_PATH) -> None:
    """Upload a precomputed rank index to the release so an API with RANK_INDEX_URL set loads the
    tag->tier lookup instead of building a ~200 MB dict from the full dataset. Run after
    scripts/export_rank_index.py (the crawler does this each cycle).

    ``path`` picks the container — ``RANK_INDEX_NPZ_PATH`` (current) or ``RANK_INDEX_PATH``
    (legacy gzipped JSON, still published during the migration so reverting RANK_INDEX_URL lands
    on a *fresh* artifact rather than a frozen one). ``gh`` names the Release asset after the
    file's basename, so this argument alone decides which asset is written."""
    if not path.exists():
        raise FileNotFoundError(
            f"No rank index at {path} — build it first (scripts/export_rank_index.py).")
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(path), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload (rank index) failed: {res.stderr.strip()}")
    print(f"published {path.name} ({path.stat().st_size / 1e6:.1f} MB) -> release '{tag}'")


def publish_meta_report(tag: str = DEFAULT_TAG) -> None:
    """Upload the meta-drift report (meta_report.json, a few KB) to the release so an API with
    META_REPORT_URL set *serves* it instead of recomputing drift over the full dataset per
    request — two streaming passes over every match, minutes on a small cloud CPU. The crawler
    writes + publishes it after each cycle's meta check (see scripts/collect.py)."""
    if not META_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"No meta report at {META_REPORT_PATH} — the crawler's meta check writes it.")
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(META_REPORT_PATH), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload (meta report) failed: {res.stderr.strip()}")
    print(f"published {META_REPORT_PATH.name} ({META_REPORT_PATH.stat().st_size / 1e3:.1f} KB) -> release '{tag}'")


def publish_itemstats(tag: str = DEFAULT_TAG) -> None:
    """Upload the per-item win-rate table (itemstats.json.gz) to the release so an API with
    ITEMSTATS_URL set serves data-driven loadout picks instead of the effect heuristic. Run after
    scripts/export_itemstats.py (which needs the collected ownership profiles)."""
    if not ITEMSTATS_PATH.exists():
        raise FileNotFoundError(
            f"No item stats at {ITEMSTATS_PATH} — build them first (scripts/export_itemstats.py).")
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(ITEMSTATS_PATH), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload (itemstats) failed: {res.stderr.strip()}")
    print(f"published {ITEMSTATS_PATH.name} ({ITEMSTATS_PATH.stat().st_size / 1e3:.1f} KB) -> release '{tag}'")


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish the dataset and/or model/stats/rank index to a GitHub Release.")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="release tag to upload to")
    ap.add_argument("--model", action="store_true", help="also upload winprob.npz (the model)")
    ap.add_argument("--stats", action="store_true", help="also upload stats.json.gz (precomputed stats)")
    ap.add_argument("--rank", action="store_true", help="also upload rank_index.json.gz (legacy rank index)")
    ap.add_argument("--rank-npz", action="store_true", help="also upload rank_index.npz (current rank index)")
    ap.add_argument("--meta", action="store_true", help="also upload meta_report.json (drift report)")
    ap.add_argument("--itemstats", action="store_true", help="also upload itemstats.json.gz (per-item win rates)")
    ap.add_argument("--only-model", action="store_true", help="upload only winprob.npz, not the dataset")
    ap.add_argument("--only-stats", action="store_true", help="upload only stats.json.gz, not the dataset")
    ap.add_argument("--only-rank", action="store_true", help="upload only rank_index.json.gz, not the dataset")
    ap.add_argument("--only-rank-npz", action="store_true", help="upload only rank_index.npz, not the dataset")
    ap.add_argument("--only-meta", action="store_true", help="upload only meta_report.json, not the dataset")
    ap.add_argument("--only-itemstats", action="store_true", help="upload only itemstats.json.gz, not the dataset")
    args = ap.parse_args()
    if args.only_model:
        publish_model(args.tag)
        return
    if args.only_stats:
        publish_stats(args.tag)
        return
    if args.only_rank:
        publish_rank_index(args.tag)
        return
    if args.only_rank_npz:
        publish_rank_index(args.tag, RANK_INDEX_NPZ_PATH)
        return
    if args.only_meta:
        publish_meta_report(args.tag)
        return
    if args.only_itemstats:
        publish_itemstats(args.tag)
        return
    publish(args.tag)
    if args.model:
        publish_model(args.tag)
    if args.stats:
        publish_stats(args.tag)
    if args.rank:
        publish_rank_index(args.tag)
    if args.rank_npz:
        publish_rank_index(args.tag, RANK_INDEX_NPZ_PATH)
    if args.meta:
        publish_meta_report(args.tag)
    if args.itemstats:
        publish_itemstats(args.tag)


if __name__ == "__main__":
    main()
