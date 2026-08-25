"""Pull published artifacts (the matches dataset and the win-prob model) from remote URLs.

The home crawler publishes ``data/raw/matches.jsonl`` (gzipped) and, after a retrain, the
``winprob.npz`` model to a GitHub Release. The deployed API calls :func:`sync_matches` and
:func:`sync_model` periodically to refresh its local copies so it can rebuild draft stats and
hot-swap the model without a restart. Each downloads to the same path the engine reads by
default, so a plain rebuild/reload picks up the new bytes.

Robust by design: a conditional GET (ETag) skips the download when nothing changed, a content
hash avoids needless rebuilds when the bytes are identical, and any network/HTTP failure
leaves the last-good local copy in place (returns False rather than raising).
"""
from __future__ import annotations

import hashlib
import logging
import zlib
from pathlib import Path

import httpx

from bsdraft.constants import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

MATCHES_PATH = RAW_DIR / "matches.jsonl"
_ETAG_PATH = RAW_DIR / ".matches.etag"
_SHA_PATH = RAW_DIR / ".matches.sha"

MODEL_PATH = PROCESSED_DIR / "winprob.npz"
_MODEL_ETAG_PATH = PROCESSED_DIR / ".winprob.etag"
_MODEL_SHA_PATH = PROCESSED_DIR / ".winprob.sha"

STATS_PATH = PROCESSED_DIR / "stats.json"
_STATS_ETAG_PATH = PROCESSED_DIR / ".stats.etag"
_STATS_SHA_PATH = PROCESSED_DIR / ".stats.sha"

RANK_INDEX_PATH = PROCESSED_DIR / "rank_index.npz"
_RANK_ETAG_PATH = PROCESSED_DIR / ".rank_index_npz.etag"
_RANK_SHA_PATH = PROCESSED_DIR / ".rank_index_npz.sha"

META_REPORT_PATH = PROCESSED_DIR / "meta_report.json"
_META_ETAG_PATH = PROCESSED_DIR / ".meta_report.etag"
_META_SHA_PATH = PROCESSED_DIR / ".meta_report.sha"

ITEMSTATS_PATH = PROCESSED_DIR / "itemstats.json"
_ITEMSTATS_ETAG_PATH = PROCESSED_DIR / ".itemstats.etag"
_ITEMSTATS_SHA_PATH = PROCESSED_DIR / ".itemstats.sha"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _sync_file(url: str, dest: Path, etag_path: Path, sha_path: Path,
               timeout: float, label: str) -> bool:
    """Refresh ``dest`` from ``url`` if it changed. Returns True iff the local copy was
    rewritten. Conditional GET (ETag) skips the download when nothing changed; a content hash
    skips the rewrite when the bytes are identical; any network/HTTP failure leaves the
    last-good local copy in place (returns False, never raises).

    The response is **streamed** straight to disk, gunzipping on the fly when gzip-framed (a URL
    may point at .gz or the raw file; winprob.npz is zip-framed, not gzip, so it passes through
    untouched). This keeps peak RAM at one chunk rather than the whole file — matches.jsonl is
    >130 MB decompressed, and holding it (plus zlib's scratch buffers) in memory was blowing the
    512 MB free-tier limit on every hourly refresh."""
    if not url:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    etag = _read(etag_path)
    if etag and dest.exists():
        headers["If-None-Match"] = etag

    tmp = dest.parent / (dest.name + ".tmp")
    hasher = hashlib.sha256()       # over the *decompressed* bytes, matching the stored .sha
    new_etag = ""
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 304:
                    return False
                resp.raise_for_status()
                new_etag = resp.headers.get("ETag", "")
                dec = None          # zlib gzip decompressor, created once we see the magic bytes
                sniffed = False
                with open(tmp, "wb") as out:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        if not sniffed:
                            sniffed = True
                            if chunk[:2] == b"\x1f\x8b":
                                dec = zlib.decompressobj(wbits=31)  # 16 + MAX_WBITS = gzip framing
                        if dec is not None:
                            chunk = dec.decompress(chunk)
                        if chunk:
                            hasher.update(chunk)
                            out.write(chunk)
                    if dec is not None:
                        tail = dec.flush()
                        if tail:
                            hasher.update(tail)
                            out.write(tail)
    except Exception as e:  # noqa: BLE001 — never let a sync failure take down serving
        logger.warning("%s sync failed (%s); keeping last-good copy", label, e)
        tmp.unlink(missing_ok=True)
        return False

    if new_etag:
        etag_path.write_text(new_etag, encoding="utf-8")

    sha = hasher.hexdigest()
    if sha == _read(sha_path) and dest.exists():
        tmp.unlink(missing_ok=True)
        return False  # bytes identical — nothing downstream to rebuild

    tmp.replace(dest)  # atomic swap on POSIX
    sha_path.write_text(sha, encoding="utf-8")
    logger.info("%s updated (%.2f MB)", label, dest.stat().st_size / 1e6)
    return True


def sync_matches(url: str, timeout: float = 60.0) -> bool:
    """Refresh the local matches dataset from ``url``. Returns True iff local data changed."""
    return _sync_file(url, MATCHES_PATH, _ETAG_PATH, _SHA_PATH, timeout, "matches")


def sync_model(url: str, timeout: float = 60.0) -> bool:
    """Refresh the local win-prob model (winprob.npz) from ``url``. Returns True iff it changed,
    so the caller can reload and hot-swap the served model."""
    return _sync_file(url, MODEL_PATH, _MODEL_ETAG_PATH, _MODEL_SHA_PATH, timeout, "model")


def sync_stats(url: str, timeout: float = 60.0) -> bool:
    """Refresh the precomputed empirical stats (stats.json) from ``url``. Returns True iff it
    changed, so the caller can reload and hot-swap the served stats — no in-memory rebuild from
    the full match dataset (which OOMs a small instance as the data grows)."""
    return _sync_file(url, STATS_PATH, _STATS_ETAG_PATH, _STATS_SHA_PATH, timeout, "stats")


def sync_rank_index(url: str, timeout: float = 60.0) -> bool:
    """Refresh the precomputed player-rank index (rank_index.npz) from ``url``. Returns True iff
    it changed, so the caller can reload it — no in-memory rebuild of the ~3M-entry tag->tier
    dict from the full match dataset (~200 MB, which OOMs a small instance; see
    :mod:`bsdraft.engine.rank_store`). The npz passes through the gzip sniff below untouched
    (PK-framed, like winprob.npz); the loader dispatches on content, so a legacy gzipped-JSON
    URL still works through this same path."""
    return _sync_file(url, RANK_INDEX_PATH, _RANK_ETAG_PATH, _RANK_SHA_PATH, timeout, "rank index")


def sync_meta_report(url: str, timeout: float = 60.0) -> bool:
    """Refresh the precomputed meta-drift report (meta_report.json, a few KB) from ``url``.
    ``/api/meta`` serves this file directly — recomputing drift streams the full match dataset
    twice per data change, which takes minutes on a small cloud CPU (see
    :mod:`bsdraft.engine.drift`)."""
    return _sync_file(url, META_REPORT_PATH, _META_ETAG_PATH, _META_SHA_PATH, timeout, "meta report")


def sync_itemstats(url: str, timeout: float = 60.0) -> bool:
    """Refresh the precomputed per-item win-rate table (itemstats.json.gz Release asset) from
    ``url``. Returns True iff it changed. Built off-box from the matches x ownership-profiles join
    (needs the profiles, which only the home machine collects); the API just LOADS the small table
    so /api/loadout can serve measured picks with no in-memory join."""
    return _sync_file(url, ITEMSTATS_PATH, _ITEMSTATS_ETAG_PATH, _ITEMSTATS_SHA_PATH, timeout, "itemstats")
