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
import json
import math
import time
import zlib
from pathlib import Path
from typing import Callable, Optional

import httpx

from bsdraft.constants import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)
_sync_status: dict = {}

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


def sync_status() -> dict:
    """Small per-artifact status snapshot; errors contain categories, never remote URLs/secrets."""
    return {name: dict(value) for name, value in _sync_status.copy().items()}


def artifact_identity(path: Path, sha_path: Path) -> dict:
    return {"sha256": _read(sha_path) or None,
            "size_bytes": path.stat().st_size if path.exists() else None}


def _validate_artifact(path: Path, label: str) -> None:
    """Validate staged publications before replacing a usable local artifact."""
    if not path.stat().st_size:
        raise ValueError("empty artifact")
    if label == "model":
        import numpy as np
        from bsdraft.models.serve import WinProbModel
        from bsdraft.data import reference as reference
        model = WinProbModel(path, validate_current_era=True)
        if not model.available or not all(np.isfinite(w).all() for w in model._w.values()):
            raise ValueError("invalid model weights or balance era")
        ids = [b.id for b in reference.pickable_brawlers()][:6]
        maps = reference.load_ranked_maps()
        if len(ids) < 6 or not maps:
            raise ValueError("missing reference vocabulary")
        probability = model.prob(ids[:3], ids[3:], maps[0].id, maps[0].mode)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("invalid model prediction")
    elif label == "stats":
        from bsdraft.data.balance_eras import current_balance_era
        from bsdraft.engine.stats_store import load_stats
        global_stats, brackets = load_stats(path)
        era = current_balance_era()
        for table in (global_stats, *brackets.values()):
            if not isinstance(table.n, int) or table.n < 0:
                raise ValueError("invalid stats sample count")
            if era and (table.analysis_era_id != era.id or table.analysis_start_ts != era.start_ts):
                raise ValueError("stats balance era mismatch")
            for name in ("b_games", "b_wins", "map_games", "bm_games", "bm_wins",
                         "cnt_games", "cnt_wins", "syn_games", "syn_wins"):
                if any(not math.isfinite(v) or v < 0 for v in getattr(table, name).values()):
                    raise ValueError("invalid stats value")
    elif label == "rank index":
        from bsdraft.engine.rank_store import load_rank_index
        load_rank_index(path)
    elif label == "meta report":
        from bsdraft.engine.drift import load_report
        report = load_report(path)
        if report.n_recent < 0 or report.n_prior < 0 or report.newest_ts < 0:
            raise ValueError("invalid meta sample count")
    elif label == "itemstats":
        data = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("version") != 1 or
                not isinstance(data.get("cells"), dict) or
                not isinstance(data.get("meta"), dict)):
            raise ValueError("invalid itemstats schema")
        for cell in data["cells"].values():
            if not isinstance(cell, dict):
                raise ValueError("invalid itemstats cell")
            for key in ("delta", "item_winrate", "n_eff", "n_players"):
                if key in cell and (not isinstance(cell[key], (int, float)) or
                                    not math.isfinite(cell[key])):
                    raise ValueError("invalid itemstats value")
    elif label == "matches":
        # Transport completion is checked below. Sampling boundaries avoids replaying the
        # entire multi-GB dataset on the public host just to validate a download.
        with path.open("rb") as source:
            first = source.readline(1 << 20)
            source.seek(max(0, path.stat().st_size - (1 << 20)))
            tail = source.read().splitlines()
        for line in (first, tail[-1] if tail else b""):
            row = json.loads(line)
            if not isinstance(row, dict) or not {"team_a", "team_b", "map_id"} <= row.keys():
                raise ValueError("invalid match schema")


def _sync_file(url: str, dest: Path, etag_path: Path, sha_path: Path,
               timeout: float, label: str,
               validator: Optional[Callable[[Path], None]] = None) -> bool:
    """Stream to staging, verify gzip completion and schema, then atomically promote.

    Failed downloads and invalid HTTP-200 publications preserve the file and conditional-GET
    metadata. The SHA is a content identity, not proof of independent publication provenance.
    """
    if not url:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    etag = _read(etag_path)
    if etag and dest.exists():
        headers["If-None-Match"] = etag
    tmp = dest.parent / (dest.name + ".tmp")
    hasher = hashlib.sha256()
    started = time.monotonic()
    now = time.time()
    previous = _sync_status.get(label, {})
    _sync_status[label] = {**previous, "last_attempt": now}
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 304 and dest.exists():
                    _sync_status[label].update(last_success=time.time(), error=None)
                    return False
                resp.raise_for_status()
                new_etag = resp.headers.get("ETag", "")
                dec = None
                prefix = b""
                sniffed = False
                with open(tmp, "wb") as out:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        if time.monotonic() - started > timeout:
                            raise TimeoutError("artifact deadline exceeded")
                        if not chunk:
                            continue
                        if not sniffed:
                            prefix += chunk
                            if len(prefix) < 2:
                                continue
                            chunk, prefix = prefix, b""
                            sniffed = True
                            if chunk[:2] == b"\x1f\x8b":
                                dec = zlib.decompressobj(wbits=31)
                        if dec is not None:
                            chunk = dec.decompress(chunk)
                        if chunk:
                            hasher.update(chunk)
                            out.write(chunk)
                    if prefix:
                        hasher.update(prefix)
                        out.write(prefix)
                    if dec is not None:
                        tail = dec.flush()
                        if not dec.eof or dec.unused_data:
                            raise ValueError("incomplete or trailing gzip data")
                        if tail:
                            hasher.update(tail)
                            out.write(tail)
        sha = hasher.hexdigest()
        if sha != _read(sha_path) or not dest.exists():
            (validator or (lambda path: _validate_artifact(path, label)))(tmp)
            tmp.replace(dest)
            sha_path.write_text(sha, encoding="utf-8")
            changed = True
        else:
            tmp.unlink(missing_ok=True)
            changed = False
        # Do not pin the remote ETag until validation/promotion succeeds.
        etag_path.write_text(new_etag, encoding="utf-8")
        _sync_status[label].update(last_success=time.time(), error=None, sha256=sha)
        if changed:
            logger.info("%s updated (%.2f MB)", label, dest.stat().st_size / 1e6)
        return changed
    except Exception as exc:  # no signed URL, auth header, or upstream response in public errors
        _sync_status[label].update(error=type(exc).__name__)
        logger.warning("%s sync rejected (%s); keeping last-good copy", label, type(exc).__name__)
        tmp.unlink(missing_ok=True)
        return False


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
