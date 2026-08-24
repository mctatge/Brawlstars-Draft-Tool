"""Serialize the player-rank index to a compact artifact and load it back.

Lets the deployed API **load** a precomputed ``tag -> Ranked-tier`` index instead of
**building** it in memory from the full match dataset. The built form is a Python dict with one
entry per crawled tag — at 1.3M tags that's ~200 MB resident and ~45 s to build, scanning the
whole ``matches.jsonl``; it grows with the crawl and was threatening Render's 512 MB free tier
(see ``docs`` / the OOM history). The home machine (with RAM to spare) builds the full index and
publishes ``rank_index.json.gz``; the API syncs it and loads it into a compact NumPy-backed
lookup (~20 MB), with the match data never resident for this. Mirrors the stats/model
publish-load split (:mod:`bsdraft.engine.stats_store`, ``winprob.npz``).

Only the **tier** is served (1-22); the ``build_rank_index`` timestamp is just for picking the
latest tier per tag during the build, so it's dropped from the artifact. The serve form keeps the
tags as a single sorted NumPy byte-string array (``np.searchsorted`` for O(log n) lookup) and the
tiers as a ``uint8`` array — far smaller than 1.3M Python str/int/dict objects.

Two container formats, dispatched on the file's **magic bytes**, never its name — the sync layer
writes every artifact to one fixed local path regardless of what the URL served, so the extension
is not evidence of anything:

- **``.npz`` (current, version 2)** — ``np.savez_compressed`` of the serve arrays themselves:
  ``version`` (0-d int32), ``tags`` (1-D ``S``, ascending), ``tiers`` (1-D uint8). Loading is
  essentially a decompress straight into the destination arrays, with no Python object ever
  materialized per tag. Measured on the live 3.02M-tag index: **13.7 MB asset, ~0.3 s, ~66 MB
  peak RSS**.
- **gzipped JSON (legacy, version 1)** — ``{"version":1,"tags":[...ascending...],"tiers":[...]}``.
  Same logical content, but decoding it costs a Python ``str`` per tag; even the slice-at-a-time
  reader below peaks at **~263 MB** on that same index (13.9 MB asset, ~4.7 s). That transient is
  what OOM-killed the 512 MB Render instance, so the JSON path is now read-only legacy.

The JSON **reader** is kept indefinitely on purpose: it costs nothing at runtime and it is the
rollback — a one-line revert of ``RANK_INDEX_URL`` returns to a format known to work in production.

Tags are sorted by their **encoded** bytes, which is not the same order as sorting the Python
strings: :func:`_ascii_bytes` encodes ascii-and-ignore, so ``sorted(["AB", "Aé"])`` encodes to
``[b'AB', b'A']`` — *descending*, which would silently break the binary search. :func:`_sorted_arrays`
is the one place that ordering is established, so the writer and the in-memory build cannot drift.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

FORMAT_VERSION = 1        # legacy gzipped-JSON container
NPZ_FORMAT_VERSION = 2    # current .npz container

# Widest tag the writer will emit. 'S' is fixed-width, so ONE oversized tag widens all ~3M rows
# (at S10 the tags array is 30 MB; at S40 it would be 121 MB). Real Brawl Stars tags are 9 chars,
# so this is a loud publish-time alarm on the machine with RAM to spare, not a routine limit.
_MAX_TAG_BYTES = 16
# The loader's bound is deliberately looser than the writer's: it is the hard memory ceiling for
# an artifact we did not just build (32 x 3.0M = 96 MB), not a claim about what we publish.
_MAX_LOAD_TAG_BYTES = 32
# Sanity ceiling on the row count — a corrupt/hostile header could otherwise declare anything.
_MAX_TAGS = 50_000_000
_NPZ_MAGIC = b"PK\x03\x04"    # .npz is a zip archive
_GZIP_MAGIC = b"\x1f\x8b"


def _ascii_bytes(tags) -> np.ndarray:
    """Pack tags into a sorted-order ``S`` byte array, encoding ascii-and-ignore so a stray
    non-ASCII tag can't raise (numpy ``dtype='S'`` would otherwise UnicodeEncodeError). Mirrors
    the same encoding :meth:`RankIndex.get` uses for the query, so lookups stay consistent."""
    return np.array([t.encode("ascii", "ignore") if isinstance(t, str) else t for t in tags], dtype="S")


def _sorted_arrays(idx: Dict[str, object]):
    """``{tag: (ts, tier)}`` (or ``{tag: tier}``) → the parallel serve arrays, sorted and validated.

    The one place tag order is established, so the artifact writer and the in-memory build can't
    drift — which matters now that the loader *enforces* ordering rather than trusting it.

    Sorting the **encoded** bytes rather than the Python strings is load-bearing: ``_ascii_bytes``
    encodes ascii-and-ignore, so ``sorted(["AB", "Aé"])`` encodes to ``[b'AB', b'A']`` — descending,
    which would silently break :meth:`RankIndex.get`'s binary search for every tag after it.
    Sorting *after* encoding makes the ascending invariant true by construction.
    """
    n = len(idx)
    tags = _ascii_bytes(idx.keys())
    # int16, not uint8: the range check has to see an out-of-range value, not a wrapped one. (numpy
    # 2.x raises on the out-of-range cast, but requirements-serve.txt floors at an unpinned
    # numpy>=1.26, so check explicitly instead of depending on which version the box resolves.)
    tiers = np.fromiter(((v[1] if isinstance(v, tuple) else v) for v in idx.values()),
                        dtype=np.int16, count=n)
    if n:
        # Tier range is a CORRECTNESS check, so it belongs on every path into the serve arrays:
        # `.astype(uint8)` below wraps silently, where the old `np.array(..., dtype=uint8)` raised.
        # The tag-WIDTH bound is not correctness — it's about the published artifact's size — so it
        # lives in _save_npz, and the in-memory build stays as permissive as it has always been.
        lo, hi = int(tiers.min()), int(tiers.max())
        if lo < 1 or hi > 22:
            raise ValueError(f"tier out of the 1-22 Ranked range (saw {lo}..{hi})")
    order = np.argsort(tags, kind="stable")
    return tags[order], tiers[order].astype(np.uint8)


class RankIndex:
    """Compact, read-only ``tag -> tier`` lookup backed by parallel NumPy arrays.

    ``tags`` is ascending (lexicographic, ASCII) so :meth:`get` binary-searches it. Build it via
    :meth:`from_mapping` (from the dict :func:`bsdraft.engine.playerrank.build_rank_index` returns)
    or :func:`load_rank_index` (from the published artifact)."""

    __slots__ = ("_tags", "_tiers")

    def __init__(self, tags: np.ndarray, tiers: np.ndarray):
        # tags: sorted ascending, dtype 'S*'; tiers: uint8, parallel to tags.
        self._tags = tags
        self._tiers = tiers

    @classmethod
    def from_mapping(cls, idx: Dict[str, object]) -> "RankIndex":
        """From ``{tag: (ts, tier)}`` (what ``build_rank_index`` returns) or ``{tag: tier}``."""
        return cls(*_sorted_arrays(idx))

    @classmethod
    def empty(cls) -> "RankIndex":
        """A valid index with no entries — every lookup returns None.

        What the deployed API serves when the artifact can't be loaded, *instead* of rebuilding
        from the match dataset: that rebuild is ~200 MB at 3M tags, i.e. the OOM this artifact
        exists to prevent. Ranks reading as unknown for a refresh cycle beats taking the site down
        for everyone."""
        return cls(np.array([], dtype="S1"), np.array([], dtype=np.uint8))

    @classmethod
    def from_arrays(cls, tags_sorted, tiers) -> "RankIndex":
        """From already-sorted parallel lists/arrays (the artifact's on-disk form)."""
        return cls(_ascii_bytes(tags_sorted), np.asarray(tiers, dtype=np.uint8))

    def get(self, tag: str) -> Optional[int]:
        """The Ranked tier (1-22) for ``tag``, or None if it isn't in the index."""
        if self._tags.size == 0:
            return None
        key = tag.encode("ascii", "ignore")
        i = int(np.searchsorted(self._tags, key))
        if i < self._tags.size and self._tags[i] == key:
            return int(self._tiers[i])
        return None

    def __len__(self) -> int:
        return int(self._tags.size)


def index_payload(idx: Dict[str, Tuple[int, int]]) -> dict:
    """Serialize ``build_rank_index``'s ``{tag: (ts, tier)}`` to the compact artifact dict."""
    items = sorted((tag, int(v[1] if isinstance(v, tuple) else v)) for tag, v in idx.items())
    return {
        "version": FORMAT_VERSION,
        "tags": [t for t, _ in items],
        "tiers": [t for _, t in items],
    }


def _save_npz(idx: Dict[str, Tuple[int, int]], path: Path) -> Path:
    """Write the current ``.npz`` container: the serve arrays themselves, compressed.

    ``savez_compressed`` rather than ``savez`` because it is free: measured on the live 3.02M-tag
    index the peak is within ~5 MB either way (numpy fills a preallocated destination from small
    transients), while the asset is 13.7 MB instead of 33.2 MB — so the crawler's hourly re-upload
    and Render's re-download stay exactly where they are today (13.9 MB as gzipped JSON).
    """
    tags, tiers = _sorted_arrays(idx)
    if tags.dtype.itemsize > _MAX_TAG_BYTES:
        raise ValueError(
            f"tag of {tags.dtype.itemsize} bytes exceeds the {_MAX_TAG_BYTES}-byte publish bound; "
            f"'S' is fixed-width, so one oversized tag widens all {tags.size} rows")
    tmp = path.with_name(path.name + ".tmp")
    # Write through an OPEN FILE OBJECT, never a path: np.savez APPENDS ".npz" to any path that
    # doesn't already end in it, so np.savez("rank_index.npz.tmp") silently produces
    # "rank_index.npz.tmp.npz" and leaves the temp path we're about to rename empty. A file object
    # is written as-is. (Verified on numpy 2.4.6.)
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, version=np.array(NPZ_FORMAT_VERSION, dtype=np.int32),
                            tags=tags, tiers=tiers)
    tmp.replace(path)      # atomic: a reader never sees a half-written artifact
    return path


def save_rank_index(idx: Dict[str, Tuple[int, int]], path) -> Path:
    """Write the ``tag -> tier`` index to ``path``, in the container its suffix names: ``.npz``
    (current), gzipped JSON for ``.gz``, plain JSON otherwise.

    Suffix dispatch is fine *here* — the caller chooses the filename — but never on the read side,
    where the name comes from whatever the sync layer happened to save the download as."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npz":
        return _save_npz(idx, path)
    data = json.dumps(index_payload(idx), separators=(",", ":")).encode("utf-8")
    if path.suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=6) as fh:
            fh.write(data)
    else:
        path.write_bytes(data)
    return path


# How many bytes of the tags array to decode at a time. The chunk is the only part of the index
# that exists as Python strings at once, so this is the knob that trades load time against peak
# memory: 4 MB of JSON text is ~300k tags, a few MB of transient objects.
_TAG_CHUNK_BYTES = 4 << 20


def _element_spans(blob: bytes, start: int, end: int, approx: int):
    """Yield ``(a, b)`` offsets carving ``blob[start:end]`` — a JSON array body of strings — into
    slices that are each themselves a valid array body.

    Offsets rather than slices on purpose: the tags body is ~30 MB, and returning it (or the
    chunks) as new bytes objects would copy exactly the megabytes this loader exists to avoid.
    Cuts land on the 3-byte ``","`` separator, which cannot occur inside a Brawl Stars tag (ASCII
    alphanumerics; JSON would escape an embedded quote anyway). If no separator is found the
    remainder is yielded whole, so a surprise degrades to one big chunk rather than corrupt data.
    """
    pos = start
    while pos < end:
        if end - pos <= approx:
            yield pos, end
            return
        i = blob.find(b'","', pos + approx, end)
        if i == -1:
            yield pos, end
            return
        yield pos, i + 1        # keep this element's closing quote
        pos = i + 2             # resume at the next element's opening quote


def _array_span(blob: bytes, key: bytes):
    """``(start, end)`` of the body of ``"<key>":[ ... ]``, or None if not laid out that way."""
    at = blob.find(b'"' + key + b'":[')
    if at < 0:
        return None
    start = at + len(key) + 4
    end = blob.find(b"]", start)     # neither tags nor tiers contain a bracket
    return None if end < 0 else (start, end)


def _load_frugally(blob: bytes) -> Optional[RankIndex]:
    """Decode the artifact without ever holding the whole index as Python objects.

    ``json.loads`` on the full document is what makes this file dangerous on a 512 MB box: at
    2.5M tags it retains ~193 MB of Python strings, and :meth:`RankIndex.from_arrays` then peaks
    another ~156 MB converting them — ~350 MB transient for a 28 MB result, which is what
    OOM-killed the Render instance on 2026-08-20. Decoding the tags array in slices and folding
    each slice straight into a NumPy array keeps the peak near the result size.

    Returns None if the document isn't in the expected flat layout, so the caller can fall back
    to the plain parse rather than guess.
    """
    tags_span = _array_span(blob, b"tags")
    tiers_span = _array_span(blob, b"tiers")
    if tags_span is None or tiers_span is None:
        return None
    blocks = []
    for a, b in _element_spans(blob, tags_span[0], tags_span[1], _TAG_CHUNK_BYTES):
        if b <= a:
            continue
        # Round-trip through json for the chunk so escapes and unicode stay correct; only this
        # slice's strings are alive at once, and the list is dropped as soon as it is packed.
        blocks.append(_ascii_bytes(json.loads(b"[" + blob[a:b] + b"]")))
    if not blocks:
        tags = np.array([], dtype="S1")
    elif len(blocks) == 1:
        tags = blocks[0]
    else:
        # Concatenating different fixed widths would truncate, so widen every block first, and
        # release each one as it lands so both copies are never fully resident.
        width = max(blk.dtype.itemsize for blk in blocks)
        total = sum(blk.shape[0] for blk in blocks)
        tags = np.empty(total, dtype=f"S{width}")
        at = 0
        for i in range(len(blocks)):
            blk = blocks[i]
            tags[at:at + blk.shape[0]] = blk
            at += blk.shape[0]
            blocks[i] = None
        del blocks
    # Tiers are 1-22, so every int the parser produces is a cached CPython singleton — the list
    # is pointers, not objects, and is cheap enough to take in one bite.
    a, b = tiers_span
    tiers = np.array(json.loads(b"[" + blob[a:b] + b"]"), dtype=np.uint8)
    if tags.size != tiers.size:
        return None
    return RankIndex(tags, tiers)


def _load_npz(path: Path) -> RankIndex:
    """Load the ``.npz`` container, validating every assumption :meth:`RankIndex.get` relies on.

    Everything raises ``ValueError``, the one exception the caller already reacts to — numpy leaks
    ``BadZipFile`` / ``KeyError`` / ``OSError`` for a corrupt archive, which would otherwise escape
    as a 500 instead of degrading.
    """
    try:
        with np.load(path, allow_pickle=False) as z:      # `with` closes the fd; see below
            missing = {"version", "tags", "tiers"} - set(z.files)
            if missing:
                raise ValueError(f"rank index npz missing member(s): {sorted(missing)}")
            ver = int(z["version"])
            if ver != NPZ_FORMAT_VERSION:
                raise ValueError(f"unsupported rank index npz version {ver!r} "
                                 f"(expected {NPZ_FORMAT_VERSION})")
            # Materialize INSIDE the with-block — outside it the archive is closed and the lazy
            # member read fails. tiers first: it's 3 MB against tags' 30 MB, so a malformed archive
            # (including an object-dtype member, which allow_pickle=False rejects only on ACCESS,
            # not on np.load) fails cheap.
            tiers = z["tiers"]
            tags = z["tags"]
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 — normalize numpy/zipfile errors to the one the caller handles
        raise ValueError(f"unreadable rank index npz: {type(e).__name__}: {e}") from e

    if tags.ndim != 1 or tiers.ndim != 1:
        raise ValueError(f"rank index arrays must be 1-D (got {tags.ndim}-D tags, {tiers.ndim}-D tiers)")
    if tags.dtype.kind != "S":                 # 'U' would be 4x the bytes for the same content
        raise ValueError(f"rank index tags must be byte strings, got dtype {tags.dtype}")
    if tiers.dtype != np.uint8:                # int64 would be 8x, and every functional test passes
        raise ValueError(f"rank index tiers must be uint8, got dtype {tiers.dtype}")
    if tags.dtype.itemsize > _MAX_LOAD_TAG_BYTES:
        raise ValueError(f"rank index tag width {tags.dtype.itemsize} exceeds {_MAX_LOAD_TAG_BYTES} bytes")
    if tags.size != tiers.size:
        raise ValueError(f"rank index arrays misaligned: {tags.size} tags vs {tiers.size} tiers")
    if tags.size > _MAX_TAGS:
        raise ValueError(f"rank index declares {tags.size} tags (ceiling {_MAX_TAGS})")
    # Ordering is a correctness precondition, not a nicety: an unsorted array doesn't fail the
    # binary search, it returns the WRONG tier. Non-strict `<=` on purpose — a duplicate tag
    # resolves arbitrarily, which is a far smaller blast radius than rejecting the whole index.
    # Measured at 0.14 s / ~3 MB on the live 3.02M-tag index, so it always runs.
    if tags.size > 1 and not bool(np.all(tags[:-1] <= tags[1:])):
        raise ValueError("rank index tags are not ascending; lookups would return wrong tiers")
    # Construct directly: from_arrays would re-encode via _ascii_bytes's per-element Python loop —
    # 3M iterations, which is precisely the cost this format exists to delete.
    return RankIndex(tags, tiers)


def load_rank_index(path) -> RankIndex:
    """Load a :class:`RankIndex` from the published artifact: ``.npz``, or legacy (gzipped) JSON.

    Dispatch is on **magic bytes, not the filename** — ``data/sync.py`` writes whatever the URL
    served to one fixed local path, so during a URL migration the extension and the contents
    routinely disagree.

    The sniff also has to happen before ``np.load`` is ever reached: ``np.load`` treats any
    non-npy/npz blob as a raw pickle (on the legacy artifact it raises *"This file contains pickled
    (object) data"*), so routing downloaded bytes into it would leave untrusted input one
    ``allow_pickle`` flag away from code execution.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic == _NPZ_MAGIC:
        return _load_npz(path)
    raw = path.read_bytes()
    if raw[:2] == _GZIP_MAGIC:
        raw = gzip.decompress(raw)
    ver_at = raw.find(b'"version":')
    ver = None
    if ver_at >= 0:
        try:
            ver = int(raw[ver_at + 10:ver_at + 20].split(b",")[0].split(b"}")[0])
        except ValueError:
            ver = None
    if ver is None:                       # not laid out as expected — parse properly to find out
        ver = json.loads(raw).get("version")
    if ver != FORMAT_VERSION:  # format drift -> raise so the caller falls back to a fresh build
        raise ValueError(f"unsupported rank index format version {ver!r} (expected {FORMAT_VERSION})")
    frugal = _load_frugally(raw)
    if frugal is not None:
        return frugal
    payload = json.loads(raw)
    return RankIndex.from_arrays(payload["tags"], payload["tiers"])
