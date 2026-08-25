"""The rank-index artifact's load path, which has to fit a 2.5M-entry index into a 512 MB box.

`json.loads` on the whole document retained ~193 MB of Python strings and `from_arrays` peaked
another ~156 MB converting them — ~350 MB transient for a 28 MB result. That OOM-killed the
Render instance on 2026-08-20 every time anyone looked up a rank, taking the whole API with it.
The loader now decodes the tags array in slices; these tests pin that the frugal path is exactly
equivalent to the plain one, including at the chunk seams where it could silently truncate.
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pytest

from bsdraft.engine import rank_store as RS


def _index(n: int, width=lambda i: 8) -> dict:
    """`{tag: (ts, tier)}` as build_rank_index returns it, with controllable tag widths."""
    return {f"{i:0{width(i)}X}"[-width(i):]: (1000 + i, (i % 22) + 1) for i in range(n)}


def _roundtrip(tmp_path, idx, name="rank_index.json.gz"):
    path = RS.save_rank_index(idx, tmp_path / name)
    return RS.load_rank_index(path)


def _plain_load(path) -> RS.RankIndex:
    """The whole-document parse the frugal path replaced — the reference implementation."""
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    payload = json.loads(raw)
    return RS.RankIndex.from_arrays(payload["tags"], payload["tiers"])


def _same(a: RS.RankIndex, b: RS.RankIndex) -> bool:
    width = max(a._tags.dtype.itemsize, b._tags.dtype.itemsize)
    return (np.array_equal(a._tags.astype(f"S{width}"), b._tags.astype(f"S{width}"))
            and np.array_equal(a._tiers, b._tiers))


def test_roundtrip_preserves_every_lookup(tmp_path):
    idx = _index(500)
    loaded = _roundtrip(tmp_path, idx)
    assert len(loaded) == 500
    for tag, (_, tier) in idx.items():
        assert loaded.get(tag) == tier
    assert loaded.get("NOTATAG") is None


def test_frugal_and_plain_paths_agree(tmp_path, monkeypatch):
    # A chunk size well below the payload forces many seams; the plain parse is the oracle.
    monkeypatch.setattr(RS, "_TAG_CHUNK_BYTES", 64)
    path = RS.save_rank_index(_index(2000), tmp_path / "r.json.gz")
    assert _same(RS.load_rank_index(path), _plain_load(path))


def test_seams_do_not_drop_or_duplicate_entries(tmp_path, monkeypatch):
    # The cut lands on the b'","' separator; an off-by-one there would eat a tag's quote and
    # silently shift every tier after it against the wrong tag.
    monkeypatch.setattr(RS, "_TAG_CHUNK_BYTES", 16)
    idx = _index(777)
    loaded = _roundtrip(tmp_path, idx)
    assert len(loaded) == len(idx)
    for tag, (_, tier) in idx.items():
        assert loaded.get(tag) == tier, f"{tag} came back wrong across a chunk seam"


def test_varying_tag_widths_are_not_truncated(tmp_path, monkeypatch):
    # Blocks get different fixed 'S' widths; concatenating without widening first would clip the
    # long tags down to the first block's itemsize.
    monkeypatch.setattr(RS, "_TAG_CHUNK_BYTES", 32)
    idx = {}
    for i in range(300):
        tag = ("T" * (2 + i % 12)) + f"{i:04d}"      # 6..17 chars
        idx[tag] = (1, (i % 22) + 1)
    loaded = _roundtrip(tmp_path, idx)
    for tag, (_, tier) in idx.items():
        assert loaded.get(tag) == tier


def test_single_chunk_path(tmp_path, monkeypatch):
    monkeypatch.setattr(RS, "_TAG_CHUNK_BYTES", 1 << 20)   # everything in one bite
    idx = _index(50)
    loaded = _roundtrip(tmp_path, idx)
    assert len(loaded) == 50 and loaded.get(next(iter(idx))) is not None


def test_empty_index(tmp_path):
    loaded = _roundtrip(tmp_path, {})
    assert len(loaded) == 0 and loaded.get("ANY") is None


def test_uncompressed_artifact_still_loads(tmp_path):
    idx = _index(40)
    loaded = _roundtrip(tmp_path, idx, name="rank_index.json")
    assert len(loaded) == 40


def test_version_mismatch_raises_so_the_caller_rebuilds(tmp_path):
    path = tmp_path / "r.json.gz"
    payload = {"version": RS.FORMAT_VERSION + 1, "tags": ["AA"], "tiers": [3]}
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="unsupported rank index format version"):
        RS.load_rank_index(path)


def test_unexpected_layout_falls_back_instead_of_guessing(tmp_path):
    # Keys reordered and spaced — the offset scan won't find its markers. The loader must fall
    # back to the plain parse rather than return a half-decoded index.
    path = tmp_path / "r.json.gz"
    payload = {"tiers": [5, 9], "tags": ["AAA", "BBB"], "version": RS.FORMAT_VERSION}
    path.write_bytes(gzip.compress(json.dumps(payload, indent=2).encode()))
    loaded = RS.load_rank_index(path)
    assert loaded.get("AAA") == 5 and loaded.get("BBB") == 9


def test_mismatched_array_lengths_fall_back(tmp_path, monkeypatch):
    # A truncated tiers array must not produce an index whose tags and tiers are misaligned.
    path = tmp_path / "r.json.gz"
    payload = {"version": RS.FORMAT_VERSION, "tags": ["AAA", "BBB"], "tiers": [5]}
    path.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode()))
    assert RS._load_frugally(gzip.decompress(path.read_bytes())) is None


def test_lookup_is_exact_not_prefix(tmp_path):
    # Binary search over fixed-width byte strings could match a prefix if compared loosely.
    idx = {"ABC": (1, 4), "ABCDEF": (1, 9)}
    loaded = _roundtrip(tmp_path, idx)
    assert loaded.get("ABC") == 4 and loaded.get("ABCDEF") == 9
    assert loaded.get("ABCD") is None and loaded.get("AB") is None


# ---------------------------------------------------------------------------
# The .npz container (current format).
#
# Loading the gzipped-JSON form peaked at 263 MB RSS on the live 3.02M-tag index — against ~195 MB
# resident on a 512 MB box, that is the OOM. The .npz form stores the serve arrays themselves:
# same 13.7 MB asset, ~66 MB peak, ~0.3 s. These tests pin the guarantees that swap depends on:
# it decodes to a byte-identical index, it never reaches numpy's pickle path, and anything
# malformed raises ValueError so the API degrades instead of rebuilding (which is the OOM again).
# ---------------------------------------------------------------------------

def _raw_npz(tmp_path, name="r.npz", *, version=RS.NPZ_FORMAT_VERSION, tags=None, tiers=None, omit=()):
    """Write an npz archive member-by-member, so a test can express a MALFORMED artifact that
    ``save_rank_index`` would never produce."""
    members = {}
    if "version" not in omit:
        members["version"] = np.array(version, dtype=np.int32)
    if "tags" not in omit:
        members["tags"] = np.array([b"AA", b"BB"], dtype="S2") if tags is None else tags
    if "tiers" not in omit:
        members["tiers"] = np.array([3, 7], dtype=np.uint8) if tiers is None else tiers
    path = tmp_path / name
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **members)
    return path


# --- A. allow_pickle safety: the artifact is downloaded from a public URL -------------------

def test_object_array_member_is_rejected_by_the_public_loader(tmp_path):
    # np.load itself SUCCEEDS on this and even lists the members; allow_pickle=False only bites on
    # member ACCESS. So the assertion has to be on load_rank_index, not on np.load.
    path = _raw_npz(tmp_path, tags=np.array([{"evil": 1}], dtype=object))
    with pytest.raises(ValueError):
        RS.load_rank_index(path)


def test_loader_never_enables_allow_pickle(tmp_path, monkeypatch):
    # A refactor fence: someone adding mmap_mode or copying a np.load call must not drop this.
    seen = {}
    real = RS.np.load

    def spy(*a, **kw):
        seen["allow_pickle"] = kw.get("allow_pickle", "MISSING")
        return real(*a, **kw)

    monkeypatch.setattr(RS.np, "load", spy)
    RS.load_rank_index(RS.save_rank_index(_index(20), tmp_path / "r.npz"))
    assert seen["allow_pickle"] is False


def test_legacy_json_never_reaches_np_load(tmp_path, monkeypatch):
    # np.load treats a non-npy/npz blob as a raw PICKLE — on the real legacy artifact it raises
    # "This file contains pickled (object) data". The magic-byte sniff must run first so those
    # downloaded bytes are never handed to it at all.
    monkeypatch.setattr(RS.np, "load", lambda *a, **kw: pytest.fail("np.load reached for JSON"))
    loaded = _roundtrip(tmp_path, _index(30), name="legacy.json.gz")
    assert len(loaded) == 30


# --- B. dispatch on CONTENT, not filename ---------------------------------------------------
# sync.py writes whatever the URL served to one fixed local path, so during the URL migration the
# extension and the contents routinely disagree. Both directions must work.

def test_npz_content_loads_from_a_json_named_path(tmp_path):
    src = RS.save_rank_index(_index(50), tmp_path / "r.npz")
    misnamed = tmp_path / "rank_index.json"
    misnamed.write_bytes(src.read_bytes())
    assert len(RS.load_rank_index(misnamed)) == 50


def test_gzipped_json_content_loads_from_an_npz_named_path(tmp_path):
    src = RS.save_rank_index(_index(50), tmp_path / "r.json.gz")
    misnamed = tmp_path / "rank_index.npz"
    misnamed.write_bytes(src.read_bytes())
    assert len(RS.load_rank_index(misnamed)) == 50


def test_unrecognized_container_raises_rather_than_guessing(tmp_path):
    path = tmp_path / "r.npz"
    path.write_bytes(b"\x00\x01not-an-artifact")
    with pytest.raises(ValueError):
        RS.load_rank_index(path)


# --- C. cross-format equivalence: the rollout safety net -------------------------------------

def test_npz_and_json_produce_identical_indexes(tmp_path):
    # The single strongest guard that swapping the published format changes nothing observable.
    idx = _index(2000, width=lambda i: 6 + (i % 5))
    a = RS.load_rank_index(RS.save_rank_index(idx, tmp_path / "r.json.gz"))
    b = RS.load_rank_index(RS.save_rank_index(idx, tmp_path / "r.npz"))
    assert _same(a, b)
    assert _same(b, _plain_load(tmp_path / "r.json.gz"))    # ...and to the pre-existing oracle
    for tag, (_, tier) in idx.items():
        assert b.get(tag) == tier


# --- D. validation chain: every failure is a ValueError the API already degrades on ----------

@pytest.mark.parametrize("kwargs, why", [
    ({"omit": ("version",)}, "missing version member"),
    ({"omit": ("tiers",)}, "missing tiers member"),
    ({"version": RS.NPZ_FORMAT_VERSION + 1}, "future format version"),
    ({"tiers": np.array([3, 7], dtype=np.int64)}, "int64 tiers = 8x the memory"),
    ({"tags": np.array(["AA", "BB"], dtype="U2")}, "unicode tags = 4x the memory"),
    ({"tags": np.array([b"AA", b"BB", b"CC"], dtype="S2")}, "tags/tiers length mismatch"),
    ({"tags": np.array([b"BB", b"AA"], dtype="S2")}, "descending tags"),
    ({"tags": np.array([[b"AA"], [b"BB"]], dtype="S2")}, "2-D tags"),
    ({"tags": np.array([b"A" * 40, b"B" * 40], dtype="S40")}, "tag width over the load ceiling"),
])
def test_malformed_npz_raises_valueerror(tmp_path, kwargs, why):
    with pytest.raises(ValueError):
        RS.load_rank_index(_raw_npz(tmp_path, **kwargs))


def test_truncated_archive_is_normalized_to_valueerror(tmp_path):
    # Raw numpy raises zipfile.BadZipFile here; the API only catches ValueError to degrade.
    path = RS.save_rank_index(_index(500), tmp_path / "r.npz")
    path.write_bytes(path.read_bytes()[: 1024])
    with pytest.raises(ValueError):
        RS.load_rank_index(path)


# --- E. sortedness: a silent-wrong-answer class the JSON format hid --------------------------

def test_writer_sorts_by_ENCODED_bytes_not_python_strings(tmp_path):
    # _ascii_bytes drops non-ASCII, so sorted(["AB","Aé"]) encodes to [b'AB', b'A'] — DESCENDING.
    # Sorting the strings first (the old index_payload order) puts the artifact out of order and
    # every lookup past that point returns the wrong tier.
    idx = {"AB": (1, 5), "Aé": (1, 9), "AA": (1, 2)}
    loaded = RS.load_rank_index(RS.save_rank_index(idx, tmp_path / "r.npz"))
    tags = loaded._tags
    assert list(tags) == sorted(tags), "artifact tags are not ascending by encoded bytes"


def test_unsorted_tags_are_rejected_not_silently_mis_searched(tmp_path):
    # RankIndex.get would return a WRONG tier here rather than raise, so the loader must catch it.
    with pytest.raises(ValueError, match="ascending"):
        RS.load_rank_index(_raw_npz(tmp_path, tags=np.array([b"ZZ", b"AA"], dtype="S2")))


def test_duplicate_tags_degrade_to_one_entry_rather_than_emptying_the_index(tmp_path):
    # The check is non-strict (<=) on purpose: a duplicate resolves one tag arbitrarily, while a
    # strict (<) check would reject the artifact and take the whole rank feature down.
    loaded = RS.load_rank_index(_raw_npz(tmp_path, tags=np.array([b"AA", b"AA"], dtype="S2")))
    assert loaded.get("AA") in (3, 7)


# --- F. dtype / width preservation -----------------------------------------------------------

def test_dtypes_and_width_survive_the_roundtrip(tmp_path):
    idx = {}
    for i in range(300):
        idx[("T" * (2 + i % 10)) + f"{i:04d}"] = (1, (i % 22) + 1)   # 6..15 chars, under the bound
    loaded = RS.load_rank_index(RS.save_rank_index(idx, tmp_path / "r.npz"))
    assert loaded._tiers.dtype == np.uint8 and loaded._tags.dtype.kind == "S"
    assert loaded._tags.dtype.itemsize == max(len(t) for t in idx)
    for tag, (_, tier) in idx.items():
        assert loaded.get(tag) == tier


def test_writer_raises_on_an_oversized_tag(tmp_path):
    # 'S' is fixed width: one 40-char tag widens all rows (30 MB -> 121 MB at 3M) and every
    # functional test would still pass.
    with pytest.raises(ValueError, match="exceeds"):
        RS.save_rank_index({"A" * 40: (1, 5)}, tmp_path / "r.npz")


@pytest.mark.parametrize("tier", [0, 23, -1, 300])
def test_writer_raises_on_an_out_of_range_tier(tmp_path, tier):
    # Pin OUR behavior, not numpy's: the uint8 cast wraps on some versions and raises on others.
    with pytest.raises(ValueError, match="1-22"):
        RS.save_rank_index({"AAA": (1, tier)}, tmp_path / "r.npz")


# --- H. resource hygiene: this loader re-runs on a timer --------------------------------------

def test_repeated_loads_do_not_leak_file_descriptors(tmp_path):
    # np.load returns an open archive handle. Leaking one per refresh would also pin the inode
    # that sync.py's tmp.replace() swapped out, so the API would keep reading the stale artifact.
    path = RS.save_rank_index(_index(200), tmp_path / "r.npz")
    RS.load_rank_index(path)
    before = len(os.listdir("/dev/fd"))
    for _ in range(40):
        RS.load_rank_index(path)
    assert len(os.listdir("/dev/fd")) <= before + 2


def test_loaded_index_owns_its_memory(tmp_path):
    # Not a mmap: sync.py replaces this file underneath a live index every refresh.
    path = RS.save_rank_index(_index(100), tmp_path / "r.npz")
    loaded = RS.load_rank_index(path)
    path.unlink()
    assert len(loaded) == 100 and loaded.get(next(iter(_index(100)))) is not None


# --- I. empty index ---------------------------------------------------------------------------

def test_empty_npz_roundtrips(tmp_path):
    loaded = RS.load_rank_index(RS.save_rank_index({}, tmp_path / "r.npz"))
    assert len(loaded) == 0 and loaded.get("ANY") is None
    assert loaded._tags.dtype.kind == "S"      # a bare np.array([]) would be float64


def test_RankIndex_empty_is_a_working_index():
    idx = RS.RankIndex.empty()
    assert len(idx) == 0 and idx.get("ANY") is None      # must not IndexError


def test_over_length_lookup_does_not_copy_the_whole_array(tmp_path):
    """An over-wide key must be answered without reaching searchsorted.

    `tag` comes straight from the unauthenticated /api/rank query string, and numpy promotes BOTH
    operands to the wider dtype — so a long tag copies every row at the key's width. Measured at
    3.03M tags a 200-byte tag allocates ~495 MB, which OOM-kills the 512 MB instance. The guard is
    exact, not a heuristic: an 'S10' array cannot hold an 11-byte value, so no match is possible.
    """
    idx = RS.load_rank_index(RS.save_rank_index(_index(500), tmp_path / "r.npz"))
    width = idx._tags.dtype.itemsize
    calls = []
    real = RS.np.searchsorted

    def spy(a, v, *args, **kw):
        calls.append(len(v))
        return real(a, v, *args, **kw)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(RS.np, "searchsorted", spy)
    try:
        assert idx.get("A" * (width + 1)) is None
        assert idx.get("A" * 200) is None
        assert calls == [], "an over-wide key reached searchsorted and promoted the whole array"
        # ...and a legitimately-sized key still searches and still resolves.
        tag, (_, tier) = next(iter(_index(500).items()))
        assert idx.get(tag) == tier and calls, "in-width lookups must still use searchsorted"
    finally:
        monkey.undo()


# ---------------------------------------------------------------------------
# The API's fallback. This is the half that actually removes the OOM: an unreadable artifact used
# to fall back to RankIndex.from_mapping(build_rank_index()) -- a ~200 MB transient at 3M tags,
# which OOM-kills the 512 MB instance and takes the SITE down, not just ranks. On a host with an
# artifact URL configured, that rebuild must now be unreachable.
# ---------------------------------------------------------------------------

@pytest.fixture
def api(monkeypatch):
    """``api.main`` with a cleared rank cache, and the in-memory rebuild wired to fail the test."""
    import bsdraft.api.main as M
    monkeypatch.setattr(M, "_rank_idx_cache", None, raising=False)
    return M


def _forbid_rebuild(api, monkeypatch):
    monkeypatch.setattr(api, "build_rank_index", lambda *a, **k: pytest.fail(
        "fell back to building from matches — that is the ~200 MB OOM path"))


def test_deployed_host_degrades_to_empty_when_the_artifact_is_missing(api, monkeypatch, tmp_path):
    monkeypatch.setattr(api.settings, "rank_index_url", "https://example.invalid/rank_index.npz")
    monkeypatch.setattr(api.sync, "RANK_INDEX_PATH", tmp_path / "absent.npz")
    _forbid_rebuild(api, monkeypatch)
    idx = api._rank_index()
    assert len(idx) == 0 and idx.get("ANY") is None


def test_deployed_host_degrades_to_empty_when_the_artifact_is_corrupt(api, monkeypatch, tmp_path):
    bad = tmp_path / "rank_index.npz"
    bad.write_bytes(b"PK\x03\x04truncated-garbage")     # PK magic, so it routes to the npz loader
    monkeypatch.setattr(api.settings, "rank_index_url", "https://example.invalid/rank_index.npz")
    monkeypatch.setattr(api.sync, "RANK_INDEX_PATH", bad)
    _forbid_rebuild(api, monkeypatch)
    assert len(api._rank_index()) == 0


def test_deployed_host_loads_a_valid_artifact(api, monkeypatch, tmp_path):
    path = RS.save_rank_index({"AAA": (1, 13)}, tmp_path / "rank_index.npz")
    monkeypatch.setattr(api.settings, "rank_index_url", "https://example.invalid/rank_index.npz")
    monkeypatch.setattr(api.sync, "RANK_INDEX_PATH", path)
    _forbid_rebuild(api, monkeypatch)
    assert api._rank_index().get("AAA") == 13


def test_host_without_an_artifact_url_still_builds_in_memory(api, monkeypatch):
    # The home machine and local dev have the RAM and no artifact — they must keep working.
    monkeypatch.setattr(api.settings, "rank_index_url", "")
    monkeypatch.setattr(api, "build_rank_index", lambda *a, **k: {"AAA": (1, 7)})
    assert api._rank_index().get("AAA") == 7


# ---------------------------------------------------------------------------
# Plumbing consistency. The publisher, the sync destination, the exporter default, and the
# RANK_INDEX_URL in render.yaml each name the artifact independently; nothing functional ties
# them together, so a rename that misses one silently routes production onto the degraded
# empty-index path while every unit test above stays green. Pin them to each other.
# ---------------------------------------------------------------------------

def test_rank_index_artifact_name_agrees_everywhere():
    import re
    from pathlib import Path
    from bsdraft.collect import publish as P
    from bsdraft.data import sync as SY
    # scripts/ isn't importable as a package; read DEFAULT_OUT from the source instead.
    repo = Path(__file__).resolve().parents[2]
    export_src = (repo / "backend" / "scripts" / "export_rank_index.py").read_text()
    m = re.search(r'DEFAULT_OUT\s*=\s*PROCESSED_DIR\s*/\s*"([^"]+)"', export_src)
    assert m, "couldn't find DEFAULT_OUT in export_rank_index.py"
    exporter_name = m.group(1)

    # render.yaml has no pyyaml in requirements — regex the URL out.
    render = (repo / "render.yaml").read_text()
    m = re.search(r"RANK_INDEX_URL\s*\n\s*value:\s*(\S+)", render)
    assert m, "couldn't find RANK_INDEX_URL in render.yaml"
    url_name = m.group(1).rsplit("/", 1)[-1]

    assert (P.RANK_INDEX_NPZ_PATH.name == SY.RANK_INDEX_PATH.name == exporter_name == url_name), (
        f"artifact name split-brain: publisher={P.RANK_INDEX_NPZ_PATH.name} "
        f"sync={SY.RANK_INDEX_PATH.name} exporter={exporter_name} url={url_name}")


def test_npz_streams_through_sync_gzip_sniff_untouched(tmp_path, monkeypatch):
    # _sync_file gunzips gzip-framed downloads on the fly. An npz is PK-framed, so it must land
    # byte-identical — a refactor that decompresses unconditionally would corrupt it.
    from bsdraft.data import sync as SY
    src = RS.save_rank_index(_index(300), tmp_path / "rank_index.npz")
    blob = src.read_bytes()

    class _Resp:
        status_code = 200
        headers = {}
        def __init__(self, content): self._c = content
        def iter_bytes(self, chunk_size=65536):
            for i in range(0, len(self._c), chunk_size):
                yield self._c[i:i + chunk_size]
        def raise_for_status(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stream(self, method, url, **kw): return _Resp(blob)

    monkeypatch.setattr(SY.httpx, "Client", _Client)
    dest = tmp_path / "synced" / "rank_index.npz"
    monkeypatch.setattr(SY, "RANK_INDEX_PATH", dest)
    monkeypatch.setattr(SY, "_RANK_ETAG_PATH", tmp_path / "synced" / ".etag")
    monkeypatch.setattr(SY, "_RANK_SHA_PATH", tmp_path / "synced" / ".sha")
    assert SY.sync_rank_index("https://example.invalid/rank_index.npz") is True
    assert dest.read_bytes() == blob, "sync mutated the npz bytes"
    assert len(RS.load_rank_index(dest)) == 300
