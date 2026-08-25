"""Serialize built ``DraftStats`` to a compact artifact and load it back.

Lets the deployed API **load** precomputed empirical stats instead of **rebuilding** them in
memory from the full match dataset — which OOMs a small instance (Render's 512 MB free tier)
once the crawler grows the data past ~110k matches. The home machine (with RAM to spare) builds
the full stats and publishes ``stats.json.gz``; the API syncs and loads it in tens of MB, with
the match data never resident. Mirrors the model's ``winprob.npz`` publish/load split.

Format: gzipped JSON. The ``DraftStats`` tables are sparse dicts keyed by ints / int-pairs /
brawler-pairs, which JSON's string keys fit more naturally than a numeric array archive.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from bsdraft.engine.stats import DraftStats

FORMAT_VERSION = 1
_DICTS_1 = ("b_games", "b_wins", "map_games")                # int key
# Int-keyed tables absent from artifacts published before they existed: the loader defaults
# them empty instead of raising, so an old artifact stays loadable (empty just means "no
# recent-window signal" and consumers fall back). Same lesson as the singleton-syn-key note
# below — one missing/odd key must never poison the whole load.
_DICTS_1_OPT = ("map_games_recent",)                         # int key, added 2026-08-25
_DICTS_2 = ("bm_games", "bm_wins", "cnt_games", "cnt_wins")  # (int, int) key  -> "a_b"
_DICTS_S = ("syn_games", "syn_wins")                         # frozenset{int,int} -> "a_b" (sorted)


def _pair(k: str) -> Tuple[int, int]:
    a, b = k.split("_")
    return int(a), int(b)


def _table_to_dict(s: DraftStats) -> dict:
    out: dict = {"n": s.n, "halflife_days": s.halflife_days, "bracket": s.bracket}
    for name in _DICTS_1 + _DICTS_1_OPT:
        out[name] = {str(k): round(v, 6) for k, v in getattr(s, name).items()}
    for name in _DICTS_2:
        out[name] = {f"{a}_{b}": round(v, 6) for (a, b), v in getattr(s, name).items()}
    for name in _DICTS_S:
        out[name] = {"_".join(map(str, sorted(fs))): round(v, 6) for fs, v in getattr(s, name).items()}
    return out


def _dict_to_table(d: dict, fallback=None) -> DraftStats:
    # Build a blank DraftStats (empty matches -> no build work), then fill its tables.
    s = DraftStats(matches=[], halflife_days=d["halflife_days"], bracket=d.get("bracket"), fallback=fallback)
    s.n = d["n"]
    for name in _DICTS_1:
        setattr(s, name, defaultdict(float, {int(k): v for k, v in d[name].items()}))
    for name in _DICTS_1_OPT:
        setattr(s, name, defaultdict(float, {int(k): v for k, v in (d.get(name) or {}).items()}))
    for name in _DICTS_2:
        setattr(s, name, defaultdict(float, {_pair(k): v for k, v in d[name].items()}))
    for name in _DICTS_S:
        # A team that fielded the same brawler twice (a rare battle-log glitch) collapses its
        # pair frozenset to a single element, which serializes without a "_". Parse however many
        # ids are present — exactly what the in-memory build holds — instead of demanding two:
        # one such key once made every artifact load raise, silently dropping the API to the
        # capped 60k rebuild for weeks.
        setattr(s, name, defaultdict(float, {frozenset(int(x) for x in k.split("_")): v
                                             for k, v in d[name].items()}))
    return s


def stats_payload(global_stats: DraftStats, brackets: Dict[str, DraftStats]) -> dict:
    return {
        "version": FORMAT_VERSION,
        "global": _table_to_dict(global_stats),
        "brackets": {name: _table_to_dict(s) for name, s in brackets.items()},
    }


def load_payload(payload: dict) -> Tuple[DraftStats, Dict[str, DraftStats]]:
    g = _dict_to_table(payload["global"])
    br = {name: _dict_to_table(d, fallback=g) for name, d in payload.get("brackets", {}).items()}
    return g, br


def save_stats(global_stats: DraftStats, brackets: Dict[str, DraftStats], path) -> Path:
    """Write the global + per-bracket tables to ``path`` (gzipped JSON if it ends in ``.gz``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(stats_payload(global_stats, brackets), separators=(",", ":")).encode("utf-8")
    if path.suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=6) as fh:
            fh.write(data)
    else:
        path.write_bytes(data)
    return path


def load_stats(path) -> Tuple[DraftStats, Dict[str, DraftStats]]:
    """Load ``(global_stats, {bracket: stats})`` from a (optionally gzipped) JSON artifact."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return load_payload(json.loads(raw))
