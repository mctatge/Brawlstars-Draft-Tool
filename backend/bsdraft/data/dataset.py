"""Build model-ready arrays from collected matches (data/raw/matches.jsonl).

Each labeled ranked match becomes: team_a/team_b brawler indices (3 each), a map index,
a mode index, the label (did team_a win), the timestamp (for recency weighting), and the
queue type (solo vs premade). Draws / unlabeled / unknown-brawler rows are dropped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from bsdraft.constants import RAW_DIR
from bsdraft.data import encoders as E


@dataclass
class Dataset:
    team_a: np.ndarray     # (N, 3) int64 brawler index
    team_b: np.ndarray     # (N, 3) int64
    map_idx: np.ndarray    # (N,)   int64
    mode_idx: np.ndarray   # (N,)   int64
    y: np.ndarray          # (N,)   float32  (1 if team_a won)
    ts: np.ndarray         # (N,)   int64    epoch seconds
    queue_type: np.ndarray  # (N,)  object

    def __len__(self) -> int:
        return int(self.y.shape[0])


def iter_matches(path: Optional[Path] = None, contains: Optional[str] = None) -> Iterator[dict]:
    """Yield each stored match as a dict. ``contains`` is an optional raw-substring prefilter:
    lines that don't contain it are skipped *before* ``json.loads``. A normalized player tag is
    serialized verbatim into the line (see :mod:`bsdraft.collect.match`), so every match a tag
    appears in must contain it as a substring — passing ``contains=tag`` lets a per-player scan
    skip parsing the ~99.9% of lines the tag never appears in (seconds saved on the full dataset).
    The substring test admits rare false positives, so callers still apply an exact check."""
    path = path or (RAW_DIR / "matches.jsonl")
    if not Path(path).exists():
        return  # no data yet (e.g. cloud cold start before the first sync) -> empty iterator
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if contains is not None and contains not in line:
                continue
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def recent_matches(n: int, path: Optional[Path] = None) -> list:
    """The ``n`` most-recent matches (by ``ts``), loaded with bounded memory via a size-``n``
    min-heap — so the stats build's peak RAM stays flat as the dataset grows past what a small
    instance (e.g. Render's 512 MB free tier) can hold. ``n <= 0`` loads everything."""
    if not n or n <= 0:
        return list(iter_matches(path))
    import heapq
    from itertools import count
    tie = count()                       # unique tiebreak so dicts are never compared
    heap: list = []                     # (ts, tiebreak, match) — keeps the n largest ts
    for r in iter_matches(path):
        item = (int(r.get("ts") or 0), next(tie), r)
        if len(heap) < n:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    heap.sort()
    return [r for _, _, r in heap]


def count_matches(path: Optional[Path] = None) -> int:
    """Total number of stored matches — a fast newline count over matches.jsonl that never parses
    JSON, so it stays cheap on the full multi-GB dataset (peak RAM = one 1 MB chunk). Each match is
    serialized as exactly one line (see :mod:`bsdraft.collect.match`), so the newline count is the
    match count. Returns 0 when there's no data yet (e.g. a cloud cold start before the first sync)."""
    path = path or (RAW_DIR / "matches.jsonl")
    if not Path(path).exists():
        return 0
    total = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            total += chunk.count(b"\n")
    return total


def build_dataset(path: Optional[Path] = None, ranked_maps_only: bool = True) -> Dataset:
    bidx = E.brawler_encoder()
    a_rows, b_rows, maps, modes, ys, tss, qs = [], [], [], [], [], [], []
    for r in iter_matches(path):
        if r.get("a_won") is None:
            continue
        a_ids = [p["brawler_id"] for p in r["team_a"]]
        b_ids = [p["brawler_id"] for p in r["team_b"]]
        if any(b not in bidx for b in a_ids + b_ids):
            continue
        map_idx = E.encode_map(r.get("map_id"))
        if ranked_maps_only and map_idx == 0:
            continue
        a_rows.append([bidx[b] for b in a_ids])
        b_rows.append([bidx[b] for b in b_ids])
        maps.append(map_idx)
        modes.append(E.encode_mode(r.get("mode", "")))
        ys.append(1.0 if r["a_won"] else 0.0)
        tss.append(int(r.get("ts", 0)))
        qs.append(r.get("queue_type", ""))

    return Dataset(
        team_a=np.array(a_rows, dtype=np.int64).reshape(-1, 3),
        team_b=np.array(b_rows, dtype=np.int64).reshape(-1, 3),
        map_idx=np.array(maps, dtype=np.int64),
        mode_idx=np.array(modes, dtype=np.int64),
        y=np.array(ys, dtype=np.float32),
        ts=np.array(tss, dtype=np.int64),
        queue_type=np.array(qs, dtype=object),
    )


def summary(ds: Dataset) -> str:
    n = len(ds)
    if n == 0:
        return "samples=0 (no labeled matches yet)"
    solo = int((ds.queue_type == "soloRanked").sum())
    team = int((ds.queue_type == "teamRanked").sum())
    return (
        f"samples={n}  team_a win-rate={ds.y.mean():.3f}  "
        f"distinct maps={len(set(ds.map_idx.tolist()))}  "
        f"modes={len(set(ds.mode_idx.tolist()))}  "
        f"solo={solo} team={team}"
    )


if __name__ == "__main__":
    print(summary(build_dataset()))
