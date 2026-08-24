"""Build the player-rank index (tag -> Ranked tier) from ALL collected matches and write the
compact artifact the deployed API loads — instead of building the ~1.3M-entry dict (~200 MB,
~45 s) in memory from the full dataset, which threatens a small instance (Render's 512 MB free
tier) as the crawl grows.

    PYTHONPATH=backend python backend/scripts/export_rank_index.py

Run on a machine with the data + RAM to spare (the home crawler box). Publish it with
``python -m bsdraft.collect.publish --only-rank-npz`` (the crawler does both each cycle). The API
pulls it via ``RANK_INDEX_URL`` and loads the arrays straight back — measured on the live 3.0M-tag
index: 13.7 MB asset, ~0.3 s, ~66 MB peak RSS.
Output: data/processed/rank_index.npz.

The container follows the ``--out`` suffix, so ``--out data/processed/rank_index.json.gz`` still
writes the legacy gzipped JSON (~263 MB to load — the format this replaced).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from bsdraft.constants import PROCESSED_DIR
from bsdraft.engine.playerrank import build_rank_index
from bsdraft.engine.rank_store import save_rank_index

DEFAULT_OUT = PROCESSED_DIR / "rank_index.npz"


def export(out: Path = DEFAULT_OUT) -> Path:
    """Build the tag->tier index from all matches and save it to ``out``."""
    t = time.time()
    idx = build_rank_index()
    save_rank_index(idx, out)
    mb = out.stat().st_size / 1e6
    print(f"built rank index for {len(idx)} tags -> {out} ({mb:.2f} MB) in {time.time()-t:.1f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + save the precomputed rank index for the API to load.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output artifact (.npz, or legacy .json / .json.gz)")
    args = ap.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
