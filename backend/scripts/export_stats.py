"""Build current-balance-era empirical draft stats and write the compact artifact the deployed
API loads instead of replaying the full history at startup.

    PYTHONPATH=backend python backend/scripts/export_stats.py

Run on the home crawler box, which has the match data. Publish it with
``python -m bsdraft.collect.publish --only-stats`` (the crawler does both each cycle). The API
pulls it via ``STATS_URL`` and loads it in tens of MB. Output: data/processed/stats.json.gz.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from bsdraft.config import settings
from bsdraft.constants import PROCESSED_DIR
from bsdraft.data.balance_eras import current_balance_era
from bsdraft.engine.stats import build_bracketed
from bsdraft.engine.stats_store import save_stats

DEFAULT_OUT = PROCESSED_DIR / "stats.json.gz"


def export(out: Path = DEFAULT_OUT, max_matches: int = 0, all_eras: bool = False) -> Path:
    """Build current-era stats by default; ``all_eras`` is an explicit research override."""
    t = time.time()
    era = None if all_eras else current_balance_era()
    g, br = build_bracketed(
        halflife_days=settings.stats_halflife_days,
        max_matches=max_matches,
        analysis_start_ts=era.start_ts if era else 0,
        analysis_era_id=era.id if era else "",
    )
    save_stats(g, br, out)
    mb = out.stat().st_size / 1e6
    label = era.id if era else "all eras"
    print(f"built {label} stats from {g.n} matches ({len(br)} bracket(s)) -> {out} "
          f"({mb:.2f} MB) in {time.time()-t:.1f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + save precomputed stats for the API to load.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output artifact (.json or .json.gz)")
    ap.add_argument("--max-matches", type=int, default=0,
                    help="cap matches used within the selected era (0 = all)")
    ap.add_argument("--all-eras", action="store_true",
                    help="explicit research/backtest mode: include pre-balance matches")
    args = ap.parse_args()
    export(args.out, args.max_matches, args.all_eras)


if __name__ == "__main__":
    main()
