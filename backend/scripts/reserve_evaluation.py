"""Reserve a fresh dataset's final-test watermark before any training/evaluation."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from bsdraft.constants import REPO_ROOT
from bsdraft.data import dataset as D, encoders as E
from bsdraft.data.balance_eras import current_balance_era
from bsdraft.models.evaluation import load_incumbent, temporal_split
from bsdraft.models.evaluation_ledger import reserve_snapshot
from bsdraft.models.releases import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--incumbent-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-legacy-incumbent", action="store_true")
    args = parser.parse_args()
    _, evaluation, _ = load_incumbent(args.incumbent_npz, required=True,
                                      allow_legacy=args.allow_legacy_incumbent)
    era = current_balance_era()
    if era is None:
        raise SystemExit("evaluation reservation requires an active balance era")
    brawlers = E.brawler_encoder()
    newest = 0
    timestamps = []
    for match in D.iter_matches(args.matches):
        ts = int(match.get("ts") or 0)
        if ts < era.start_ts or match.get("a_won") is None or not E.encode_map(match.get("map_id")):
            continue
        if any(player["brawler_id"] not in brawlers for side in ("team_a", "team_b") for player in match[side]):
            continue
        newest = max(newest, ts)
        timestamps.append(ts)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    reservation = reserve_snapshot(args.repository, through_ts=newest, dataset_sha256=sha256_file(args.matches),
                                   incumbent_through_ts=int(evaluation.get("data_through_ts", 0)),
                                   source_commit=commit, attempt_url=os.environ.get("RUN_URL", ""),
                                   validate_floor=lambda floor: temporal_split(timestamps, previous_data_through_ts=floor))
    args.output.write_text(json.dumps(reservation, indent=2))
    print(f"reserved snapshot through {newest}; test rows must be newer than {reservation['test_after_ts']}")


if __name__ == "__main__":
    main()
