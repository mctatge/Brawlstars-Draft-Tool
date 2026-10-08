"""Consume temporal-test snapshots before training, including attempts that later fail.

The retrain workflow serializes writers with its concurrency group. A GitHub release body
PATCH reserves the new dataset watermark without deleting the previous ledger.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

LEDGER_TAG = "model-evaluation"


def _gh(*args):
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    return result


def reserve_snapshot(repository: str, *, through_ts: int, dataset_sha256: str,
                     incumbent_through_ts: int, source_commit: str, attempt_url: str = "", validate_floor=None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid ledger repository")
    if through_ts <= 0 or not re.fullmatch(r"[0-9a-f]{64}", dataset_sha256):
        raise ValueError("invalid evaluation snapshot")
    endpoint = f"repos/{repository}/releases/tags/{LEDGER_TAG}"
    current = _gh("api", endpoint)
    release_id, previous = None, 0
    if current.returncode == 0:
        release = json.loads(current.stdout)
        ledger = json.loads(release["body"])
        if ledger.get("schema_version") != 1 or not isinstance(ledger.get("consumed_through_ts"), int):
            raise ValueError("invalid evaluation ledger; refusing to reset it")
        release_id, previous = release["id"], ledger["consumed_through_ts"]
    elif "404" not in current.stderr:
        raise RuntimeError(f"evaluation ledger unavailable: {current.stderr.strip()}")
    floor = max(previous, incumbent_through_ts)
    if through_ts <= floor:
        raise ValueError("dataset snapshot was already consumed by an earlier evaluation attempt")
    if validate_floor is not None:
        validate_floor(floor)  # an undersized split must not consume newly collected rows
    reservation = {"schema_version": 1, "reservation_id": str(uuid.uuid4()),
                   "test_after_ts": floor, "consumed_through_ts": through_ts,
                   "dataset_sha256": dataset_sha256, "source_commit": source_commit,
                   "reserved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                   "attempt_url": attempt_url}
    with tempfile.TemporaryDirectory(prefix="bsdraft-evaluation-ledger-") as directory:
        body = Path(directory) / "reservation.json"
        body.write_text(json.dumps(reservation, allow_nan=False))
        if release_id is None:
            result = _gh("release", "create", LEDGER_TAG, "--target", source_commit,
                         "--title", "Consumed model evaluation snapshots", "--notes-file", str(body), "--latest=false")
        else:
            patch = Path(directory) / "patch.json"
            patch.write_text(json.dumps({"body": body.read_text()}))
            result = _gh("api", "--method", "PATCH", f"repos/{repository}/releases/{release_id}", "--input", str(patch))
        if result.returncode:
            raise RuntimeError(f"could not reserve evaluation snapshot: {result.stderr.strip()}")
    readback = _gh("api", endpoint)
    if readback.returncode or json.loads(json.loads(readback.stdout)["body"]) != reservation:
        raise RuntimeError("evaluation reservation readback failed; do not train on this snapshot")
    return reservation
