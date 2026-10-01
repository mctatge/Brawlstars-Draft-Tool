"""Reviewable balance-era boundaries for live analysis artifacts.

The raw match archive is intentionally retained for drift detection and research.  Live
recommendation tables and model training, however, must opt into the active balance era so a
balance change cannot inherit thin map cells from the previous meta.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bsdraft.constants import REPO_ROOT

MANIFEST_PATH = REPO_ROOT / "data" / "reference" / "balance_eras.json"


@dataclass(frozen=True)
class BalanceEra:
    id: str
    label: str
    start_ts: int
    precision: str
    source_url: str = ""
    note: str = ""


def _timestamp(value: str) -> int:
    raw = str(value).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp())


def load_balance_eras(path: Optional[Path] = None) -> list[BalanceEra]:
    """Load and validate the ordered balance-era manifest."""
    manifest = json.loads((path or MANIFEST_PATH).read_text(encoding="utf-8"))
    eras: list[BalanceEra] = []
    previous = -1
    seen: set[str] = set()
    for raw in manifest.get("eras", []):
        era_id = str(raw["id"])
        if era_id in seen:
            raise ValueError(f"duplicate balance era: {era_id}")
        start_ts = _timestamp(raw["effective_from"])
        if start_ts <= previous:
            raise ValueError("balance eras must be strictly chronological")
        seen.add(era_id)
        eras.append(BalanceEra(
            id=era_id,
            label=str(raw.get("label") or era_id),
            start_ts=start_ts,
            precision=str(raw.get("precision") or "unknown"),
            source_url=str(raw.get("source_url") or ""),
            note=str(raw.get("note") or ""),
        ))
        previous = start_ts
    return eras


def current_balance_era(path: Optional[Path] = None) -> Optional[BalanceEra]:
    """Return the manifest-selected active era, or ``None`` for an unconfigured project."""
    manifest_path = path or MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eras = load_balance_eras(manifest_path)
    current_id = manifest.get("current_era")
    if not current_id:
        return eras[-1] if eras else None
    for era in eras:
        if era.id == current_id:
            return era
    raise ValueError(f"current balance era {current_id!r} is not in the manifest")
