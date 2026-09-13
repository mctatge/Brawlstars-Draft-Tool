"""Refresh the bundled reference catalog (brawlers, maps) from the keyless Brawlify API.

Brawl Stars ships new brawlers and rotates ranked maps each season. The model's brawler
vocabulary and the UI's pickable lists come from ``data/reference/*.json`` (static snapshots),
so when an update lands you must refresh them — otherwise a new brawler is invisible to the
tool (absent from the embedding index and the pick list) and silently encodes to index 0.

This fetches the catalogs, validates them, rewrites the snapshots in place (atomic), and
prints what changed. Run it when the meta banner / drift report flags a new brawler. After a
brawler is added, retrain so it gets a real embedding row, then publish + commit:

    PYTHONPATH=backend python backend/scripts/refresh_reference.py             # fetch + write
    PYTHONPATH=backend python backend/scripts/refresh_reference.py --dry-run   # report only
    PYTHONPATH=backend python backend/scripts/refresh_reference.py --accessories-only
        # refresh existing gadget/star-power details without changing brawler/map vocabularies

    PYTHONPATH=backend python backend/scripts/train.py \\
      && PYTHONPATH=backend python backend/scripts/export_model.py \\
      && PYTHONPATH=backend python -m bsdraft.collect.publish --model
    git add data/reference/brawlers.json data/reference/maps.json && git commit
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import httpx

from bsdraft.constants import RANKED_MODES, REFERENCE_DIR
from bsdraft.data.catalog import (
    CATALOG_HOSTS,
    diff_catalogs,
    fetch_catalog,
    refresh_accessory_details,
    validate as _validate,
)

# ``api.brawlify.com`` began bot-blocking automated requests (HTTP 403 "Security Check");
# ``api.brawlapi.com`` serves the identical payload. Both are tried in order — see
# bsdraft/data/catalog.py, which owns the fetching, validation and diffing this script reuses.
BRAWLERS_URL = f"{CATALOG_HOSTS[0]}/v1/brawlers"
MAPS_URL = f"{CATALOG_HOSTS[0]}/v1/maps"


def _fetch_list(url: str, path: str) -> dict:
    """Fetch one catalog. The default URL means "try the known hosts in order"; anything else is
    an explicit override and is fetched verbatim."""
    override = None if url == f"{CATALOG_HOSTS[0]}/v1/{path}" else url
    payload, src = fetch_catalog(path, url=override)
    print(f"  source: {src}")
    return payload


def _current_list(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("list", [])
    except (json.JSONDecodeError, OSError):
        return []


def _brawler_names(items: List[dict]) -> Dict[int, str]:
    return {x["id"]: x.get("name", str(x["id"])) for x in items if isinstance(x.get("id"), int)}


def _ranked_map_names(items: List[dict]) -> Dict[int, str]:
    """Active maps in the ranked mode set, id -> 'Name (Mode)' — mirrors reference.load_ranked_maps."""
    out: Dict[int, str] = {}
    for x in items:
        if not isinstance(x.get("id"), int) or x.get("disabled"):
            continue
        mode = (x.get("gameMode") or {}).get("name")
        if mode in RANKED_MODES:
            out[x["id"]] = f'{x.get("name", "?")} ({mode})'
    return out


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    # Minified, matching the committed snapshots so diffs stay small.
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def refresh(brawlers_url: str = BRAWLERS_URL, maps_url: str = MAPS_URL, dry_run: bool = False,
            accessories_only: bool = False) -> bool:
    """Fetch + validate both catalogs, report the diff vs the local snapshots, and (unless
    dry_run) rewrite them. Returns True iff a new brawler was detected."""
    b_path = REFERENCE_DIR / "brawlers.json"
    m_path = REFERENCE_DIR / "maps.json"

    before_list = _current_list(b_path)
    before_b = _brawler_names(before_list)
    before_m = _ranked_map_names(_current_list(m_path))

    b_payload = _fetch_list(brawlers_url, "brawlers")
    if accessories_only:
        merged = {"list": copy.deepcopy(before_list)}
        detail_changes = refresh_accessory_details(merged, b_payload)
        print(f"existing accessory details: {len(detail_changes)} change(s)")
        for note in detail_changes:
            print(f"  {note}")
        if dry_run:
            print("\n--dry-run: no files written.")
            return False
        if detail_changes:
            _write_atomic(b_path, merged)
            print(f"\nwrote {b_path} (existing brawlers/accessories only)")
        else:
            print("  (no existing accessory details changed)")
        return False
    m_payload = _fetch_list(maps_url, "maps")
    after_list = _validate(b_payload, "brawlers")
    after_b = _brawler_names(after_list)
    after_m = _ranked_map_names(_validate(m_payload, "maps"))

    new_brawlers = [f"{after_b[i]} (#{i})" for i in after_b if i not in before_b]
    gone_brawlers = [f"{before_b[i]} (#{i})" for i in before_b if i not in after_b]
    new_maps = sorted(after_m[i] for i in after_m if i not in before_m)
    # Star power / gadget changes matter to mastery + loadout scoring but used to pass silently:
    # this script only ever diffed brawler and map *names*.
    acc_changes = diff_catalogs(before_list, after_list).accessory_changes

    print(f"brawlers: {len(before_b)} -> {len(after_b)}")
    if new_brawlers:
        print("  NEW: " + ", ".join(new_brawlers))
    if gone_brawlers:
        print("  removed: " + ", ".join(gone_brawlers))
    if acc_changes:
        print(f"star powers / gadgets: {len(acc_changes)} change(s)")
        for c in acc_changes:
            detail = f"{c.old_name!r} -> {c.name!r}" if c.change == "renamed" else c.name
            print(f"  {c.change} {c.kind}: {c.brawler} — {detail}")
    print(f"ranked maps: {len(before_m)} -> {len(after_m)}")
    if new_maps:
        print("  NEW: " + ", ".join(new_maps))
    if not (new_brawlers or gone_brawlers or new_maps or acc_changes):
        print("  (no changes vs local snapshots)")

    if dry_run:
        print("\n--dry-run: no files written.")
        return bool(new_brawlers)

    _write_atomic(b_path, b_payload)
    _write_atomic(m_path, m_payload)
    print(f"\nwrote {b_path}\n      {m_path}")
    if new_brawlers:
        print("\nNew brawler(s) — they encode to index 0 until you retrain. Roll out:")
        print("  python backend/scripts/train.py && python backend/scripts/export_model.py \\")
        print("    && python -m bsdraft.collect.publish --model")
        print("  git add data/reference/*.json && git commit")
    return bool(new_brawlers)


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh data/reference/{brawlers,maps}.json from Brawlify.")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; don't write")
    ap.add_argument("--accessories-only", action="store_true",
                    help="refresh names/descriptions for existing accessories without adding "
                         "brawlers or changing maps")
    ap.add_argument("--brawlers-url", default=BRAWLERS_URL, help="override the brawlers catalog URL")
    ap.add_argument("--maps-url", default=MAPS_URL, help="override the maps catalog URL")
    args = ap.parse_args()
    try:
        refresh(args.brawlers_url, args.maps_url, dry_run=args.dry_run,
                accessories_only=args.accessories_only)
    except (httpx.HTTPError, ValueError) as e:
        raise SystemExit(f"refresh failed: {e}")


if __name__ == "__main__":
    main()
