"""Watch the live brawler catalog and diff it against the committed snapshot.

The reference catalog in ``data/reference/*.json`` is a *committed static snapshot*: it defines
the model's brawler vocabulary (the embedding index), the UI's pickable lists, and — through
each brawler's class — the composition/game-plan reasoning. Anything the catalog doesn't know
about is invisible to the tool, and a brand-new brawler silently encodes to embedding index 0.

This module is the **structured** half of the new-content watch, complementing the prose half in
:mod:`bsdraft.collect.patchnotes`:

  * here — ids, names, and descriptions straight from the keyless catalog API: new/removed/renamed
    **brawlers**, their **star powers** and **gadgets**, plus **class/rarity** changes (which move
    composition reasoning). Exact, machine-checkable, no parsing of English.
  * patchnotes — everything the catalog does NOT expose: **gears, hypercharges, buffies** and the
    actual balance numbers. Verified: ``/v1/{gadgets,starpowers,gears}`` are 404 and brawler
    records carry no gears/hypercharge/buffies field, so the release notes are the only source
    for those. (Buffies do show up per-player on the roster as owned/not-owned flags; the curated
    cumulative availability policy in ``data/reference/buffies.json`` supplies the missing
    existence half used by ``engine/mastery.py``.)

Source hosts: ``api.brawlify.com`` began bot-blocking automated requests (HTTP 403 "Security
Check") — ``api.brawlapi.com`` serves the identical payload and is tried first, with the
brawlify host kept as a fallback in case that flips back.

Stdlib + httpx only (no torch/pandas), so it is safe on every dependency tier.

    PYTHONPATH=backend python -m bsdraft.data.catalog             # human diff vs the snapshot
    PYTHONPATH=backend python -m bsdraft.data.catalog --json      # machine-readable (CI)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import httpx

from bsdraft.constants import BRAWLER_CLASSES, REFERENCE_DIR

# Ordered by preference: the brawlify host now 403s behind bot protection, but keep it as a
# fallback so a flip back (or an outage on the primary) doesn't break the watch.
CATALOG_HOSTS = ("https://api.brawlapi.com", "https://api.brawlify.com")
_UA = "bsdraft-catalog/1.0 (Brawl Stars draft tool)"

# Accessory kinds nested inside a brawler record. Gears/hypercharges/buffies are deliberately
# absent — the API does not expose them (see the module docstring).
ACCESSORY_FIELDS = (("starPowers", "star power"), ("gadgets", "gadget"))


# --------------------------------------------------------------------------- data

@dataclass
class AccessoryChange:
    kind: str            # "star power" | "gadget"
    change: str          # "added" | "removed" | "renamed" | "description"
    accessory_id: int
    brawler: str
    name: str
    old_name: str = ""   # only for "renamed"
    old_description: str = ""   # only for "description"
    new_description: str = ""   # only for "description"


@dataclass
class BrawlerChange:
    change: str          # "added" | "removed" | "renamed" | "class" | "rarity"
    brawler_id: int
    name: str
    detail: str = ""     # e.g. "Damage Dealer -> Assassin"
    old_value: str = ""  # for renamed/class/rarity — the previous value
    new_value: str = ""  # …and the new one, so safety rules don't parse `detail`


@dataclass
class CatalogDiff:
    n_before: int
    n_after: int
    brawler_changes: List[BrawlerChange] = field(default_factory=list)
    accessory_changes: List[AccessoryChange] = field(default_factory=list)

    # --- convenience views -------------------------------------------------
    def _b(self, kind: str) -> List[BrawlerChange]:
        return [c for c in self.brawler_changes if c.change == kind]

    def _a(self, kind: str) -> List[AccessoryChange]:
        return [c for c in self.accessory_changes if c.change == kind]

    @property
    def new_brawlers(self) -> List[BrawlerChange]:
        return self._b("added")

    @property
    def removed_brawlers(self) -> List[BrawlerChange]:
        return self._b("removed")

    @property
    def changed(self) -> bool:
        return bool(self.brawler_changes or self.accessory_changes)

    # A degraded upstream payload rarely *removes* only one thing, so a burst of edits to
    # existing entries is treated as suspicious even when nothing disappears outright.
    MAX_EDITS = 5

    @property
    def destructive(self) -> List[str]:
        """Reasons this diff must NOT land unattended. Removals are rare and usually mean the
        upstream API served a partial/degraded payload rather than Supercell deleting content —
        exactly the case a human should eyeball before it rewrites the model's vocabulary."""
        why: List[str] = []
        if self.n_after < self.n_before:
            why.append(f"catalog shrank ({self.n_before} -> {self.n_after} brawlers)")
        for c in self.removed_brawlers:
            why.append(f"brawler removed: {c.name} (#{c.brawler_id})")
        for c in self._a("removed"):
            why.append(f"{c.kind} removed: {c.brawler} — {c.name}")
        # Losing a class is data loss too: it drops the brawler to UNCLASSIFIED and silently
        # degrades the composition reasoning, without removing anything the count would catch.
        for c in self._b("class"):
            if c.old_value and c.new_value in ("", "Unknown"):
                why.append(f"class lost: {c.name} ({c.old_value} -> {c.new_value or 'missing'})")
        edits = len(self._b("renamed")) + len(self._b("class")) + len(self._b("rarity"))
        if edits > self.MAX_EDITS:
            why.append(f"{edits} edits to existing brawlers in one diff "
                       f"(> {self.MAX_EDITS}) — looks like a schema change, not a patch")
        return why

    @property
    def safe_to_automerge(self) -> bool:
        """Purely additive (plus renames/reclassifications): nothing the snapshot already has
        disappears. Only these land without review."""
        return self.changed and not self.destructive

    def summary(self) -> str:
        if not self.changed:
            return f"catalog unchanged ({self.n_after} brawlers) — snapshot is current"
        lines = [f"catalog changed: {self.n_before} -> {self.n_after} brawlers"]
        for label, items in (
            ("NEW brawler(s)", self.new_brawlers),
            ("REMOVED brawler(s)", self.removed_brawlers),
            ("renamed", self._b("renamed")),
            ("class change", self._b("class")),
            ("rarity change", self._b("rarity")),
        ):
            for c in items:
                suffix = f"  [{c.detail}]" if c.detail else ""
                lines.append(f"  {label}: {c.name} (#{c.brawler_id}){suffix}")
        for c in self.accessory_changes:
            if c.change == "renamed":
                lines.append(f"  {c.kind} renamed: {c.brawler} — {c.old_name!r} -> {c.name!r}")
            elif c.change == "description":
                lines.append(f"  {c.kind} description updated: {c.brawler} — {c.name}")
            else:
                lines.append(f"  {c.kind} {c.change}: {c.brawler} — {c.name}")
        if self.destructive:
            lines.append("  ⚠ DESTRUCTIVE — needs review: " + "; ".join(self.destructive))
        return "\n".join(lines)


# --------------------------------------------------------------------------- fetch / validate

def validate(payload: dict, label: str) -> List[dict]:
    """Make sure a payload looks like a real catalog before it is trusted to overwrite a good
    local snapshot. Raises ValueError on anything suspicious (wrong URL, HTML error page, empty
    list, malformed items) so a bad fetch never clobbers working data."""
    if not isinstance(payload, dict) or not isinstance(payload.get("list"), list):
        raise ValueError(f"{label}: response has no 'list' array — wrong URL or the API changed?")
    items = payload["list"]
    if not items:
        raise ValueError(f"{label}: 'list' is empty — refusing to overwrite local data")
    bad = [x for x in items
           if not isinstance(x, dict) or not isinstance(x.get("id"), int) or not x.get("name")]
    if bad:
        raise ValueError(f"{label}: {len(bad)} item(s) missing an integer id/name — refusing to overwrite")
    return items


def dedupe_accessories(payload: dict, label: str = "brawlers") -> List[str]:
    """Strip accessories the catalog serves under more than one brawler, in place.

    Accessory ids are Supercell-assigned and belong to exactly one brawler, but the live API has
    served the same gadget under two (Bolt carrying Brock's "Rocket Laces"/"Rocket Fuel",
    2026-08). A duplicated id corrupts everything keyed off a brawler's kit — loadout effect
    classification, itemstats ownership inference, mastery build scores — so it must never reach
    the snapshot.

    Resolution: the accessory's own description names its owner ("Brock jumps to targeted
    area…"), so a duplicate whose description names exactly one claimant is kept there and
    stripped from the rest. Anything the descriptions can't disambiguate raises ValueError —
    refusing to overwrite a good snapshot beats guessing. Returns notes of what was stripped,
    for the caller to log."""
    claims: Dict[int, List[Tuple[dict, dict]]] = {}
    for b in (payload.get("list") or []):
        for fieldname, _kind in ACCESSORY_FIELDS:
            for a in (b.get(fieldname) or []):
                if isinstance(a, dict) and isinstance(a.get("id"), int):
                    claims.setdefault(a["id"], []).append((b, a))
    notes: List[str] = []
    for aid, claimants in sorted(claims.items()):
        if len(claimants) < 2:
            continue
        desc = next((a.get("description") for _, a in claimants if a.get("description")), "") or ""
        named = [(b, a) for b, a in claimants
                 if b.get("name") and re.search(rf"\b{re.escape(b['name'])}\b", desc)]
        if len(named) != 1:
            owners = ", ".join(sorted({b.get("name", "?") for b, _ in claimants}))
            raise ValueError(
                f"{label}: accessory {claimants[0][1].get('name', '?')!r} (#{aid}) is claimed by "
                f"{owners} and its description does not single out an owner — refusing to trust "
                f"this payload")
        owner = named[0][0]
        for b, a in claimants:
            if b is owner:
                continue
            for fieldname, kind in ACCESSORY_FIELDS:
                if a in (b.get(fieldname) or []):
                    b[fieldname].remove(a)
                    notes.append(f"stripped {kind} {a.get('name', '?')!r} (#{aid}) from "
                                 f"{b.get('name', '?')} — description names "
                                 f"{owner.get('name', '?')}")
    return notes


def refresh_accessory_details(current_payload: dict, live_payload: dict) -> List[str]:
    """Update known brawlers' accessory details from a validated live payload, in place.

    This deliberately preserves the current brawler list and accessory membership: a catalog
    refresh that adds brawlers or removes accessories can change model vocabulary or user-facing
    choices and needs a separate rollout. Existing accessory ids are stable, so their name/path,
    description, image, release flag, and catalog version can be refreshed safely. Returns one
    human-readable note per changed accessory.
    """
    live_brawlers = {
        b.get("id"): b for b in (live_payload.get("list") or [])
        if isinstance(b, dict) and isinstance(b.get("id"), int)
    }
    notes: List[str] = []
    detail_fields = ("name", "path", "version", "description", "descriptionHtml",
                     "imageUrl", "released")
    for current_brawler in current_payload.get("list") or []:
        if not isinstance(current_brawler, dict):
            continue
        live_brawler = live_brawlers.get(current_brawler.get("id"))
        if live_brawler is None:
            continue
        bname = current_brawler.get("name", str(current_brawler.get("id")))
        for fieldname, kind in ACCESSORY_FIELDS:
            live_items = {
                a.get("id"): a for a in (live_brawler.get(fieldname) or [])
                if isinstance(a, dict) and isinstance(a.get("id"), int)
            }
            for current_item in current_brawler.get(fieldname) or []:
                if not isinstance(current_item, dict):
                    continue
                live_item = live_items.get(current_item.get("id"))
                if live_item is None:
                    continue
                changed = []
                for key in detail_fields:
                    if key in live_item and current_item.get(key) != live_item[key]:
                        current_item[key] = live_item[key]
                        changed.append(key)
                if changed:
                    notes.append(f"updated {kind} {live_item.get('name', current_item.get('id'))!r} "
                                 f"(#{current_item.get('id')}) on {bname}: {', '.join(changed)}")
    return notes


def fetch_catalog(path: str, hosts: Iterable[str] = CATALOG_HOSTS,
                  timeout: float = 30.0, url: Optional[str] = None) -> Tuple[dict, str]:
    """GET ``/v1/<path>`` from the first host that returns a valid catalog, or exactly ``url``
    when given (an explicit override is used verbatim — never rewritten into another path).
    Returns ``(payload, source_url)``. Raises the last error if every candidate fails — a
    bot-block serves HTML under a 200/403, so the JSON decode, :func:`validate` and (for
    brawler catalogs) :func:`dedupe_accessories` are part of the liveness check, not just the
    status code. Every writer of ``brawlers.json`` fetches through here, so a duplicated
    accessory is repaired (or rejected) before it can reach the snapshot."""
    last: Optional[Exception] = None
    candidates = [url] if url else [f"{h}/v1/{path}" for h in hosts]
    for url in candidates:
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout,
                              headers={"User-Agent": _UA}) as client:
                resp = client.get(url)
            resp.raise_for_status()
            payload = resp.json()
            validate(payload, path)
            if path == "brawlers":
                for note in dedupe_accessories(payload, path):
                    print(f"catalog: {note}", file=sys.stderr)
            return payload, url
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as e:
            last = e
            continue
    raise ValueError(f"no source served a valid {path} catalog ({', '.join(candidates)}): {last}")


def load_snapshot(path: Path) -> List[dict]:
    """The committed snapshot's ``list``, or [] when absent/unreadable (first run)."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("list", [])
    except (json.JSONDecodeError, OSError):
        return []


# --------------------------------------------------------------------------- diff

def _class_of(b: dict) -> str:
    return (b.get("class") or {}).get("name") or ""


def _rarity_of(b: dict) -> str:
    return (b.get("rarity") or {}).get("name") or ""


def _accessories(brawlers: Iterable[dict]) -> Dict[int, Tuple[str, str, str, str]]:
    """accessory id -> (kind, brawler, name, description) across every brawler."""
    out: Dict[int, Tuple[str, str, str, str]] = {}
    for b in brawlers:
        bname = b.get("name", str(b.get("id")))
        for fieldname, kind in ACCESSORY_FIELDS:
            for a in (b.get(fieldname) or []):
                if isinstance(a, dict) and isinstance(a.get("id"), int):
                    out[a["id"]] = (kind, bname, a.get("name", ""),
                                    a.get("description", "") or "")
    return out


def diff_catalogs(before: List[dict], after: List[dict]) -> CatalogDiff:
    """Compare two brawler catalogs by id. Ids are stable (Supercell assigns them), so an id
    present on one side only is a genuine add/remove, while a differing name/class/rarity on the
    same id is a rename/reclassification."""
    ob = {b["id"]: b for b in before if isinstance(b.get("id"), int)}
    nb = {b["id"]: b for b in after if isinstance(b.get("id"), int)}

    changes: List[BrawlerChange] = []
    for bid in sorted(set(nb) - set(ob)):
        changes.append(BrawlerChange("added", bid, nb[bid].get("name", str(bid)),
                                     detail=" / ".join(x for x in (_rarity_of(nb[bid]),
                                                                   _class_of(nb[bid])) if x)))
    for bid in sorted(set(ob) - set(nb)):
        changes.append(BrawlerChange("removed", bid, ob[bid].get("name", str(bid))))
    for bid in sorted(set(ob) & set(nb)):
        o, n = ob[bid], nb[bid]
        if o.get("name") != n.get("name"):
            changes.append(BrawlerChange("renamed", bid, n.get("name", ""),
                                         detail=f"{o.get('name')} -> {n.get('name')}",
                                         old_value=o.get("name", ""), new_value=n.get("name", "")))
        if _class_of(o) != _class_of(n):
            changes.append(BrawlerChange("class", bid, n.get("name", ""),
                                         detail=f"{_class_of(o) or '?'} -> {_class_of(n) or '?'}",
                                         old_value=_class_of(o), new_value=_class_of(n)))
        if _rarity_of(o) != _rarity_of(n):
            changes.append(BrawlerChange("rarity", bid, n.get("name", ""),
                                         detail=f"{_rarity_of(o) or '?'} -> {_rarity_of(n) or '?'}",
                                         old_value=_rarity_of(o), new_value=_rarity_of(n)))

    oa, na = _accessories(before), _accessories(after)
    acc: List[AccessoryChange] = []
    for aid in sorted(set(na) - set(oa)):
        kind, bname, aname, _description = na[aid]
        acc.append(AccessoryChange(kind, "added", aid, bname, aname))
    for aid in sorted(set(oa) - set(na)):
        kind, bname, aname, _description = oa[aid]
        acc.append(AccessoryChange(kind, "removed", aid, bname, aname))
    for aid in sorted(set(oa) & set(na)):
        kind, bname, aname, description = na[aid]
        if oa[aid][2] != aname:
            acc.append(AccessoryChange(kind, "renamed", aid, bname, aname, old_name=oa[aid][2]))
        if oa[aid][3] != description:
            acc.append(AccessoryChange(kind, "description", aid, bname, aname,
                                       old_description=oa[aid][3],
                                       new_description=description))

    return CatalogDiff(n_before=len(ob), n_after=len(nb),
                       brawler_changes=changes, accessory_changes=acc)


def diff_against_snapshot(hosts: Iterable[str] = CATALOG_HOSTS
                          ) -> Tuple[CatalogDiff, dict, str]:
    """Fetch the live brawler catalog and diff it against the committed snapshot. Returns
    ``(diff, live_payload, source_url)`` — the payload is handed back so a caller can write it
    without re-fetching (and thus without risking a different payload than the one diffed)."""
    payload, url = fetch_catalog("brawlers", hosts)
    before = load_snapshot(REFERENCE_DIR / "brawlers.json")
    return diff_catalogs(before, payload["list"]), payload, url


# --------------------------------------------------------------------------- apply

def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    # Minified, matching the committed snapshots so diffs stay small.
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def resolve_classes_from_notes(names: Iterable[str]) -> Dict[str, str]:
    """Look up each name's class in the latest official release notes.

    The catalog API tags brand-new brawlers as class "Unknown" (verified: 20 of 107 live
    brawlers, i.e. every recent release), and an unclassified brawler silently degrades the
    composition/game-plan reasoning. The notes print "Nori - Legendary - Assassin", so the two
    watchers together fill the gap the catalog alone leaves. Best-effort: any failure yields {}
    and the brawler simply stays unclassified until a human adds the override."""
    wanted = {n for n in names if n}
    if not wanted:
        return {}
    try:  # imported lazily so a notes/network hiccup can't break a plain catalog diff
        from bsdraft.collect import patchnotes
        report = patchnotes.fetch_latest()
    except Exception:  # noqa: BLE001 — advisory enrichment only
        return {}
    if report is None:
        return {}
    norm = {re.sub(r"[^a-z0-9]", "", n.lower()): n for n in wanted}
    out: Dict[str, str] = {}
    for info in report.new_brawler_info:
        key = re.sub(r"[^a-z0-9]", "", info.name.lower())
        if key in norm and info.cls:
            out[norm[key]] = info.cls  # keyed by the CATALOG's spelling, which reference.py reads
    return out


def apply_catalog(payload: dict, class_overrides: Optional[Dict[str, str]] = None,
                  renames: Optional[Iterable[Tuple[str, str]]] = None) -> List[Path]:
    """Write the refreshed brawler snapshot (and any class-override updates) in place. Returns
    the paths written. Callers must have validated ``payload`` first — :func:`fetch_catalog` does.

    ``maps.json`` is deliberately NOT written here. Ranked-map indices are positional (encoders
    builds them from the (mode, name)-sorted list), so inserting one map shifts every later map
    onto a neighbour's trained embedding row — an in-range, silently-wrong lookup no
    out-of-vocabulary guard can catch. Refreshing maps is therefore tied to a retrain and stays
    in ``scripts/refresh_reference.py``, which reports a ranked-map diff for a human."""
    written: List[Path] = []
    b_path = REFERENCE_DIR / "brawlers.json"
    _write_atomic(b_path, payload)
    written.append(b_path)

    from bsdraft.data.reference import CLASS_OVERRIDES_PATH
    doc = json.loads(CLASS_OVERRIDES_PATH.read_text(encoding="utf-8"))
    ov = doc.setdefault("overrides", {})
    touched = False
    # A rename would orphan the override (it is keyed by name), silently dropping the brawler to
    # UNCLASSIFIED — carry it across before adding anything new.
    for old, new in (renames or []):
        if old in ov and new and new not in ov:
            ov[new] = ov.pop(old)
            touched = True
    for k, v in (class_overrides or {}).items():
        if k not in ov:
            ov[k] = v
            touched = True
    if touched:
        doc["_updated"] = _today()
        CLASS_OVERRIDES_PATH.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(CLASS_OVERRIDES_PATH)
    return written


def _today() -> str:
    """UTC date stamp for the overrides file's ``_updated`` field."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- rendering

def render_pr(diff: CatalogDiff, source_url: str,
              class_overrides: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """Render ``(title, body_markdown)`` for the catalog-refresh pull request."""
    new = diff.new_brawlers
    if new:
        title = "Catalog: add " + ", ".join(c.name for c in new)
    elif diff.accessory_changes and not diff._b("added"):
        title = f"Catalog: {len(diff.accessory_changes)} accessory change(s)"
    else:
        title = f"Catalog: {len(diff.brawler_changes)} brawler change(s)"

    lines = [
        f"Automated refresh of `data/reference/` — the live catalog no longer matches the "
        f"committed snapshot.",
        "",
        f"Source: `{source_url}` · brawlers **{diff.n_before} → {diff.n_after}**",
        "",
    ]
    if new:
        lines += ["## 🆕 New brawlers", "",
                  "| Brawler | Id | Rarity / class |", "|---|---|---|"]
        for c in new:
            lines.append(f"| **{c.name}** | `{c.brawler_id}` | {c.detail or '—'} |")
        lines += ["",
                  "_Until the model retrains they encode to embedding index 0; the catalog "
                  "commit alone makes them pickable in the UI._", ""]
    if class_overrides:
        lines += ["## 🏷 Class overrides added", "",
                  "The catalog tags brand-new brawlers as class `Unknown`, which would leave them "
                  "`UNCLASSIFIED` and degrade composition reasoning. Classes taken from the "
                  "official release notes:", ""]
        lines += [f"- **{k}** → `{v}`" for k, v in sorted(class_overrides.items())]
        lines.append("")
    if diff.accessory_changes:
        lines += [f"## 🔧 Star power / gadget changes ({len(diff.accessory_changes)})", "",
                  "| Change | Kind | Brawler | Name |", "|---|---|---|---|"]
        for c in diff.accessory_changes:
            if c.change == "renamed":
                nm = f"`{c.old_name}` → `{c.name}`"
            elif c.change == "description":
                nm = f"{c.name} (description updated)"
            else:
                nm = c.name
            lines.append(f"| {c.change} | {c.kind} | {c.brawler} | {nm} |")
        lines.append("")
    rest = [c for c in diff.brawler_changes if c.change != "added"]
    if rest:
        lines += ["## ♻️ Other brawler changes", ""]
        lines += [f"- {c.change}: **{c.name}** (`{c.brawler_id}`) {c.detail}" for c in rest]
        lines.append("")
    if diff.destructive:
        lines += ["## ⚠️ Needs review — not auto-merged", "",
                  "This diff removes content, which usually means the upstream API served a "
                  "partial payload rather than Supercell deleting anything:", ""]
        lines += [f"- {r}" for r in diff.destructive]
        lines += ["", "Confirm against the live catalog before merging.", ""]
    lines += ["---",
              "_Opened by the `catalog-watch` job (`.github/workflows/keepwarm.yml`) from "
              "`bsdraft.data.catalog`. Touches `brawlers.json` + `class_overrides.json` only — "
              "`maps.json` is not auto-refreshed, because ranked-map indices are positional and "
              "inserting a map would shift every later map onto a neighbour's trained embedding "
              "row (run `scripts/refresh_reference.py` alongside a retrain for maps). Retraining "
              "happens on the home crawler; a new brawler gets a real embedding row then._"]
    return title, "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def _to_dict(diff: CatalogDiff) -> dict:
    d = asdict(diff)
    d["changed"] = diff.changed
    d["safe_to_automerge"] = diff.safe_to_automerge
    d["destructive"] = diff.destructive
    return d


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diff the live brawler catalog against the committed reference snapshot.")
    ap.add_argument("--json", action="store_true", help="emit the diff as JSON (for CI)")
    ap.add_argument("--write", action="store_true",
                    help="write the refreshed brawler snapshot and any class-override updates "
                         "when the catalog changed (maps are left to refresh_reference.py — see "
                         "apply_catalog)")
    ap.add_argument("--pr-body", metavar="PATH", default=None,
                    help="CI mode: write the pull-request Markdown to PATH and print a one-line "
                         "JSON header {changed,safe_to_automerge,title,destructive,...}")
    args = ap.parse_args()
    try:
        diff, payload, url = diff_against_snapshot()
    except (httpx.HTTPError, ValueError) as e:
        raise SystemExit(f"catalog check failed: {e}")

    overrides: Dict[str, str] = {}
    if diff.changed and (args.write or args.pr_body):
        # Only brawlers the catalog itself couldn't classify need an override from the notes.
        unknown = [c.name for c in diff.new_brawlers
                   if not (c.detail or "").endswith(tuple(BRAWLER_CLASSES))]
        overrides = resolve_classes_from_notes(unknown)

    if args.write and diff.changed:
        renames = [(c.old_value, c.new_value) for c in diff.brawler_changes
                   if c.change == "renamed"]
        written = apply_catalog(payload, overrides, renames)
        # stderr: --pr-body mode requires stdout to carry the JSON header and nothing else,
        # or the caller's `jq` chokes on the progress lines.
        for p in written:
            print(f"wrote {p}", file=sys.stderr)

    if args.pr_body:
        title, body = render_pr(diff, url, overrides)
        with open(args.pr_body, "w", encoding="utf-8") as fh:
            fh.write(body)
        print(json.dumps({
            "changed": diff.changed,
            "safe_to_automerge": diff.safe_to_automerge,
            "title": title,
            "destructive": diff.destructive,
            "n_before": diff.n_before,
            "n_after": diff.n_after,
            "new_brawlers": [c.name for c in diff.new_brawlers],
            "class_overrides": overrides,
        }, ensure_ascii=False))
    elif args.json:
        print(json.dumps(_to_dict(diff), ensure_ascii=False))
    else:
        print(f"source: {url}")
        print(diff.summary())


if __name__ == "__main__":
    main()
