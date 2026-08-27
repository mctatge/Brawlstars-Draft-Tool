"""Scrape the Ranked **free / "boosted" brawler rotation** from the official release notes.

Each Ranked season Supercell makes a small set of **fully-maxed brawlers free for everyone**
(Power 11 with every star power / gadget / gear / hypercharge), regardless of whether the player
owns them — they rotate each season. This is exactly the set that should count as *available*
when we personalize the draft to a player's roster: they can play these even if they're not in
the collection.

No catalog or player API exposes the rotation (verified: ``/v1/events`` is empty and the player
profile carries no such field), so — exactly like :mod:`bsdraft.collect.patchnotes` — we read
Supercell's release-notes blog directly. This module deliberately **reuses that module's secure
plumbing** (on-host allow-list, manual-redirect ``_fetch``, ``__NEXT_DATA__`` extraction, blog
discovery, rich-text walking and catalog name resolution), so the boosted watch inherits the same
"only ever fetch supercell.com" guarantee and needs no key / no IP lock.

Where the rotation lives on the page (learned by inspection — there is no ranked API):

  * Inside the "Maps, Game Modes, Environments & Rotation Changes" section there is a
    ``heading-3`` titled **"Ranked"**. It is a *sub*-heading, not a top-level ``bodyCollection``
    block, so we scan every block's rich text for it rather than keying on a section title.
  * Under it, one group per season: a **bold paragraph** "Season N", then an ``unordered-list``
    whose items include ``"Featured game mode: <mode>"`` and ``"Free Brawler Rotation:"``. The
    rotation's brawlers are a **nested list** under that item — one ``list-item`` per brawler
    ("Berry", "Tara", "Meg"). (A ``_text_of`` of the *outer* item concatenates them into
    "BerryTaraMeg", so we read the leaf items — with a delimiter-free segmentation fallback for
    the flatter layout Supercell sometimes ships.)

The notes carry **no per-season dates**, so which rotation is *live right now* can't be derived
here — the earliest-listed season is treated as active and the rest as upcoming, and the
``boosted-watch`` CI job routes any change through a **human-reviewed PR** (a seasonal wholesale
swap, not an additive diff, so it must not auto-merge).

Pure ``httpx`` + stdlib (no torch/pandas) — safe to import on any dependency tier.

    PYTHONPATH=backend python -m bsdraft.collect.boosted            # human summary
    PYTHONPATH=backend python -m bsdraft.collect.boosted --json     # machine-readable
    PYTHONPATH=backend python -m bsdraft.collect.boosted --write    # update data/reference/ranked_boosted.json
    PYTHONPATH=backend python -m bsdraft.collect.boosted --url <release-notes URL>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from bsdraft.constants import REFERENCE_DIR
from bsdraft.data import reference as R
# Reuse patchnotes' vetted plumbing verbatim — same allow-listed host, same secure fetch, same
# rich-text/name-resolution helpers. Importing the private helpers keeps the two watchers in
# lockstep: a hardening fix to _fetch / _safe_url protects this scraper too.
from bsdraft.collect.patchnotes import (
    _canon,
    _cell,
    _fetch,
    _fingerprint,
    _name_index,
    _next_data,
    _resolve_id,
    _safe_url,
    _slug_of,
    _text_of,
    find_latest_release_notes,
)

BOOSTED_PATH = REFERENCE_DIR / "ranked_boosted.json"

# The label that introduces the rotation list under a season, matched loosely (case/punctuation).
_ROTATION_LABEL = "free brawler rotation"
_SEASON_RE = re.compile(r"season\s+(\d+)", re.I)


# --------------------------------------------------------------------------- data

@dataclass
class Rotation:
    season: str                              # "Season 1"
    season_num: int                          # 1
    brawlers: List[str]                      # names exactly as printed, e.g. ["Berry","Tara","Meg"]
    brawler_ids: List[Optional[int]]         # resolved against the catalog (None = unmatched)
    featured_mode: str = ""                  # "Gem Grab" — context only, not used for scoring

    @property
    def unresolved(self) -> List[str]:
        return [n for n, i in zip(self.brawlers, self.brawler_ids) if i is None]

    def to_committed(self) -> dict:
        """The shape stored in ranked_boosted.json — names only (human-readable; the serving
        layer re-resolves to ids, so a catalog rename can't strand a hard-coded id)."""
        d = {"season": self.season, "brawlers": list(self.brawlers)}
        if self.featured_mode:
            d["featured_mode"] = self.featured_mode
        return d


@dataclass
class BoostedReport:
    url: str
    title: str
    publish_date: str
    fingerprint: str                         # dedup key over slug + rotation content
    rotations: List[Rotation] = field(default_factory=list)
    # A "Ranked" heading was present but no rotation parsed under it — a likely layout change,
    # distinct from a page that simply has no ranked section this cycle. CI escalates on this.
    layout_warning: bool = False
    note: str = ""

    @property
    def active(self) -> Optional[Rotation]:
        """The current season's rotation. The notes list seasons in order with no dates, so the
        earliest-listed (lowest-numbered) is treated as live and the rest as upcoming; a human
        confirms this in the watch PR."""
        return min(self.rotations, key=lambda r: r.season_num) if self.rotations else None

    @property
    def upcoming(self) -> List[Rotation]:
        act = self.active
        return [r for r in self.rotations if r is not act]

    @property
    def unresolved(self) -> List[str]:
        seen: Dict[str, None] = {}
        for r in self.rotations:
            for n in r.unresolved:
                seen.setdefault(n, None)
        return list(seen)

    def summary(self) -> str:
        lines = [f"{self.title}  ({self.publish_date})", f"  {self.url}",
                 f"  fingerprint {self.fingerprint}"]
        if not self.rotations:
            lines.append("  (no ranked free-brawler rotation parsed)")
        for r in self.rotations:
            tag = "ACTIVE" if r is self.active else "upcoming"
            mode = f"  [featured: {r.featured_mode}]" if r.featured_mode else ""
            lines.append(f"  {tag}: {r.season} — {', '.join(r.brawlers)}{mode}")
        if self.unresolved:
            lines.append(f"  ⚠ unmatched name(s): {', '.join(self.unresolved)}")
        if self.layout_warning:
            lines.append("  ⚠ LAYOUT WARNING — a 'Ranked' section parsed no rotation (see note)")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- parse

def _first_paragraph_text(list_item: dict) -> str:
    """Text of a ``list-item``'s first *direct* paragraph child (ignoring nested lists), i.e. the
    label line — "Free Brawler Rotation:" or "Featured game mode: Gem Grab"."""
    for c in list_item.get("content", []) or []:
        if isinstance(c, dict) and c.get("nodeType") == "paragraph":
            return _text_of(c).strip()
    return ""


def _leaf_names(list_item: dict) -> List[str]:
    """The brawler names in a rotation item's nested list — one ``list-item`` per name."""
    out: List[str] = []
    for c in list_item.get("content", []) or []:
        if isinstance(c, dict) and c.get("nodeType") in ("unordered-list", "ordered-list"):
            for li in c.get("content", []) or []:
                if isinstance(li, dict) and li.get("nodeType") == "list-item":
                    txt = _text_of(li).strip()
                    if txt:
                        out.append(txt)
    return out


def _segment_names(text: str) -> List[str]:
    """Greedily peel catalog brawler names off a delimiter-free run — the fallback for the flat
    layout where the rotation reads "Free Brawler Rotation:BerryTaraMeg" instead of a nested list.
    Names are tried longest-first at each position, so "Larry & Lawrie" wins over "Larry" and
    "Colette" isn't shadowed by "Cole"."""
    keys = sorted(
        ((re.sub(r"[^a-z0-9]", "", b.name.lower()), b.name) for b in R.load_brawlers()),
        key=lambda kv: len(kv[0]), reverse=True,
    )
    s = re.sub(r"[^a-z0-9]", "", text.lower())
    out: List[str] = []
    i = 0
    while i < len(s):
        for key, name in keys:
            if key and s.startswith(key, i):
                out.append(name)
                i += len(key)
                break
        else:
            i += 1  # unmatched filler between names — skip a char and keep going
    return out


def _rotation_names(season_list: dict, index: Dict[str, R.Brawler]) -> Tuple[List[str], str]:
    """Extract ``(brawler_names, featured_mode)`` from a season's ``unordered-list`` node."""
    names: List[str] = []
    featured = ""
    for li in season_list.get("content", []) or []:
        if not (isinstance(li, dict) and li.get("nodeType") == "list-item"):
            continue
        lead = _first_paragraph_text(li)
        low = lead.lower()
        if low.startswith(_ROTATION_LABEL):
            names = _leaf_names(li)
            if not names:  # flat layout: the names are concatenated onto the label line
                tail = re.split(r"rotation\s*:?", _text_of(li), maxsplit=1, flags=re.I)
                names = _segment_names(tail[-1]) if len(tail) > 1 else []
        elif low.startswith("featured game mode"):
            featured = lead.split(":", 1)[1].strip() if ":" in lead else ""
    return names, featured


def _rotations_from_nodes(nodes: List[dict], index: Dict[str, R.Brawler]
                          ) -> Tuple[List[Rotation], bool]:
    """Walk one block's top-level rich-text sequence for the "Ranked" subsection and its season
    rotations. Returns ``(rotations, saw_ranked)`` — ``saw_ranked`` drives the layout warning."""
    rotations: List[Rotation] = []
    saw_ranked = False
    in_ranked = False
    season: Optional[str] = None
    season_num: Optional[int] = None
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nt = n.get("nodeType")
        if nt in ("heading-2", "heading-3", "heading-4"):
            in_ranked = "ranked" in _canon(_text_of(n))
            saw_ranked = saw_ranked or in_ranked
            season = season_num = None
            continue
        if not in_ranked:
            continue
        if nt == "paragraph":
            m = _SEASON_RE.search(_text_of(n).strip())
            if m:
                season, season_num = f"Season {m.group(1)}", int(m.group(1))
            continue
        if nt in ("unordered-list", "ordered-list") and season is not None:
            names, featured = _rotation_names(n, index)
            if names:
                rotations.append(Rotation(
                    season=season, season_num=season_num or 0, brawlers=names,
                    brawler_ids=[_resolve_id(x, index) for x in names], featured_mode=featured,
                ))
            season = season_num = None
    return rotations, saw_ranked


def parse_boosted(html: str, url: str) -> BoostedReport:
    """Parse one release-notes page into a :class:`BoostedReport`."""
    pp = _next_data(html).get("props", {}).get("pageProps", {})
    title = (pp.get("title") or "").strip()
    publish_date = (pp.get("publishDate") or "").strip()
    body = pp.get("bodyCollection") or []
    index = _name_index()

    rotations: List[Rotation] = []
    saw_ranked = False
    for block in body:
        if not isinstance(block, dict):
            continue
        rich = block.get("text") or {}
        doc = rich.get("json", rich) if isinstance(rich, dict) else {}
        rots, saw = _rotations_from_nodes(doc.get("content", []) or [], index)
        rotations.extend(rots)
        saw_ranked = saw_ranked or saw

    layout_warning = saw_ranked and not rotations
    note = ""
    if layout_warning:
        note = ("a 'Ranked' section is present but no 'Free Brawler Rotation' parsed — the "
                "release-notes rich-text layout may have changed; check it manually and update "
                "backend/bsdraft/collect/boosted.py")

    fp_parts = [f"{r.season}: {', '.join(r.brawlers)}" for r in rotations]
    return BoostedReport(
        url=url, title=title, publish_date=publish_date,
        fingerprint=_fingerprint(_slug_of(url), fp_parts),
        rotations=rotations, layout_warning=layout_warning, note=note,
    )


def fetch_latest(fetch=_fetch) -> Optional[BoostedReport]:
    """Discover and parse the latest release-notes page. None if none is found."""
    found = find_latest_release_notes(fetch)
    if not found:
        return None
    url, _pd, _title = found
    return parse_boosted(fetch(url), url)


# --------------------------------------------------------------------------- committed file

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _to_document(report: BoostedReport, committed: Optional[dict] = None) -> dict:
    """The committed ``ranked_boosted.json`` shape. ``valid_until`` / per-entry ``active_from``
    are *hand-staged* season-flip boundaries the serving layer honors (all UTC; see the emitted
    ``_comment``) — a scrape knows rotation *content*, never dates, so a rewrite carries the
    committed dates forward for seasons whose **names still match** rather than silently
    destroying a staged handover. A season that changed identity drops its dates (a human
    restages; the PR body says so)."""
    active = report.active
    committed = committed or {}
    old_active = committed.get("active") or {}
    valid_until = (committed.get("valid_until")
                   if active and old_active.get("season") == active.season else None)
    staged = {e.get("season"): e for e in (committed.get("upcoming") or []) if isinstance(e, dict)}
    upcoming = []
    for r in report.upcoming:
        entry = r.to_committed()
        old = staged.get(r.season) or {}
        for k in ("active_from", "valid_until"):   # hand-staged boundaries survive a rescrape
            if old.get(k):
                entry[k] = old[k]
        upcoming.append(entry)
    return {
        "_comment": ("Ranked free / 'boosted' brawlers — the maxed brawlers everyone may use in "
                     "Ranked this season, scraped from the release-notes 'Ranked' section by "
                     "backend/bsdraft/collect/boosted.py (the boosted-watch CI job). 'active' is "
                     "the current season; the serving layer resolves the names to ids and treats "
                     "them as owned-at-full-loadout when personalizing. Hand-editable. All dates "
                     "are UTC: a bare date means that whole UTC day, a full ISO datetime (e.g. "
                     "2026-08-19T10:00:00Z) pins the hour. 'valid_until' is the last moment a "
                     "rotation serves; it may sit on an entry or, legacy, at the top level for "
                     "'active'. An 'upcoming' entry may carry a hand-staged 'active_from': the "
                     "loader hands over to it automatically from that moment; an entry without "
                     "one is never served, and one with no 'valid_until' is capped at 45 days "
                     "past its start so an unmaintained file goes quiet instead of asserting a "
                     "dead free set forever. 'grants' records brawlers made free OUTSIDE the "
                     "seasonal rotation — the notes do not announce these — and is carried "
                     "through untouched by a rewrite. A boosted-watch rewrite carries these "
                     "dates forward for seasons whose names still match."),
        "source_url": _safe_url(report.url),
        "scraped_at": _today(),
        "valid_until": valid_until,
        "active": active.to_committed() if active else None,
        "upcoming": upcoming,
        # Grants are free brawlers the notes never mention (see reference._active_grants), so a
        # scrape knows nothing about them and must carry them through verbatim — rebuilding this
        # document from the page alone would silently delete a live one.
        "grants": list(committed.get("grants") or []),
    }


def load_committed() -> Optional[dict]:
    if not BOOSTED_PATH.exists():
        return None
    try:
        return json.loads(BOOSTED_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _rotation_state(doc: Optional[dict]) -> Dict[str, List[str]]:
    """Season -> [brawler names] for a committed document, for change detection."""
    if not doc:
        return {}
    out: Dict[str, List[str]] = {}
    for entry in ([doc.get("active")] + list(doc.get("upcoming") or [])):
        if isinstance(entry, dict) and entry.get("season"):
            out[entry["season"]] = list(entry.get("brawlers") or [])
    return out


def has_changed(report: BoostedReport) -> bool:
    """True when the scraped rotations differ from what's committed (drives whether CI opens a PR)."""
    scraped = {r.season: list(r.brawlers) for r in report.rotations}
    return scraped != _rotation_state(load_committed())


def write_document(report: BoostedReport) -> Path:
    """Write ``ranked_boosted.json`` atomically. Refuses to persist a rotation with an unmatched
    name (a catalog/notes mismatch a human should resolve first)."""
    if report.unresolved:
        raise ValueError(f"refusing to write: unmatched brawler name(s) {report.unresolved} — "
                         "add an alias in patchnotes._NAME_ALIASES or check the notes")
    BOOSTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BOOSTED_PATH.parent / (BOOSTED_PATH.name + ".tmp")
    tmp.write_text(json.dumps(_to_document(report, load_committed()), ensure_ascii=False, indent=2)
                   + "\n", encoding="utf-8")
    tmp.replace(BOOSTED_PATH)
    return BOOSTED_PATH


# --------------------------------------------------------------------------- PR rendering

FINGERPRINT_MARKER = "bs-boosted"


def render_pr(report: BoostedReport) -> Tuple[str, str]:
    """Render ``(title, body_markdown)`` for the boosted-rotation update PR. All scraped text
    flows through :func:`_cell`; the link target through :func:`_safe_url`."""
    active = report.active
    date = report.publish_date[:10] or "?"
    if active:
        title = f"Ranked boosted brawlers: {active.season} — {', '.join(active.brawlers)} ({date})"
    elif report.layout_warning:
        title = f"⚠️ Ranked 'Free Brawler Rotation' parsed empty — layout may have changed ({date})"
    else:
        title = f"Ranked boosted brawlers: rotation update ({date})"
    title = title[:200]

    lines = [
        f"The Ranked **free / boosted brawler rotation** in the release notes "
        f"**[{_cell(report.title)}]({_safe_url(report.url)})** no longer matches "
        f"`data/reference/ranked_boosted.json`.",
        "",
        "These are the maxed brawlers everyone may use in Ranked this season; the draft tool "
        "treats them as owned-at-full-loadout when personalizing. Scraped straight from the "
        "notes — the leading signal, no key / no IP lock. See "
        "`backend/bsdraft/collect/boosted.py`.",
        "",
        "> ⚠️ **Confirm which season is live in-game and set it as `active`** before merging — "
        "the notes list seasons in order but carry no dates, so the earliest is assumed current.",
        "",
    ]
    if report.rotations:
        lines += ["| | Season | Featured mode | Free brawlers |", "|---|---|---|---|"]
        for r in report.rotations:
            mark = "**active**" if r is active else "upcoming"
            lines.append(f"| {mark} | {_cell(r.season)} | {_cell(r.featured_mode) or '—'} | "
                         f"{', '.join(_cell(n) for n in r.brawlers)} |")
        lines.append("")
    if report.unresolved:
        lines += ["**Unmatched name(s):** " + ", ".join(f"`{_cell(n)}`" for n in report.unresolved)
                  + " — the catalog spells these differently; add an alias in "
                  "`patchnotes._NAME_ALIASES` or refresh the catalog before merging.", ""]
    if report.note:
        lines += [f"> ⚠️ {_cell(report.note)}", ""]
    lines += [
        f"_Fingerprint `{report.fingerprint}` — a new PR opens only when the rotation content "
        "changes. Edit `active`/`valid_until`/staged `active_from` dates (all UTC) in the "
        "committed file as needed: the rewrite carries dates forward only for seasons whose "
        "names still match, so **restage any season-flip boundary for renamed seasons** before "
        "merging._",
        "",
        f"<!-- {FINGERPRINT_MARKER}:{report.fingerprint} -->",
    ]
    return title, "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def _to_dict(report: BoostedReport) -> dict:
    d = asdict(report)
    d["active"] = report.active.to_committed() if report.active else None
    d["unresolved"] = report.unresolved
    d["changed"] = has_changed(report)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scrape the Ranked free/boosted brawler rotation from the release notes.")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument("--url", default=None,
                    help="parse this specific release-notes URL instead of discovering the latest")
    ap.add_argument("--write", action="store_true",
                    help="write data/reference/ranked_boosted.json when the rotation changed")
    ap.add_argument("--pr-body", metavar="PATH", default=None,
                    help="CI mode: write the PR Markdown to PATH and print a one-line JSON header "
                         "{changed,layout_warning,title,active,unresolved}")
    args = ap.parse_args()
    try:
        report = parse_boosted(_fetch(args.url), args.url) if args.url else fetch_latest()
    except (httpx.HTTPError, ValueError) as e:
        raise SystemExit(f"boosted scrape failed: {e}")
    if report is None:
        raise SystemExit("no release-notes page found on the blog index")

    # Emit the PR body + header FIRST, so CI always gets the machine-readable header even when the
    # file can't be safely written (an unmatched name / layout change routes to a human instead).
    if args.pr_body:
        title, body = render_pr(report)
        with open(args.pr_body, "w", encoding="utf-8") as fh:
            fh.write(body)
        print(json.dumps({
            "changed": has_changed(report),
            "layout_warning": report.layout_warning,
            "fingerprint": report.fingerprint,
            "title": title,
            "active": report.active.to_committed() if report.active else None,
            "unresolved": report.unresolved,
        }, ensure_ascii=False))

    if args.write:
        if report.unresolved:
            print(f"not writing: unmatched brawler name(s) {report.unresolved} — needs a human",
                  file=sys.stderr)
        elif has_changed(report):
            print(f"wrote {write_document(report)}", file=sys.stderr)
        else:
            print("ranked_boosted.json already current — nothing to write", file=sys.stderr)

    if args.json and not args.pr_body:
        print(json.dumps(_to_dict(report), ensure_ascii=False))
    elif not args.pr_body and not args.json:
        print(report.summary())


if __name__ == "__main__":
    main()
