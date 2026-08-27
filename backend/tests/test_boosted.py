"""Unit tests for the Ranked boosted-brawler scraper (bsdraft.collect.boosted) and the serving
loader (bsdraft.data.reference.load_ranked_boosted).

Offline — parses a committed HTML fixture trimmed from the *real* release-notes "Ranked"
subsection, plus synthetic ``__NEXT_DATA__`` pages that exercise the layout variations the parser
must survive (nested-list rotation, flat/concatenated rotation, season ordering, layout drift).

    PYTHONPATH=backend python -m pytest backend/tests/test_boosted.py    # or run directly
"""
from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bsdraft.collect import boosted as B
from bsdraft.data import reference as R

FIX = Path(__file__).resolve().parent / "fixtures" / "patchnotes"
JUNE_URL = "https://supercell.com/en/games/brawlstars/blog/release-notes/release-notes-june-2026/"


def _fixture(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


# --- minimal __NEXT_DATA__ builders ---------------------------------------------

def _text(value, *marks):
    return {"nodeType": "text", "value": value, "marks": [{"type": m} for m in marks], "data": {}}

def _p(value, *marks):
    return {"nodeType": "paragraph", "data": {}, "content": [_text(value, *marks)]}

def _h3(value):
    return {"nodeType": "heading-3", "data": {}, "content": [_text(value, "bold")]}

def _li(*content):
    return {"nodeType": "list-item", "data": {}, "content": list(content)}

def _ul(*items):
    return {"nodeType": "unordered-list", "data": {}, "content": list(items)}

def _leaf_ul(*names):
    """A rotation's nested list — one list-item per brawler name."""
    return _ul(*[_li(_p(n)) for n in names])

def _rt(*content):
    return {"json": {"nodeType": "document", "data": {}, "content": list(content)}}

def _mk_html(blocks, title="Release Notes X", publish_date="2026-08-04T00:00:00.000+03:00"):
    nd = {"props": {"pageProps": {"title": title, "publishDate": publish_date,
          "bodyCollection": [{"__typename": "TextBlock", "title": t, "text": rt}
                             for t, rt in blocks]}}}
    return ('<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(nd, ensure_ascii=False) + "</script>")


def _season(label, featured, names, nested=True):
    """The nodes for one season group: a bold "Season N" paragraph then its rotation list."""
    if nested:
        rot = _li(_p("Free Brawler Rotation:"), _leaf_ul(*names))
    else:  # flat layout: names concatenated onto the label line, no nested list
        rot = _li(_p("Free Brawler Rotation:" + "".join(names)))
    feat = _li(_p(f"Featured game mode: {featured}"))
    return [_p(label, "bold"), _ul(feat, rot)]


def _ranked_block(*season_groups, title="Maps, Game Modes, Environments & Rotation Changes"):
    content = [_h3("Ranked")]
    for g in season_groups:
        content.extend(g)
    return (title, _rt(*content))


# --- real fixture ----------------------------------------------------------------

def _real() -> B.BoostedReport:
    return B.parse_boosted(_fixture("release_notes_ranked_boosted.html"), JUNE_URL)


def test_real_fixture_parses_both_rotations():
    r = _real()
    assert [(x.season, x.brawlers) for x in r.rotations] == [
        ("Season 1", ["Berry", "Tara", "Meg"]),
        ("Season 2", ["Trunk", "Willow", "Kaze"]),
    ]
    assert not r.layout_warning and not r.unresolved


def test_real_fixture_active_is_current_season():
    r = _real()
    assert r.active.season == "Season 1"
    assert r.active.brawlers == ["Berry", "Tara", "Meg"]
    assert r.active.featured_mode == "Gem Grab"
    assert [u.season for u in r.upcoming] == ["Season 2"]


def test_real_fixture_all_names_resolve():
    for rot in _real().rotations:
        assert None not in rot.brawler_ids, rot.season


# --- synthetic layout variations -------------------------------------------------

def test_nested_list_layout():
    html = _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]))])
    r = B.parse_boosted(html, JUNE_URL)
    assert r.active.brawlers == ["Berry", "Tara", "Meg"]


def test_flat_concatenated_layout_uses_segmentation():
    # The looser layout where names run together on the label line ("...Rotation:BerryTaraMeg").
    html = _mk_html([_ranked_block(
        _season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"], nested=False))])
    r = B.parse_boosted(html, JUNE_URL)
    assert r.active.brawlers == ["Berry", "Tara", "Meg"]
    assert not r.layout_warning


def test_segment_names_splits_delimiter_free_run():
    assert B._segment_names("BerryTaraMeg") == ["Berry", "Tara", "Meg"]
    # longest-first: "Larry & Lawrie" wins over "Larry", and "Colette" isn't shadowed by "Cole"
    assert B._segment_names("Larry & LawrieColette") == ["Larry & Lawrie", "Colette"]


def test_active_is_lowest_numbered_even_if_listed_out_of_order():
    html = _mk_html([_ranked_block(
        _season("Season 2", "Brawl Ball", ["Trunk", "Willow", "Kaze"]),
        _season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]))])
    r = B.parse_boosted(html, JUNE_URL)
    assert r.active.season == "Season 1"


def test_ranked_subsection_found_inside_any_block():
    # "Ranked" is a heading-3 nested in a larger section, not a top-level bodyCollection block.
    html = _mk_html([
        ("Bug Fixes", _rt(_p("fixed things"))),
        _ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"])),
    ])
    r = B.parse_boosted(html, JUNE_URL)
    assert r.active.brawlers == ["Berry", "Tara", "Meg"]


def test_ranked_subsection_ends_at_next_heading():
    # A season group after a *different* heading must not be read as a ranked rotation.
    block = ("Maps", _rt(
        _h3("Ranked"), *_season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]),
        _h3("Maps"), *_season("Season 9", "Heist", ["Shelly", "Colt", "Nita"]),
    ))
    r = B.parse_boosted(_mk_html([block]), JUNE_URL)
    assert [x.season for x in r.rotations] == ["Season 1"]


# --- layout warning --------------------------------------------------------------

def test_layout_warning_when_ranked_present_but_no_rotation():
    html = _mk_html([("Maps", _rt(_h3("Ranked"), _p("General ranked prose, no rotation")))])
    r = B.parse_boosted(html, JUNE_URL)
    assert r.layout_warning and not r.rotations
    assert "layout may have changed" in r.note


def test_no_layout_warning_when_ranked_section_absent():
    html = _mk_html([("Bug Fixes", _rt(_p("fixed things")))])
    r = B.parse_boosted(html, JUNE_URL)
    assert not r.layout_warning and not r.rotations


def test_unresolved_name_is_reported():
    html = _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Zzzznotreal"]))])
    r = B.parse_boosted(html, JUNE_URL)
    assert "Zzzznotreal" in r.unresolved


# --- fingerprint / change detection ---------------------------------------------

def test_fingerprint_stable_and_content_sensitive():
    a = B.parse_boosted(_mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry"]))]), JUNE_URL)
    b = B.parse_boosted(_mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry"]))]), JUNE_URL)
    c = B.parse_boosted(_mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Tara"]))]), JUNE_URL)
    assert a.fingerprint == b.fingerprint and len(a.fingerprint) == 16
    assert a.fingerprint != c.fingerprint


@contextmanager
def _committed_at(doc):
    """Point both the scraper and the reference loader at a temp ranked_boosted.json."""
    old_b, old_r = B.BOOSTED_PATH, R.RANKED_BOOSTED_PATH
    R._ranked_boosted_doc.cache_clear()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ranked_boosted.json"
        if doc is not None:
            p.write_text(json.dumps(doc), encoding="utf-8")
        B.BOOSTED_PATH = R.RANKED_BOOSTED_PATH = p
        try:
            yield p
        finally:
            B.BOOSTED_PATH, R.RANKED_BOOSTED_PATH = old_b, old_r
            R._ranked_boosted_doc.cache_clear()


def test_has_changed_detects_new_rotation():
    report = B.parse_boosted(
        _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]))]), JUNE_URL)
    with _committed_at({"active": {"season": "Season 1", "brawlers": ["Berry", "Tara", "Meg"]},
                        "upcoming": []}):
        assert not B.has_changed(report)
    with _committed_at({"active": {"season": "Season 1", "brawlers": ["Shelly", "Colt", "Nita"]},
                        "upcoming": []}):
        assert B.has_changed(report)
    with _committed_at(None):
        assert B.has_changed(report)  # no committed file yet


def test_write_document_roundtrips_and_refuses_unresolved():
    good = B.parse_boosted(
        _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]))]), JUNE_URL)
    with _committed_at(None) as p:
        B.write_document(good)
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert doc["active"]["brawlers"] == ["Berry", "Tara", "Meg"]
        assert doc["valid_until"] is None
    bad = B.parse_boosted(
        _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Zzzznotreal"]))]), JUNE_URL)
    with _committed_at(None):
        try:
            B.write_document(bad)
            raise AssertionError("expected refusal to write an unmatched name")
        except ValueError:
            pass


# --- PR rendering (inherits patchnotes' _cell/_safe_url hardening) ---------------

def test_render_pr_has_table_marker_and_confirm_warning():
    r = _real()
    title, body = B.render_pr(r)
    assert "Season 1" in title and "Berry" in title
    assert "| Free brawlers |" in body                       # rotation table
    assert "Confirm which season is live" in body            # human-review nudge
    assert f"<!-- {B.FINGERPRINT_MARKER}:{r.fingerprint} -->" in body


def test_render_pr_flags_unresolved_names():
    r = B.parse_boosted(
        _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Zzzznotreal"]))]), JUNE_URL)
    _title, body = B.render_pr(r)
    assert "Unmatched name" in body and "Zzzznotreal" in body


# --- serving loader (reference.load_ranked_boosted) -----------------------------

def test_reference_loads_active_ids():
    with _committed_at({"active": {"season": "S", "brawlers": ["Berry", "Tara", "Meg"]}}):
        ids = R.load_ranked_boosted()
    assert R.brawler_by_name("Berry").id in ids and len(ids) == 3


def test_reference_missing_file_returns_empty():
    with _committed_at(None):
        assert R.load_ranked_boosted() == ()


def test_reference_expired_valid_until_returns_empty():
    with _committed_at({"valid_until": "2000-01-01",
                        "active": {"brawlers": ["Berry", "Tara", "Meg"]}}):
        assert R.load_ranked_boosted() == ()


def test_reference_future_valid_until_is_active():
    with _committed_at({"valid_until": "2999-12-31", "active": {"brawlers": ["Berry"]}}):
        assert len(R.load_ranked_boosted()) == 1


@contextmanager
def _serving_now(iso):
    """Pin the reference-side UTC clock (R._now_utc) so boundary behavior is testable; yields a
    mutable holder so a test can advance the clock mid-flight (warm-cache scenarios)."""
    class _Clock:
        now = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

    old = R._now_utc
    R._now_utc = lambda: _Clock.now
    try:
        yield _Clock
    finally:
        R._now_utc = old


def _at(iso):
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def test_reference_valid_until_expires_between_calls_despite_cache():
    # The deployed API is kept warm for days, so the expiry guard must run per *call*, not per
    # process: a rotation served fine on its last valid day must vanish the next morning without
    # a restart. Only the file parse (_ranked_boosted_doc) may be cached.
    with _serving_now("2026-08-18T12:00:00Z") as clock:
        with _committed_at({"valid_until": "2026-08-18", "active": {"brawlers": ["Berry"]}}):
            assert len(R.load_ranked_boosted()) == 1   # last valid day (inclusive) — still served
            clock.now = _at("2026-08-19T12:00:00Z")    # midnight passes; parse cache still warm
            assert R.load_ranked_boosted() == ()       # expired — dropped without a restart


_STAGED_DOC = {  # the committed shape: active season expiring, successor staged by date
    "valid_until": "2026-08-18",
    "active": {"season": "Season 1", "brawlers": ["Berry", "Tara", "Meg"]},
    "upcoming": [{"season": "Season 2", "active_from": "2026-08-19",
                  "brawlers": ["Trunk", "Willow", "Kaze"]}],
}


def test_reference_staged_upcoming_takes_over_on_its_date():
    # The season handover is data-staged: no restart, no midnight edit — the same warm process
    # serves Season 1 through its last day and Season 2 from the staged date.
    with _serving_now("2026-08-18T12:00:00Z") as clock:
        with _committed_at(_STAGED_DOC):
            assert R.load_ranked_boosted() == tuple(
                R.brawler_by_name(n).id for n in ["Berry", "Tara", "Meg"])
            clock.now = _at("2026-08-19T12:00:00Z")
            assert R.load_ranked_boosted() == tuple(
                R.brawler_by_name(n).id for n in ["Trunk", "Willow", "Kaze"])


def test_reference_datetime_boundaries_pin_the_hour():
    # When the in-game flip hour is known, a full ISO datetime stages it exactly: the boundary is
    # the instant, not the day — and everything is evaluated in UTC, so serving hosts in different
    # local timezones (Render=UTC, home Mac=EDT) agree on the FREE set at every moment.
    doc = dict(_STAGED_DOC, upcoming=[{"season": "Season 2", "active_from": "2026-08-19T10:00:00Z",
                                       "brawlers": ["Trunk"]}])
    with _committed_at(doc):
        with _serving_now("2026-08-19T09:59:59Z"):
            assert R.load_ranked_boosted() == ()       # gap: Season 1 over, hour not reached
        with _serving_now("2026-08-19T10:00:01Z"):
            assert len(R.load_ranked_boosted()) == 1   # Trunk, from the staged instant


def test_reference_staged_upcoming_does_not_serve_early():
    # A staged successor must not leak in while the active season still runs, nor before its own
    # start once the active one has expired (the gap serves nothing — fail-safe).
    doc = dict(_STAGED_DOC, upcoming=[{"season": "Season 2", "active_from": "2026-08-20",
                                       "brawlers": ["Trunk"]}])
    with _serving_now("2026-08-18T12:00:00Z"):
        with _committed_at(doc):
            assert len(R.load_ranked_boosted()) == 3   # Season 1, untouched by the staged entry
    with _serving_now("2026-08-19T12:00:00Z"):
        with _committed_at(doc):
            assert R.load_ranked_boosted() == ()       # gap day: expired, successor not started


def test_reference_unstaged_or_malformed_upcoming_never_serves():
    # No active_from (the scraper never writes one) or a malformed date → the upcoming entry
    # stays unserved even with the active season expired: serving early would mislead.
    for entry in ({"season": "S2", "brawlers": ["Trunk"]},
                  {"season": "S2", "active_from": "soon", "brawlers": ["Trunk"]}):
        with _serving_now("2026-08-19T12:00:00Z"):
            with _committed_at(dict(_STAGED_DOC, upcoming=[entry])):
                assert R.load_ranked_boosted() == ()


def test_reference_non_string_dates_fail_safe_not_500():
    # The file is hand-edited; a typo like an unquoted 20260818 is *valid JSON* with a non-string
    # type. That must fail SAFE in each direction — valid_until unreadable keeps active serving,
    # active_from unreadable keeps the successor unserved — never raise into /api/reference.
    doc = {"valid_until": 20260818,
           "active": {"season": "S1", "brawlers": ["Berry"]},
           "upcoming": [{"season": "S2", "active_from": 20260819, "brawlers": ["Trunk"]}]}
    with _serving_now("2026-08-25T12:00:00Z"):
        with _committed_at(doc):
            ids = R.load_ranked_boosted()              # must not raise
    assert ids == (R.brawler_by_name("Berry").id,)     # active serves; staged entry stays dark


def test_reference_grant_unions_in_outside_the_rotation():
    # A grant is a brawler made free OUTSIDE the seasonal rotation (the notes never announce it,
    # e.g. Nori 2026-08-25). It unions in on top of whatever rotation serves, gated by its own
    # 'since'/'valid_until' — independent of the season boundaries.
    doc = {"valid_until": "2026-08-19",
           "active": {"season": "S1", "brawlers": ["Berry", "Tara", "Meg"]},
           "upcoming": [{"season": "S2", "active_from": "2026-08-20", "brawlers": ["Trunk"]}],
           "grants": [{"brawler": "Nori", "since": "2026-08-25", "valid_until": None}]}
    nori = R.brawler_by_name("Nori").id
    with _committed_at(doc):
        with _serving_now("2026-08-24T12:00:00Z"):     # grant not yet live
            assert nori not in R.load_ranked_boosted()
        with _serving_now("2026-08-26T12:00:00Z"):     # grant live, on top of Season 2
            ids = R.load_ranked_boosted()
            assert nori in ids and R.brawler_by_name("Trunk").id in ids


def test_reference_grant_since_fail_safe():
    # A grant with no/malformed 'since' never serves (starting free-status early misleads).
    for since in (None, "someday", 20260825):
        with _serving_now("2026-08-26T12:00:00Z"):
            with _committed_at({"active": {"season": "S1", "brawlers": ["Berry"]},
                                "grants": [{"brawler": "Nori", "since": since}]}):
                assert R.brawler_by_name("Nori").id not in R.load_ranked_boosted()


def test_reference_promoted_upcoming_expires_when_unbounded():
    # The bug this closes: a promoted 'upcoming' entry once served FOREVER because only 'active'
    # was ever bounded. An un-ended rotation must now go quiet _MAX_UNBOUNDED_ROTATION_DAYS after
    # its start, so an unmaintained file fails safe (empty) instead of asserting a dead free set.
    doc = {"valid_until": "2026-08-19",
           "active": {"season": "S1", "brawlers": ["Berry"]},
           "upcoming": [{"season": "S2", "active_from": "2026-08-20", "brawlers": ["Trunk"]}]}
    with _committed_at(doc):
        with _serving_now("2026-09-01T12:00:00Z"):     # well within the cap — still serves
            assert R.brawler_by_name("Trunk").id in R.load_ranked_boosted()
        with _serving_now("2099-01-01T00:00:00Z"):     # long past the cap — quiet, not lying
            assert R.load_ranked_boosted() == ()


def test_reference_entry_valid_until_bounds_a_promoted_rotation():
    # An entry may carry its own explicit end, which bounds it precisely (no reliance on the cap).
    # `active` is expired here too, so once the promoted entry ends nothing serves.
    doc = {"valid_until": "2026-08-19", "active": {"season": "S1", "brawlers": ["Berry"]},
           "upcoming": [{"season": "S2", "active_from": "2026-08-20", "valid_until": "2026-08-31",
                         "brawlers": ["Trunk"]}]}
    trunk = R.brawler_by_name("Trunk").id
    with _committed_at(doc):
        with _serving_now("2026-08-31T23:00:00Z"):
            assert R.load_ranked_boosted() == (trunk,)
        with _serving_now("2026-09-01T00:30:00Z"):
            assert R.load_ranked_boosted() == ()


def test_write_document_preserves_grants_and_entry_dates():
    # A rescrape knows rotation *content*, never grants (the notes don't mention them) nor staged
    # dates — both must survive a rewrite verbatim, or a routine notes edit silently deletes a
    # live grant / staged handover.
    report = B.parse_boosted(
        _mk_html([_ranked_block(
            _season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]),
            _season("Season 2", "Brawl Ball", ["Trunk", "Willow", "Kaze"]))]), JUNE_URL)
    committed = {
        "active": {"season": "Season 1", "brawlers": ["Berry", "Tara", "Meg"]},
        "upcoming": [{"season": "Season 2", "active_from": "2026-08-20",
                      "valid_until": "2026-09-15", "brawlers": ["Trunk", "Willow", "Kaze"]}],
        "grants": [{"brawler": "Nori", "since": "2026-08-25", "valid_until": None}],
    }
    with _committed_at(committed) as p:
        B.write_document(report)
        doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["grants"] == committed["grants"]
    s2 = doc["upcoming"][0]
    assert s2["active_from"] == "2026-08-20" and s2["valid_until"] == "2026-09-15"


_ROTATION_KEYS = {"season", "brawlers", "featured_mode", "active_from", "valid_until"}
_DOC_KEYS = {"_comment", "source_url", "scraped_at", "valid_until", "active", "upcoming", "grants"}
_GRANT_KEYS = {"brawler", "since", "valid_until", "note"}


def test_committed_file_is_valid():
    # The checked-in data/reference/ranked_boosted.json: strict key schema (a typo like
    # 'active_form' silently disables a staged handover — reject unknown keys outright), every
    # date parseable, and any staged handover exercised end-to-end at its own boundaries.
    doc = json.loads(R.RANKED_BOOSTED_PATH.read_text(encoding="utf-8"))
    assert set(doc) <= _DOC_KEYS, f"unknown top-level key(s): {set(doc) - _DOC_KEYS}"
    entries = [e for e in [doc.get("active")] + list(doc.get("upcoming") or []) if e]
    for e in entries:
        assert set(e) <= _ROTATION_KEYS, f"unknown rotation key(s): {set(e) - _ROTATION_KEYS}"
    if doc.get("valid_until") is not None:
        assert R._parse_boundary(doc["valid_until"], end_of_day=True) is not None
    for e in doc.get("upcoming") or []:
        if "active_from" in e:
            assert R._parse_boundary(e["active_from"], end_of_day=False) is not None
    for g in doc.get("grants") or []:
        assert set(g) <= _GRANT_KEYS, f"unknown grant key(s): {set(g) - _GRANT_KEYS}"
        assert R._parse_boundary(g.get("since"), end_of_day=False) is not None, "grant needs a parseable 'since'"
        if g.get("valid_until") is not None:
            assert R._parse_boundary(g["valid_until"], end_of_day=True) is not None

    # A staged handover must actually hand over: active serves just before its end, the staged
    # successor serves from just after its start, and they differ.
    until = R._parse_boundary(doc.get("valid_until"), end_of_day=True)
    staged = [e for e in doc.get("upcoming") or [] if e.get("active_from")]
    if until and staged:
        start = R._parse_boundary(staged[0]["active_from"], end_of_day=False)
        with _committed_at(doc):
            with _serving_now((until - timedelta(seconds=1)).isoformat()):
                before = R.load_ranked_boosted()
            with _serving_now((start + timedelta(seconds=1)).isoformat()):
                after = R.load_ranked_boosted()
        assert before and after and set(before) != set(after)
    else:  # no flip staged — the plain active rotation must resolve
        with _committed_at(doc):
            with _serving_now(datetime.now(timezone.utc).isoformat()):
                assert R.load_ranked_boosted()


def test_write_document_carries_staged_dates_forward():
    # A boosted-watch rewrite knows rotation *content*, never dates: valid_until and a staged
    # active_from must survive a rewrite for seasons whose names still match (else a routine
    # notes edit would silently destroy a staged season handover), and drop on a rename.
    report = B.parse_boosted(
        _mk_html([_ranked_block(
            _season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]),
            _season("Season 2", "Brawl Ball", ["Trunk", "Willow", "Kaze"]))]), JUNE_URL)
    staged = {"valid_until": "2026-08-18",
              "active": {"season": "Season 1", "brawlers": ["Berry", "Tara", "Meg"]},
              "upcoming": [{"season": "Season 2", "active_from": "2026-08-19T10:00:00Z",
                            "brawlers": ["Trunk", "Willow", "Kaze"]}]}
    with _committed_at(staged) as p:
        B.write_document(report)
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert doc["valid_until"] == "2026-08-18"                       # same active season name
        assert doc["upcoming"][0]["active_from"] == "2026-08-19T10:00:00Z"
    renamed = dict(staged, active={"season": "Season 0", "brawlers": ["Berry"]},
                   upcoming=[{"season": "Season 9", "active_from": "2026-08-19",
                              "brawlers": ["Trunk"]}])
    with _committed_at(renamed) as p:
        B.write_document(report)
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert doc["valid_until"] is None                               # renamed → restage by hand
        assert "active_from" not in doc["upcoming"][0]


def test_reference_skips_unknown_names():
    with _committed_at({"active": {"brawlers": ["Berry", "Zzzznotreal"]}}):
        assert len(R.load_ranked_boosted()) == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
