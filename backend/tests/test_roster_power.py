"""The Ranked power-level gate: an owned brawler below the bracket's power floor can't be fielded,
so it must be dropped from the personalized roster before it's ever recommended.

Ranked doesn't normalize brawlers to a fixed power — each bracket hard-blocks selecting a brawler
below a per-brawler floor: Power 9 through Diamond, Power 11 from Mythic up. Recommending one the
player literally cannot pick (e.g. an un-maxed Bibi in Legendary) is the bug this guards.

    PYTHONPATH=backend python -m pytest backend/tests/test_roster_power.py    # or run directly
"""
from __future__ import annotations

import re
import types
from pathlib import Path

import bsdraft.api.main as M
from bsdraft.api import schemas as S
from bsdraft.engine.tiers import min_power_for_bracket

# Synthetic ids well outside any real brawler / boosted-rotation id, so the boosted brawlers the
# gate folds in can never collide with the ones under assertion here.
A11, B9, C10, D0, E7 = 90000011, 90000009, 90000010, 90000000, 90000007


def _roster(bracket, entries, personalize=True):
    roster = [S.OwnedBrawler(id=i, mastery=0.5, power=p) for i, p in entries]
    req = S.RecommendRequest(map_id=1, mode="Brawl Ball", rank_bracket=bracket,
                             personalize=personalize, roster=roster)
    return M._roster_for(req)


# --- the pure floor helper -------------------------------------------------------

def test_min_power_per_bracket():
    for b in ("Mythic", "Legendary", "Masters", "Pro"):
        assert min_power_for_bracket(b) == 11, b
    for b in ("Bronze", "Silver", "Gold", "Diamond"):
        assert min_power_for_bracket(b) == 9, b
    # unknown / unset → the universal Ranked floor, never a hidden Power-11 gate
    assert min_power_for_bracket(None) == 9
    assert min_power_for_bracket("Nonsense") == 9


# --- the gate in _roster_for -----------------------------------------------------

def test_p11_bracket_drops_under_eleven():
    # Legendary: only the Power-11 brawler survives; Power 9 and Power 10 are unfieldable.
    r = _roster("Legendary", [(A11, 11), (B9, 9), (C10, 10)])
    assert A11 in r and B9 not in r and C10 not in r


def test_mythic_is_also_power_eleven():
    # The floor jumps to 11 at Mythic, not Legendary — a Power-9 brawler is blocked there too.
    r = _roster("Mythic", [(A11, 11), (B9, 9)])
    assert A11 in r and B9 not in r


def test_low_bracket_keeps_power_nine():
    # Diamond: floor is 9, so Power 9/10/11 all field; only below 9 is dropped.
    r = _roster("Diamond", [(A11, 11), (B9, 9), (C10, 10), (E7, 7)])
    assert A11 in r and B9 in r and C10 in r and E7 not in r


def test_unknown_power_is_kept():
    # Power 0 == "roster didn't report it" (older client): keep it rather than hide on missing data.
    r = _roster("Legendary", [(D0, 0)])
    assert D0 in r


def test_unset_bracket_uses_universal_floor():
    r = _roster(None, [(B9, 9), (E7, 7)])
    assert B9 in r and E7 not in r


def test_personalize_off_returns_none():
    assert _roster("Legendary", [(A11, 11)], personalize=False) is None


# --- sent-but-empty vs omitted: the cross-visitor leak guard ---------------------

class _ServerMastery:
    """Engine-side Mastery stand-in: just the fields the fallback branch reads."""
    power = 11
    score = 0.9

    def gaps(self):
        return []


def test_empty_roster_never_falls_back_to_server_roster():
    # roster == [] means "sent, and nothing fieldable" — the client's power-floor filter can empty
    # a real roster. On the keyed host _engine.roster is whatever /api/roster fetched LAST (another
    # visitor), so falling through on [] scores one player's draft against another player's roster.
    stranger = 90000042
    prev = M._engine  # None under tests — the startup event that builds it never ran
    M._engine = types.SimpleNamespace(roster={stranger: _ServerMastery()})
    try:
        r = _roster("Legendary", [])
        assert r is not None, "personalize stays on for an empty roster"
        assert stranger not in r, "another visitor's roster leaked into personalization"
        from bsdraft.data import reference as R
        assert set(r) == set(R.load_ranked_boosted())  # only the free boosted brawlers field
    finally:
        M._engine = prev


def test_omitted_roster_falls_back_to_server_roster():
    # roster=None (field omitted — a client with no roster of its own): the server's roster is the
    # intended fallback for the local/home operator.
    mine = 90000043
    prev = M._engine
    M._engine = types.SimpleNamespace(roster={mine: _ServerMastery()})
    try:
        req = S.RecommendRequest(map_id=1, mode="Brawl Ball", rank_bracket="Legendary",
                                 personalize=True, roster=None)
        assert mine in M._roster_for(req)
    finally:
        M._engine = prev


def test_boosted_brawlers_are_added_and_clear_the_floor():
    # Boosted (free) brawlers arrive at Power 11 and are folded in after the gate, so they're always
    # recommendable regardless of bracket — while an owned sub-floor brawler beside them is dropped.
    from bsdraft.data import reference as R
    boosted = R.load_ranked_boosted()
    r = _roster("Legendary", [(B9, 9)])
    assert B9 not in r
    for bid in boosted:
        assert bid in r


def test_owned_boosted_brawler_is_priced_as_fully_maxed():
    """Ranked loans a free/boosted brawler fully maxed to *owners* too, so an owned copy that is
    under-levelled or half-built must be priced as ready with no gaps — never docked a readiness
    deficit or flagged with a loadout gap for a copy the season hands out maxed, while its real
    display score is preserved for the roster UI. Regression guard for the old ``setdefault`` that
    kept an owned free brawler's real (deficient) mastery — which made owning a weak copy recommend
    it *worse* than not owning it at all."""
    from bsdraft.engine.readiness import GAP_NO_STAR_POWER, readiness
    free = 90000055
    saved = M._free_brawler_ids
    M._free_brawler_ids = lambda: (free,)
    try:
        # Owned at Power 9 with a missing star power, in a floor-9 bracket so the entry SURVIVES the
        # power gate — the deficit path only bites above the floor, which is exactly where the bug
        # charged an owned boosted copy.
        roster = [S.OwnedBrawler(id=free, mastery=0.42, power=9, gaps=[GAP_NO_STAR_POWER])]
        req = S.RecommendRequest(map_id=1, mode="Brawl Ball", rank_bracket="Diamond",
                                 personalize=True, roster=roster)
        m = M._roster_for(req)[free]
        assert m.gaps() == []                     # no spurious "no star power" chip
        assert readiness(m.fielded())[0] == 0.0   # no deficit — it's the maxed loan
        assert m.score == 0.42                     # real comfort/build preserved for display

        # The unowned case is unchanged: no history, so the display score keeps its 0.60 default.
        req2 = S.RecommendRequest(map_id=1, mode="Brawl Ball", rank_bracket="Diamond",
                                  personalize=True, roster=[])
        assert M._roster_for(req2)[free].score == 0.60
    finally:
        M._free_brawler_ids = saved


# --- the wire contract: ``power`` must actually arrive ---------------------------
#
# Every gate test above hands ``_roster_for`` a roster that HAS a power on each entry. None of them
# would notice the one failure mode that disables the gate entirely: the client no longer sending
# ``power`` at all. ``OwnedBrawler.power`` defaults to 0 and 0 means "unknown, keep it", so a
# slimmed-down payload doesn't error or warn — it just makes every entry fieldable. The pair below
# pins that: the first states the consequence, the second guards the client that must not cause it.

def test_missing_power_defeats_the_gate():
    """Characterization, not an endorsement: an entry with no ``power`` clears any floor.

    This is exactly what a "clean up the recommend payload" change would produce, and why the
    guard below exists — nothing else in this suite fails when ``power`` stops arriving."""
    naked = [S.OwnedBrawler(id=B9, mastery=0.5), S.OwnedBrawler(id=E7, mastery=0.5)]
    req = S.RecommendRequest(map_id=1, mode="Brawl Ball", rank_bracket="Legendary",
                             personalize=True, roster=naked)
    r = M._roster_for(req)
    assert B9 in r and E7 in r, "power-less entries pass the floor — the gate is a no-op"


# The client half of the contract. /api/recommend can't tell a slimmed payload from an old client,
# so the only place this regression is catchable is the source that builds the payload.
_DRAFT_BOARD = Path(__file__).resolve().parents[2] / "frontend" / "components" / "DraftBoard.tsx"


def _const_block(src, name):
    """Yield each full `const <name> = ...;` definition found in DraftBoard.tsx.

    Accumulates lines from a declaration until its parentheses balance, so a multi-line useMemo
    comes back whole."""
    lines = src.splitlines()
    decl = re.compile(r"\bconst\s+" + re.escape(name) + r"\s*=")
    for i, ln in enumerate(lines):
        if not decl.search(ln):
            continue
        block, depth = [], 0
        for cur in lines[i:]:
            block.append(cur)
            depth += cur.count("(") - cur.count(")")
            if depth <= 0 and "(" in "".join(block):
                break
        yield "\n".join(block)


def _payload_bindings(src):
    """(what DraftBoard.tsx POSTs as `roster`, {binding name -> its definition}).

    The binding is resolved from the payload sites rather than matched by name, so renaming it
    doesn't fail this test while changing what it builds does. Identifiers that merely appear in a
    payload expression but aren't built from the roster (the `myTurn` in `myTurn ? x : null`) fall
    out on their own: only definitions that read `roster?.owned` are kept."""
    sent = re.findall(r"\broster:\s*([^,\n]+)", src)
    idents = {m for expr in sent for m in re.findall(r"[A-Za-z_$][\w$]*", expr)}
    bindings = {}
    for name in sorted(idents - {"null", "undefined", "true", "false"}):
        for block in _const_block(src, name):
            if "roster?.owned" in block:
                bindings[name] = block
                break
    return sent, bindings


def test_recommend_payload_still_carries_power():
    """The recommend body must POST whole /api/roster entries, never a narrowed projection.

    ``power`` is what the backend's floor gate reads; drop it from the payload and the gate silently
    passes everything (see ``test_missing_power_defeats_the_gate``). Rather than grep for the word
    ``power`` — which matches the unrelated ``powerFloor`` and would pass for the wrong reason — this
    checks the property that actually implies it: the payload is built from ``roster?.owned`` by
    ``.filter()`` alone, with nothing that reshapes an entry. ``.filter()`` preserves element
    identity, so what goes on the wire is the /api/roster object whole, ``power`` included. The
    item-ownership fields ride along the same way; the backend ignores those, but they cost nothing,
    and slimming them is exactly what would take ``power`` with them.

    This is a source-shape guard, so it catches the realistic regression (a projection introduced to
    trim the payload), not every conceivable one — a hand-rolled loop that rebuilt entries would slip
    past it."""
    if not _DRAFT_BOARD.exists():
        print(f"skip: {_DRAFT_BOARD} not present (backend checked out without the frontend)")
        return
    src = _DRAFT_BOARD.read_text()
    sent, bindings = _payload_bindings(src)

    assert sent, "nothing is POSTed as `roster` from DraftBoard.tsx any more"
    assert bindings, (
        f"none of the values POSTed as `roster` ({', '.join(e.strip() for e in sent)}) is built from "
        f"`roster?.owned` — the recommend payload must be the roster itself, or `power` never "
        f"reaches the backend's floor gate"
    )

    # `=> ({` catches any arrow that returns a fresh object — .map, .flatMap, a reduce — while
    # `.map(` catches the plain projection even when it's written across lines.
    reshapes = lambda s: ".map(" in s or "=> ({" in s or "=> Object" in s

    for name, block in bindings.items():
        assert ".filter(" in block, (
            f"`{name}` is no longer a plain `.filter()` over the roster, so the entries POSTed to "
            f"/api/recommend may no longer be whole /api/roster objects:\n{block}"
        )
        assert not reshapes(block), (
            f"`{name}` reshapes roster entries instead of passing them through — if the new shape "
            f"omits `power`, the backend's Ranked power-floor gate silently becomes a no-op "
            f"(every entry defaults to power 0 and clears the floor):\n{block}"
        )

    for expr in sent:
        assert any(name in expr for name in bindings), (
            f"`roster: {expr.strip()}` doesn't send the roster binding "
            f"({'/'.join(bindings)}) — `power` must stay on the wire"
        )
        assert not reshapes(expr), (
            f"`roster: {expr.strip()}` reshapes the payload inline — `power` must stay on the wire"
        )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
