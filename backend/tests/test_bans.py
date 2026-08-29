"""Ban valuation tests: the ban list must react to the ban set, not just filter it.

The property under test is the one a static threat table can't have — what a ban is worth
depends on what else is already banned, on who picks first (bans are global in Brawl Stars),
and on what the player can actually field. These run on a synthetic 10-brawler pool with a
hand-checkable model so the expected direction of every effect is known in advance.

    PYTHONPATH=backend python -m pytest backend/tests/test_bans.py   # or run directly
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

from bsdraft.engine import bans as B
from bsdraft.engine.state import DraftState

# A gently tapering pool, as a live map's meta looks. Brawlers 2 and 3 do the same job, so a
# team holding one gets little from the other; 4 is the strongest brawler with no stand-in.
STRENGTH = {1: 0.95, 2: 0.90, 3: 0.89, 4: 0.88, 5: 0.86, 6: 0.85, 7: 0.83,
            8: 0.82, 9: 0.80, 10: 0.78}
ARCHETYPE = {2: "A", 3: "A"}          # everything else is its own archetype
DUP = 0.4                              # what a second brawler of a covered archetype is worth
IDS = sorted(STRENGTH)


@dataclass
class _Brawler:
    id: int
    name: str
    cls: str


class _Ref:
    """Stand-in for data.reference — a fixed synthetic roster of brawlers (all released)."""
    def load_brawlers(self):
        return [_Brawler(i, f"B{i}", "Damage Dealer") for i in IDS]

    # bans.recommend() draws its pool from pickable_brawlers(); here every synthetic brawler is
    # released, so it mirrors load_brawlers().
    def pickable_brawlers(self):
        return self.load_brawlers()


class _Rate:
    def __init__(self, winrate):
        self.winrate = winrate
        self.raw_winrate = winrate
        self.games = 100.0
        self.confidence = 0.8


class _Stats:
    """Empirical rates ordered the same way as model strength, so threat order is stable."""
    def brawler_rate(self, brawler_id, map_id=None):
        return _Rate(0.50 + 0.05 * STRENGTH.get(brawler_id, 0.0))

    def use_rate(self, brawler_id, map_id):
        return 0.10


def _team_strength(team):
    """Strength with diminishing returns on redundancy — the second brawler covering a job
    adds little. The real net gets this from mean-pooling brawler embeddings (two similar
    brawlers move the team vector almost nowhere); this is the hand-checkable version, and it
    is why a team's value can't be read off a per-brawler ladder."""
    total, seen = 0.0, set()
    for x in sorted(team, key=lambda i: -STRENGTH.get(i, 0.0)):
        arch = ARCHETYPE.get(x, x)
        total += STRENGTH.get(x, 0.0) * (DUP if arch in seen else 1.0)
        seen.add(arch)
    return total


class _Model:
    """Stand-in for the win-prob net: logistic of the two teams' strength difference."""
    available = True
    supports_partial = True

    def __init__(self):
        self.boards = 0

    def prob_batch(self, teams_a, teams_b, map_id, mode):
        self.boards += len(teams_a)
        return [1.0 / (1.0 + math.exp(-(_team_strength(a) - _team_strength(b))))
                for a, b in zip(teams_a, teams_b)]

    def prob(self, a, b, map_id, mode):
        return self.prob_batch([list(a)], [list(b)], map_id, mode)[0]


class _Legacy(_Model):
    """Pre-mask artifact: only judges finished 3v3s."""
    supports_partial = False

    def prob_batch(self, teams_a, teams_b, map_id, mode):
        for t in list(teams_a) + list(teams_b):
            if len(t) != 3:
                raise ValueError(f"legacy artifact got a {len(t)}-brawler team")
        return super().prob_batch(teams_a, teams_b, map_id, mode)


@contextmanager
def _synthetic(survival=False):
    """Swap in the synthetic reference/name map. By default the enemy's unseen bans are switched
    off (every brawler certain to survive) so a test sees the denial projection on its own."""
    saved = (B.R, B._name_map, B.MIN_SURVIVAL)
    B.R = _Ref()
    B._name_map = lambda: {i: f"B{i}" for i in IDS}
    if not survival:
        B.MIN_SURVIVAL = 1.0        # floors every brawler's survival odds at certainty
    try:
        yield
    finally:
        B.R, B._name_map, B.MIN_SURVIVAL = saved


def _rank(state, **kw):
    """{brawler_id: BanScore} for the whole synthetic pool."""
    rows = B.recommend(state, _Stats(), kw.pop("model", None) or _Model(), top=len(IDS), **kw)
    return {r.brawler_id: r for r in rows}, [r.brawler_id for r in rows]


def _state(**kw):
    return DraftState(map_id=1, mode="Brawl Ball", **kw)


def test_substitute_absorbs_the_ban():
    """The headline property: banning one of a pair is worth less than banning it once its
    partner is gone. Nothing about brawler 2 changed — only what's left to replace it."""
    with _synthetic():
        alone, _ = _rank(_state())                      # 3 still available to cover for 2
        after, _ = _rank(_state(bans=[3]))              # a teammate already banned 3
        assert after[2].ban_value > alone[2].ban_value, (alone[2].ban_value, after[2].ban_value)


def test_prior_bans_change_the_gaps_not_just_the_rows():
    """A teammate's ban has to move the *spread* of what's left, not merely delete a row.

    Brawler 2 is covered by its substitute and 4 is not, so they sit close together. Banning 3
    uncovers 2 and the gap between them widens sharply — while both brawlers' standalone threat,
    all a static table can see, is bit-for-bit unchanged."""
    with _synthetic():
        before, _ = _rank(_state())
        after, _ = _rank(_state(bans=[3]))
    assert before[2].threat == after[2].threat and before[4].threat == after[4].threat
    assert after[2].ban_value / after[4].ban_value > 2 * (before[2].ban_value / before[4].ban_value)


def test_first_pick_discounts_your_own_pick():
    """Bans are global, so who holds the early slot decides who a ban really hurts. With first
    pick, brawler 1 is ours to take and banning it is partly self-denial; hand the same board to
    the second-pick seat and the identical ban is worth strictly more."""
    with _synthetic():
        first, _ = _rank(_state(we_pick_first=True))
        second, _ = _rank(_state(we_pick_first=False))
    assert first[1].ban_value < second[1].ban_value
    assert first[1].self_deny and not second[1].self_deny


def test_your_own_pick_stays_on_the_board():
    """A brawler the draft hands us ranks below every real ban — but it must still appear,
    flagged. On a map with one obvious threat, silently dropping it reads as an oversight
    rather than as the advice it is."""
    with _synthetic():
        rows = B.recommend(_state(we_pick_first=True), _Stats(), _Model(), top=4)
        flagged = [r for r in rows if r.self_deny]
        assert flagged, [r.name for r in rows]
        assert rows[-1].self_deny            # carried, but last
        assert not rows[0].self_deny         # never the recommendation


def test_unowned_brawlers_are_free_to_ban():
    """A brawler the player can't field can't be one of our picks, so banning it never costs
    us — it stops being self-denial the moment the roster says we don't own it."""
    with _synthetic():
        owned = {i: object() for i in IDS if i != 1}     # everything except brawler 1
        rows, _ = _rank(_state(we_pick_first=True), roster=owned)
        assert rows[1].self_deny is False
        base, _ = _rank(_state(we_pick_first=True))
        assert rows[1].ban_value > base[1].ban_value


def test_replacement_is_reported():
    with _synthetic():
        rows, order = _rank(_state())
        top = rows[order[0]]
        assert top.replacement and top.replacement != top.name


def test_you_can_only_ever_see_your_teammates_bans():
    """You never see the enemy's three bans before choosing yours, and the only bans that can
    reach the board first are your two teammates' — so all three stay priced as unseen for the
    whole ban phase, however many are showing. The count only falls once the reveal puts more
    than two on the board."""
    seen = [B._hidden_enemy_bans(_state(bans=list(range(n)))) for n in range(7)]
    assert seen[:3] == [3, 3, 3], seen      # nothing you can see is an enemy ban
    assert seen[3:] == [2, 1, 0, 0], seen   # post-reveal, they come off one at a time


def test_survival_demotes_the_ban_they_were_making():
    """Their unseen bans land on the same brawlers we'd target. Weighting a brawler by the odds
    it survives them prices two things with one number: their comps get built from the board
    that will actually exist, and a ban they were always going to make stops being worth
    spending ours on — no separate discount rule."""
    with _synthetic(survival=False):
        plain, _ = _rank(_state())
    with _synthetic(survival=True):
        priced, _ = _rank(_state())
    gaining = [i for i in IDS if plain[i].ban_value > 0 and priced[i].ban_value > 0]
    hottest = max(gaining, key=lambda i: plain[i].threat)
    coldest = min(gaining, key=lambda i: plain[i].threat)
    # It bites the obvious target hardest — not a flat scale-down of the whole list.
    kept = lambda i: priced[i].ban_value / plain[i].ban_value   # noqa: E731
    assert kept(hottest) < kept(coldest)


def test_values_stay_bounded_and_lead_with_a_real_ban():
    """Every value is a win-prob delta — bounded, positive at the top of the list, and never
    led by a brawler the draft projects onto our own side."""
    with _synthetic():
        rows, order = _rank(_state())
        assert all(-1.0 <= r.ban_value <= 1.0 for r in rows.values())
        assert rows[order[0]].ban_value > 0
        assert not rows[order[0]].self_deny


def test_survival_stops_mattering_once_every_ban_is_on_the_board():
    """With the whole ban set revealed there is nothing left to survive, so the projection runs
    on the pool exactly as it stands."""
    revealed = _state(bans=[7, 8, 9, 10, 6])
    with _synthetic(survival=True):
        priced, _ = _rank(revealed)
    with _synthetic(survival=False):
        plain, _ = _rank(revealed)
    assert all(abs(priced[i].ban_value - plain[i].ban_value) < 1e-12 for i in priced)


def test_no_model_falls_back_to_threat_order():
    with _synthetic():
        rows = B.recommend(_state(), _Stats(), None, top=6)
        assert [r.brawler_id for r in rows] == sorted(IDS, key=lambda i: -STRENGTH[i])[:6]
        assert all(r.ban_value is None and not r.self_deny for r in rows)

        class _Down(_Model):
            available = False
        assert all(r.ban_value is None for r in B.recommend(_state(), _Stats(), _Down(), top=6))


def test_legacy_artifact_keeps_threat_order():
    """Every quantity in the projection is a team read against an unknown board, which a
    pre-mask artifact can't express — so it keeps the old ordering instead of being handed
    partial teams it would reject."""
    with _synthetic():
        rows = B.recommend(_state(), _Stats(), _Legacy(), top=6)   # _Legacy raises on partials
        assert [r.brawler_id for r in rows] == sorted(IDS, key=lambda i: -STRENGTH[i])[:6]
        assert all(r.ban_value is None for r in rows)


def test_banned_and_picked_brawlers_are_excluded():
    with _synthetic():
        rows, order = _rank(_state(bans=[2], our_team=[1], their_team=[4]))
        assert not ({1, 2, 4} & set(order))


def test_projection_cost_is_bounded():
    """Every ban placement re-runs this, so the evaluation count must stay inside the ban-phase
    latency budget: one batch of lone brawlers and one of comps, not a search."""
    with _synthetic(survival=True):
        model = _Model()
        B.recommend(_state(), _Stats(), model, top=6)
        # C(10,3) comps + 10 solo reads on this pool; the real pool caps at C(12,3) + 20.
        assert model.boards <= 140, model.boards


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
