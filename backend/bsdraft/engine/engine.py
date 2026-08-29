"""Draft engine: ban recommendation, candidate pick recommendation, composition meter."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from bsdraft.constants import BRAWLER_CLASSES
from bsdraft.data import reference as R
from bsdraft.engine.bans import BanScore
from bsdraft.engine.scoring import PickScore, _class_of, model_marginals, score_candidate
from bsdraft.engine.state import DraftState
from bsdraft.engine.stats import PRIOR, DraftStats
from bsdraft.engine import bans as bans_mod
from bsdraft.engine import composition as composition_mod
from bsdraft.engine import gameplan as gameplan_mod
from bsdraft.engine import itemstats as itemstats_mod
from bsdraft.engine import purchases as purchases_mod
from bsdraft.models.serve import WinProbModel

# Minimum evidence before a brawler may be *recommended*. A candidate whose global effective
# (recency-weighted) game count is below this can only be scored from priors — a baseline-shrunk
# map winrate, a class-level role prior, and an untrained / mean model embedding — so ranking it
# is a guess dressed up as a number, not a recommendation (the board that motivated this showed a
# 0-game brawler on top at "CONFIDENCE 0%"; both an unreleased catalog leak AND a brand-new
# released brawler the crawler hasn't logged yet hit this). The floor is the stats smoothing
# ``PRIOR``: below it, the neutral prior still outweighs the brawler's own record. MIN_TABLE_MATCHES
# guards a cold start — below it the whole table is too thin for "unseen" to mean anything, so
# nothing is gated and a rebuilding backend still shows a full board.
MIN_RECO_GAMES = float(PRIOR)
MIN_TABLE_MATCHES = 5000


class DraftEngine:
    def __init__(self, stats: Optional[DraftStats] = None, model: Optional[WinProbModel] = None,
                 bracket_stats: Optional[Dict[str, DraftStats]] = None):
        self.stats = stats if stats is not None else DraftStats()
        self.bracket_stats: Dict[str, DraftStats] = bracket_stats or {}
        self.model = model
        self.roster = None        # dict[brawler_id, Mastery] when a player roster is loaded
        self.roster_name = ""

    def _stats_for(self, state: DraftState) -> DraftStats:
        """The requested rank-bracket table when it exists, else the global stats."""
        return self.bracket_stats.get(state.rank_bracket, self.stats)

    def candidates(self, state: DraftState, roster=None) -> List[int]:
        used = state.picked_or_banned()
        # pickable_brawlers() (not load_brawlers()) so an unreleased catalog entry is never
        # recommended: with no match data its signals sit at a neutral prior and it would
        # otherwise float to the top of an empty board. The roster path already excludes it (it
        # is neither owned nor free), but the no-roster / meta path relies on this filter.
        ids = [b.id for b in R.pickable_brawlers() if b.id not in used]
        if roster is not None:
            ids = [i for i in ids if i in roster]  # only brawlers the player owns
        return ids

    def recommend_bans(self, state: DraftState, top: int = 6, roster=None) -> List[BanScore]:
        """Rank bans by what they deny given the board — see `engine/bans.py` for the
        projection and why the list re-ranks as the ban set fills in."""
        return bans_mod.recommend(state, self._stats_for(state), self.model, top=top, roster=roster)

    def _data_backed(self, cands: List[int]) -> List[int]:
        """Drop candidates the collected data has essentially never seen, so a brawler with no
        record is never *recommended* from priors alone (see MIN_RECO_GAMES). Gates on the GLOBAL
        table — "have we observed this brawler at all" is a global question, independent of which
        rank-bracket table scores it or which map is up. Never returns empty: on a thin / rebuilding
        table (``n < MIN_TABLE_MATCHES``) or the pathological all-thin case it falls back to the
        ungated list, so a cold start still shows a full board rather than nothing."""
        if self.stats.n < MIN_TABLE_MATCHES:
            return cands
        kept = [c for c in cands if self.stats.brawler_rate(c, None).games >= MIN_RECO_GAMES]
        return kept or cands

    def recommend_picks(self, state: DraftState, top: int = 10, weights=None, roster=None,
                        personal=None) -> List[PickScore]:
        stats = self._stats_for(state)
        cands = self._data_backed(self.candidates(state, roster))
        # One batched model pass over all candidates (they share the enemy team), then score.
        # Bit-for-bit identical to computing each marginal inside score_candidate.
        win_probs = model_marginals(state, cands, self.model, stats)
        scored = [score_candidate(state, c, stats, self.model, weights, roster, personal, win_prob=wp)
                  for c, wp in zip(cands, win_probs)]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top]

    def recommend_purchases(self, owned: Dict[int, "purchases_mod.OwnedState"],
                            top: int = 20, rank_bracket: Optional[str] = None,
                            power_floor: Optional[int] = None, min_per_kind: int = 0) -> List[dict]:
        """Rank a player's most efficient next purchases from their ownership snapshot and Ranked
        bracket (which sets the power floor and picks the bracket's stats table, like a draft).
        Delegates to :mod:`bsdraft.engine.purchases`, feeding it the stats + item win-rate table
        (the table degrades to None when unbuilt, so the advisor falls back to economy priors)."""
        stats = self.bracket_stats.get(rank_bracket, self.stats) if rank_bracket else self.stats
        return purchases_mod.recommend_purchases(
            owned, stats, itemstats=itemstats_mod.get_itemstats(), top=top,
            rank_bracket=rank_bracket, power_floor=power_floor, min_per_kind=min_per_kind)

    def composition_report(self, state: DraftState) -> dict:
        return composition_mod.analyze(state)

    def game_plan(self, state: DraftState) -> dict:
        """Post-draft plan. Gets the same bracket stats table and model the pick board scores
        with, so the data-backed half of the plan reads from exactly what ranked the picks."""
        return gameplan_mod.game_plan(state, self._stats_for(state), self.model)

    def composition(self, state: DraftState) -> dict:
        counts = Counter(_class_of(b) for b in state.our_team)
        return {cls: counts.get(cls, 0) for cls in BRAWLER_CLASSES if counts.get(cls, 0)}


def _demo() -> None:
    stats = DraftStats()
    model = WinProbModel()
    engine = DraftEngine(stats, model)
    by_name = {b.name.lower(): b.id for b in R.load_brawlers()}

    bb_maps = [m for m in R.load_ranked_maps() if m.mode == "Brawl Ball" and stats.map_games.get(m.id, 0) > 0]
    mp = max(bb_maps, key=lambda m: stats.map_games.get(m.id, 0))
    print(f"Map: {mp.name} ({mp.mode})   model={'ON' if model.available else 'OFF'}   "
          f"games={stats.map_games.get(mp.id, 0)}")

    def show_bans(title, st):
        print(f"\n{title}")
        for b in engine.recommend_bans(st, top=6):
            swing = f"{b.ban_value:+.4f}" if b.ban_value is not None else "  n/a "
            note = " (ours!)" if b.self_deny else (f" -> {b.replacement}" if b.replacement else "")
            print(f"  {b.name:<14} {b.cls:<14} swing={swing} threat={b.threat:.3f} "
                  f"map_wr={b.map_winrate:.3f} use={b.use_rate:.0%}{note}")

    # The same map twice: banning the top target re-ranks what's left, because the survivors'
    # substitutes changed. A static threat table would just shift everything up one row.
    empty = DraftState(map_id=mp.id, mode=mp.mode)
    show_bans("Top ban suggestions (swing = projected gain in our win prob):", empty)
    first = engine.recommend_bans(empty, top=1)
    if first:
        show_bans(f"After banning {first[0].name} — note the re-ranking, not just the removal:",
                  DraftState(map_id=mp.id, mode=mp.mode, bans=[first[0].brawler_id]))

    def show(title, st):
        print(f"\n{title}")
        for s in engine.recommend_picks(st, top=6):
            extra = ""
            if s.synergy is not None:
                extra += f" syn={s.synergy:.3f}"
            if s.counter is not None:
                extra += f" cnt={s.counter:.3f}"
            if s.win_prob is not None:
                extra += f" model={s.win_prob:.3f}"
            print(f"  {s.name:<14} {s.cls:<14} score={s.score:.3f} map_wr={s.map_winrate:.3f} "
                  f"role={s.role_fit:.2f}{extra} conf={s.confidence:.2f}")

    show("First-pick suggestions (nothing on the board):",
         DraftState(map_id=mp.id, mode=mp.mode, we_pick_first=True))

    enemies = [by_name.get(n) for n in ("edgar", "mortis")]
    allies = [by_name.get("gene")]
    bans = [by_name.get(n) for n in ("spike", "surge")]
    st2 = DraftState(
        map_id=mp.id, mode=mp.mode, we_pick_first=False,
        our_team=[a for a in allies if a], their_team=[e for e in enemies if e],
        bans=[b for b in bans if b],
    )
    show("We pick (ally: Gene | enemies: Edgar, Mortis | banned: Spike, Surge):", st2)
    print("\nOur composition:", engine.composition(st2) or "(empty)")


if __name__ == "__main__":
    _demo()
