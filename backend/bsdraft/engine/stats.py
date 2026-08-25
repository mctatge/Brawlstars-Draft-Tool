"""Empirical draft statistics from collected matches.

Per-map and global brawler win/use rates, pair synergies, and matchup counters, each with
Bayesian (shrink-to-0.5) smoothing so thin samples aren't over-trusted, and exponential
recency weighting so the numbers track the live meta across balance changes. This is the
robust, interpretable signal the engine fuses with the learned model, and the source of
the "why this pick" explanations and confidence indicators.

Recency: each match is weighted by ``0.5 ** (age / half_life)``, with age measured from the
newest match in the dataset (deterministic, independent of when the build runs). With the
default ~3-week half-life a month-old game counts about a third of a fresh one, and brawlers
that fall out of rotation after a nerf shed effective sample — and thus confidence — on their
own. This mirrors the recency weighting used to train the win-prob model. Pass
``halflife_days <= 0`` to disable it (uniform weighting), e.g. for backtests.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from bsdraft.data import reference as R
from bsdraft.data.dataset import iter_matches, recent_matches
from bsdraft.engine.tiers import match_bracket

PRIOR = 20.0                   # pseudo-games at 0.5 used for smoothing
DEFAULT_HALFLIFE_DAYS = 21.0   # matches fade to half weight every ~3 weeks
MIN_BRACKET_MATCHES = 300      # only build a per-bracket table above this many matches
RECENT_WINDOW_DAYS = 3.0       # "is this map in the live rotation right now" horizon
_DAY = 86400.0


@dataclass
class Rate:
    games: float         # effective (recency-weighted) sample size
    winrate: float       # smoothed toward 0.5
    raw_winrate: float
    confidence: float    # games / (games + PRIOR)  in [0, 1)


def _rate(wins: float, games: float, prior: float = PRIOR, prior_rate: float = 0.5) -> Rate:
    """Smooth ``wins/games`` toward ``prior_rate`` (0.5 globally, or the global rate when a
    bracket table backs off)."""
    raw = wins / games if games else prior_rate
    smoothed = (wins + prior * prior_rate) / (games + prior)
    return Rate(games=games, winrate=smoothed, raw_winrate=raw, confidence=games / (games + prior))


class DraftStats:
    def __init__(
        self,
        matches: Optional[Iterable[dict]] = None,
        halflife_days: float = DEFAULT_HALFLIFE_DAYS,
        bracket: Optional[str] = None,
        fallback: Optional["DraftStats"] = None,
        recent_anchor_ts: int = 0,
    ):
        self.n = 0
        self.halflife_days = halflife_days
        self.bracket = bracket       # if set, only matches in this rank bracket are counted
        self.fallback = fallback     # rates shrink toward this (global) table when a cell is thin
        # Anchor for the recent window. 0 = this table's own newest match. A bracket table must
        # be handed the DATASET's newest ts instead: a thin bracket's own newest match can be
        # days old, and self-anchoring would report its pre-flip rotation as currently live.
        self.recent_anchor_ts = recent_anchor_ts
        self.ts_max = 0              # newest match ts seen by this table's build (0 = none)
        self.b_games: dict = defaultdict(float)
        self.b_wins: dict = defaultdict(float)
        self.bm_games: dict = defaultdict(float)   # (map_id, brawler)
        self.bm_wins: dict = defaultdict(float)
        self.map_games: dict = defaultdict(float)
        # Raw (unweighted) games per map within RECENT_WINDOW_DAYS of the newest match: the
        # rotation-liveness signal. `map_games` answers "how much does the model know about this
        # map" (slow, 21-day half-life); this answers "is it queueable right now" (a rotation
        # flip empties/fills it within the window). Anchored to the newest match, not the wall
        # clock, so a stalled crawl freezes the live set instead of draining it.
        self.map_games_recent: dict = defaultdict(float)
        self.syn_games: dict = defaultdict(float)  # frozenset{b1, b2}
        self.syn_wins: dict = defaultdict(float)
        self.cnt_games: dict = defaultdict(float)  # (attacker, defender)
        self.cnt_wins: dict = defaultdict(float)
        self._build(matches if matches is not None else iter_matches())

    def _recency_weights(self, tss: List[int]) -> List[float]:
        """``0.5 ** (age / half_life)`` per match, age measured from the newest match. Falls
        back to uniform weights when decay is disabled or timestamps are unavailable."""
        if self.halflife_days <= 0:
            return [1.0] * len(tss)
        tmax = max(tss, default=0)
        if tmax <= 0:
            return [1.0] * len(tss)
        half = self.halflife_days * _DAY
        return [0.5 ** ((tmax - ts) / half) for ts in tss]

    def _build(self, matches: Iterable[dict]) -> None:
        rows = [r for r in matches if r.get("a_won") is not None]
        if self.bracket is not None:
            rows = [r for r in rows if match_bracket(r) == self.bracket]
        tss = [int(r.get("ts") or 0) for r in rows]
        self.ts_max = max(tss, default=0)
        anchor = self.recent_anchor_ts or self.ts_max
        recent_min = anchor - RECENT_WINDOW_DAYS * _DAY
        for r, w, ts in zip(rows, self._recency_weights(tss), tss):
            won = r["a_won"]
            a = [p["brawler_id"] for p in r["team_a"]]
            b = [p["brawler_id"] for p in r["team_b"]]
            mid = r.get("map_id")
            self.n += 1
            self.map_games[mid] += w
            if anchor > 0 and ts >= recent_min:  # a timestamp-less row can't claim liveness
                self.map_games_recent[mid] += 1.0
            for team, win in ((a, won), (b, not won)):
                for x in team:
                    self.b_games[x] += w
                    self.bm_games[(mid, x)] += w
                    if win:
                        self.b_wins[x] += w
                        self.bm_wins[(mid, x)] += w
                for i in range(len(team)):
                    for j in range(i + 1, len(team)):
                        key = frozenset((team[i], team[j]))
                        self.syn_games[key] += w
                        if win:
                            self.syn_wins[key] += w
            for x in a:
                for y in b:
                    self.cnt_games[(x, y)] += w
                    self.cnt_games[(y, x)] += w
                    if won:
                        self.cnt_wins[(x, y)] += w
                    else:
                        self.cnt_wins[(y, x)] += w

    # --- accessors (a bracket table shrinks each rate toward its global fallback) ---
    def brawler_rate(self, brawler_id: int, map_id: Optional[int] = None) -> Rate:
        if map_id is None:
            w, g = self.b_wins[brawler_id], self.b_games[brawler_id]
        else:
            w, g = self.bm_wins[(map_id, brawler_id)], self.bm_games[(map_id, brawler_id)]
        pr = self.fallback.brawler_rate(brawler_id, map_id).winrate if self.fallback else 0.5
        return _rate(w, g, prior_rate=pr)

    def use_rate(self, brawler_id: int, map_id: int) -> float:
        games = self.map_games.get(map_id, 0.0)
        if games:
            return self.bm_games[(map_id, brawler_id)] / games
        return self.fallback.use_rate(brawler_id, map_id) if self.fallback else 0.0

    def synergy(self, b1: int, b2: int) -> Rate:
        key = frozenset((b1, b2))
        pr = self.fallback.synergy(b1, b2).winrate if self.fallback else 0.5
        return _rate(self.syn_wins[key], self.syn_games[key], prior_rate=pr)

    def counter(self, attacker: int, defender: int) -> Rate:
        pr = self.fallback.counter(attacker, defender).winrate if self.fallback else 0.5
        return _rate(self.cnt_wins[(attacker, defender)], self.cnt_games[(attacker, defender)], prior_rate=pr)

    def top_brawlers(self, map_id: Optional[int] = None, n: int = 15, min_games: float = 10) -> List[Tuple[int, Rate]]:
        rows = []
        for brawler_id in R.brawler_index():
            rate = self.brawler_rate(brawler_id, map_id)
            if rate.games >= min_games:
                rows.append((brawler_id, rate))
        rows.sort(key=lambda t: t[1].winrate, reverse=True)
        return rows[:n]


def build_bracketed(
    halflife_days: float = DEFAULT_HALFLIFE_DAYS,
    min_matches: int = MIN_BRACKET_MATCHES,
    matches: Optional[Iterable[dict]] = None,
    max_matches: int = 0,
) -> Tuple["DraftStats", Dict[str, "DraftStats"]]:
    """Build the global stats plus a per-rank-bracket table for each bracket with enough
    matches. Each bracket table shrinks toward the global one, so thin cells stay sensible
    instead of going noisy. ``max_matches`` caps the build to the most recent N matches
    (bounded-memory load) so peak RAM stays flat on a small instance as the dataset grows;
    0 = use all. Returns ``(global_stats, {bracket: stats})``."""
    if matches is not None:
        rows = list(matches)
    elif max_matches and max_matches > 0:
        rows = recent_matches(max_matches)
    else:
        rows = list(iter_matches())
    global_stats = DraftStats(rows, halflife_days=halflife_days)
    counts = Counter(b for b in (match_bracket(r) for r in rows if r.get("a_won") is not None) if b)
    brackets = {
        bk: DraftStats(rows, halflife_days=halflife_days, bracket=bk, fallback=global_stats,
                       recent_anchor_ts=global_stats.ts_max)
        for bk, c in counts.items() if c >= min_matches
    }
    return global_stats, brackets


def _name(brawler_id: int) -> str:
    idx = {b.id: b.name for b in R.load_brawlers()}
    return idx.get(brawler_id, str(brawler_id))


if __name__ == "__main__":
    stats = DraftStats()
    names = {b.id: b.name for b in R.load_brawlers()}
    map_names = {m.id: (m.name, m.mode) for m in R.load_ranked_maps()}
    print(f"matches: {stats.n}  (recency half-life: {stats.halflife_days:g} days)")

    print("\nGlobal top 10 brawlers (smoothed win-rate):")
    for bid, rt in stats.top_brawlers(n=10, min_games=50):
        print(f"  {names.get(bid, bid):<16} wr={rt.winrate:.3f}  games={rt.games:.0f}")

    busiest = max(stats.map_games, key=stats.map_games.get)
    mname, mmode = map_names.get(busiest, ("?", "?"))
    print(f"\nTop 10 on most-played map: {mname} ({mmode}, {stats.map_games[busiest]:.0f} games):")
    for bid, rt in stats.top_brawlers(map_id=busiest, n=10, min_games=8):
        ur = stats.use_rate(bid, busiest)
        print(f"  {names.get(bid, bid):<16} wr={rt.winrate:.3f}  use={ur:.0%}  games={rt.games:.0f}")
