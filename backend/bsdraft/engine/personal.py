"""One player's *personal* win rates, derived from collected matches.

Where ``DraftStats`` is the population meta, this is the same idea narrowed to a single
player and scored from *their* perspective: how often does *this* tag win with a given
brawler, and with that brawler *on a given map*. Each match already records all six
``player_tags`` plus every player's ``tag``/``brawler_id`` and an absolute ``a_won``
(see :mod:`bsdraft.collect.match`), so nothing new has to be collected — we just filter
the dataset to matches the player appears in and flip the label to their side.

Two things keep a tiny sample honest:

* **Bayesian shrinkage** (reusing :func:`bsdraft.engine.stats._rate`) toward a fallback
  prior — the global/bracket rate for the brawler — so a 2-game history barely moves off
  the meta and only earns its own voice as games accumulate.
* **A two-level back-off** for the map cell: personal (brawler, map) shrinks toward
  personal (brawler, overall), which in turn shrinks toward the global (brawler, map)
  rate. Per-map personal samples are the thinnest of all, so this matters most there.

Matches are recency-weighted with the same half-life as the empirical stats, so the
number tracks the player's *current* form. Duplicate copies of one game (the crawler
stores it once per crawled participant) are de-duplicated by ``match_key``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from bsdraft.collect.client import normalize_tag
from bsdraft.collect.match import parse_match
from bsdraft.data.dataset import iter_matches
from bsdraft.engine.stats import DEFAULT_HALFLIFE_DAYS, DraftStats, Rate, _rate, _DAY


class PersonalStats:
    """Per-brawler and per-(map, brawler) win rates for a single player tag."""

    def __init__(
        self,
        tag: str,
        matches: Iterable[dict],
        fallback: Optional[DraftStats] = None,
        halflife_days: float = DEFAULT_HALFLIFE_DAYS,
    ):
        self.tag = normalize_tag(tag)
        self.fallback = fallback     # rates shrink toward this (global/bracket) table
        self.n = 0                   # de-duplicated matches this player appears in
        self.b_games: dict = defaultdict(float)
        self.b_wins: dict = defaultdict(float)
        self.bm_games: dict = defaultdict(float)   # (map_id, brawler)
        self.bm_wins: dict = defaultdict(float)
        self._build(matches, halflife_days)

    def _build(self, matches: Iterable[dict], halflife_days: float) -> None:
        # Keep only labeled matches the player actually played, de-duped by match_key
        # (the same game is stored once per crawled participant — counting every copy
        # would inflate and bias the player's record toward heavily-crawled lobbies).
        rows: List[dict] = []
        seen: set = set()
        for r in matches:
            if r.get("a_won") is None:
                continue
            if self.tag not in set(r.get("player_tags") or []):
                continue
            key = r.get("match_key")
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            rows.append(r)

        for r, w in zip(rows, self._recency_weights(rows, halflife_days)):
            on_a = self.tag in {p["tag"] for p in r["team_a"]}
            team = r["team_a"] if on_a else r["team_b"]
            won = r["a_won"] if on_a else (not r["a_won"])
            bid = next((p["brawler_id"] for p in team if p["tag"] == self.tag), None)
            if bid is None:
                continue
            mid = r.get("map_id")
            self.n += 1
            self.b_games[bid] += w
            self.bm_games[(mid, bid)] += w
            if won:
                self.b_wins[bid] += w
                self.bm_wins[(mid, bid)] += w

    @staticmethod
    def _recency_weights(rows: List[dict], halflife_days: float) -> List[float]:
        """``0.5 ** (age / half_life)`` per match, age from the player's newest game."""
        if halflife_days <= 0:
            return [1.0] * len(rows)
        tss = [int(r.get("ts") or 0) for r in rows]
        tmax = max(tss, default=0)
        if tmax <= 0:
            return [1.0] * len(rows)
        half = halflife_days * _DAY
        return [0.5 ** ((tmax - ts) / half) for ts in tss]

    def games_on(self, brawler_id: int) -> float:
        """Effective (recency-weighted) games this player has on a brawler. 0 = never
        played it, so callers can omit a personal signal rather than echo the prior."""
        return self.b_games.get(brawler_id, 0.0)

    def brawler_rate(self, brawler_id: int, map_id: Optional[int] = None) -> Rate:
        """Smoothed personal win rate, shrunk toward the fallback. The returned
        ``Rate.games`` is the player's *own* effective sample (0 when unplayed), not the
        prior's — so confidence reflects how much we actually know about this player."""
        if map_id is not None and self.bm_games.get((map_id, brawler_id), 0.0) > 0:
            w, g = self.bm_wins[(map_id, brawler_id)], self.bm_games[(map_id, brawler_id)]
            prior = self.brawler_rate(brawler_id, None).winrate     # personal, all maps
        elif self.b_games.get(brawler_id, 0.0) > 0:
            w, g = self.b_wins[brawler_id], self.b_games[brawler_id]
            prior = self.fallback.brawler_rate(brawler_id, map_id).winrate if self.fallback else 0.5
        else:
            w = g = 0.0
            prior = self.fallback.brawler_rate(brawler_id, map_id).winrate if self.fallback else 0.5
        return _rate(w, g, prior_rate=prior)

    def baseline_rate(self, brawler_id: int, map_id: Optional[int] = None) -> float:
        """The *population* rate this player's record is measured against — what
        :meth:`brawler_rate` converges to when they have never played the brawler.

        Exposed so a caller can take a clean ``personal - baseline`` difference. Subtracting some
        other table's rate instead would conflate the player's edge with the gap between two
        populations: personal stats fall back to the GLOBAL table while a pick is scored against
        the rank-BRACKET table, so those two numbers are not the same quantity."""
        if self.fallback is None:
            return 0.5
        return self.fallback.brawler_rate(brawler_id, map_id).winrate

    def overall_rate(self) -> Rate:
        """This player's win rate across every brawler, smoothed toward 0.5.

        0.5 is the correct anchor rather than a population average: ranked 3v3 is symmetric, so the
        population wins exactly half its games by construction.

        For DISPLAY — it belongs in a column header, stated once. It deliberately does not enter
        the pick score: being a per-player constant it shifts every candidate identically, so it
        cannot reorder a list, and folding it in only moves the level (on a 40%-overall account it
        added ~5.5 points to every row). See the history-edge block in ``engine/scoring.py``."""
        wins = sum(self.b_wins.values())
        games = sum(self.b_games.values())
        return _rate(wins, games, prior_rate=0.5)


def matches_from_battlelog(battles: Iterable[dict], tag: str) -> List[dict]:
    """Normalize a live battle log into Match dicts (ranked 3v3 only), shaped exactly like
    the stored dataset rows so they merge by ``match_key``. Used to augment the dataset
    with a player's freshest games when an API key is available (local/home only)."""
    from dataclasses import asdict

    out: List[dict] = []
    for entry in battles:
        m = parse_match(entry, tag)
        if m is not None:
            out.append(asdict(m))
    return out


def build_personal_stats(
    tag: str,
    fallback: Optional[DraftStats] = None,
    extra_matches: Optional[Iterable[dict]] = None,
    halflife_days: float = DEFAULT_HALFLIFE_DAYS,
) -> Optional[PersonalStats]:
    """Build personal stats for ``tag`` from the synced dataset, optionally augmented with
    freshly-fetched matches (e.g. a live battle log). Returns ``None`` when the player has
    no labeled games anywhere, so callers can cleanly skip personalization."""
    tag_n = normalize_tag(tag)
    if not tag_n:
        return None
    # Stream-filter to just this player's games (bounded memory) instead of materializing the
    # whole dataset — PersonalStats only ever uses matches the tag appears in. The raw-substring
    # prefilter (contains=tag_n) skips json.loads on every line the tag doesn't appear in; the
    # exact membership check then drops the rare line where tag_n is only a substring.
    rows = [r for r in iter_matches(contains=tag_n) if tag_n in (r.get("player_tags") or ())]
    if extra_matches:
        rows.extend(extra_matches)
    ps = PersonalStats(tag_n, rows, fallback=fallback, halflife_days=halflife_days)
    return ps if ps.n > 0 else None
