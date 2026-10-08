"""Record a small public draft walkthrough from explicit aggregate artifacts.

No account, raw match history, credentials, or network calls are used. Re-run when
deliberately refreshing the dated example; this is not a live recommendation cache.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from bsdraft.data import reference as R
from bsdraft.engine.engine import DraftEngine
from bsdraft.engine.state import DraftState
from bsdraft.engine.stats_store import load_stats
from bsdraft.models.serve import WinProbModel


def capture(model_path: Path, stats_path: Path, source_note: str) -> dict:
    stats, brackets = load_stats(stats_path)
    model = WinProbModel(model_path, validate_current_era=True)
    if not model.available or stats.analysis_era_id != model.analysis_era_id:
        raise ValueError("Example requires a usable model and matching current-era stats")
    engine = DraftEngine(stats, model, bracket_stats=brackets)
    maps = [m for m in R.load_ranked_maps() if m.mode == "Knockout" and stats.map_games[m.id] > 0]
    selected = max(maps, key=lambda m: stats.map_games[m.id])
    names = {b.id: b.name for b in R.pickable_brawlers()}
    state = DraftState(selected.id, selected.mode, rank_bracket="Mythic")
    steps = []

    def record(title: str, note: str):
        picks = engine.recommend_picks(state, top=4)
        steps.append({"title": title, "note": note,
                      "allies": [names[x] for x in state.our_team],
                      "enemies": [names[x] for x in state.their_team],
                      "picks": [{k: asdict(p)[k] for k in ("brawler_id", "name", "cls", "score",
                          "map_winrate", "counter", "synergy", "win_prob", "confidence")} for p in picks]})
        return picks

    picks = record("Choose your first pick", "Start with the map and an empty board. Compare the four candidates.")
    state.our_team = [picks[0].brawler_id]
    opponent = DraftState(selected.id, selected.mode, their_team=state.our_team, rank_bracket="Mythic")
    state.their_team = [p.brawler_id for p in engine.recommend_picks(opponent, top=2)]
    picks = record("Read the enemy picks", "Your first pick is locked. Two enemy picks now change the matchup and the ranking.")
    state.our_team.append(picks[0].brawler_id)
    record("Complete your team", "With two allies locked, the remaining candidates are scored for this specific composition.")
    return {"recorded_at": datetime.now(timezone.utc).isoformat(),
            "stats_matches": stats.n,
            "source_note": source_note,
            "era_id": model.analysis_era_id, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "map": {"id": selected.id, "name": selected.name, "mode": selected.mode}, "steps": steps}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-note", required=True,
                        help="public description of the input artifacts and sample scope")
    args = parser.parse_args()
    result = capture(args.model, args.stats, args.source_note)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
