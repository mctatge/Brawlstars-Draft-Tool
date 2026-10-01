"""Guards for the live balance-era boundary."""
from __future__ import annotations

import json

from bsdraft.data.balance_eras import current_balance_era, load_balance_eras
from bsdraft.data.dataset import recent_matches
from bsdraft.engine.stats import DraftStats, build_bracketed
from bsdraft.engine.stats_store import load_payload, stats_payload


def _match(ts: int, map_id: int = 7) -> dict:
    return {
        "team_a": [{"brawler_id": b} for b in (1, 2, 3)],
        "team_b": [{"brawler_id": b} for b in (4, 5, 6)],
        "a_won": True,
        "map_id": map_id,
        "ts": ts,
    }


def test_manifest_selects_active_era():
    era = current_balance_era()
    assert era is not None
    assert era.id == "2026-09-16"
    assert era.start_ts == 1_789_516_800
    assert load_balance_eras()[-1] == era


def test_manifest_rejects_duplicate_or_non_chronological_eras(tmp_path):
    path = tmp_path / "eras.json"
    path.write_text(json.dumps({"eras": [
        {"id": "a", "effective_from": "2026-01-02T00:00:00Z"},
        {"id": "a", "effective_from": "2026-01-03T00:00:00Z"},
    ]}))
    try:
        load_balance_eras(path)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate era was accepted")


def test_recent_cap_is_applied_inside_analysis_window(tmp_path):
    path = tmp_path / "matches.jsonl"
    path.write_text("\n".join(json.dumps(_match(ts, ts)) for ts in (10, 20, 30, 40)) + "\n")
    rows = recent_matches(2, path=path, min_ts=25)
    assert [r["ts"] for r in rows] == [30, 40]


def test_stats_filter_and_artifact_metadata():
    rows = [_match(100, 100), _match(200, 200)]
    stats = DraftStats(rows, halflife_days=0, analysis_start_ts=150, analysis_era_id="era")
    assert stats.n == 1
    assert 100 not in stats.map_games
    assert stats.map_games[200] == 1.0

    global_stats, brackets = build_bracketed(
        matches=rows,
        min_matches=1,
        halflife_days=0,
        analysis_start_ts=150,
        analysis_era_id="era",
    )
    payload = stats_payload(global_stats, brackets)
    loaded, _ = load_payload(payload)
    assert loaded.n == 1
    assert loaded.analysis_start_ts == 150
    assert loaded.analysis_era_id == "era"
