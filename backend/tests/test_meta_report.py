"""meta_report.json carries the newest battle time CI's pipeline-stale check reads."""
import json

from bsdraft.engine.drift import detect_drift, load_report, save_report


def _match(ts):
    p = {"brawler_id": 16000000}
    return {"ts": ts, "a_won": True, "team_a": [p], "team_b": [p]}


def test_newest_ts_round_trips_and_old_reports_default_to_zero(tmp_path):
    report = detect_drift([_match(1_000), _match(5_000), {**_match(9_000), "a_won": None}])
    assert report.newest_ts == 5_000  # unlabeled rows don't count, same as the windows

    path = tmp_path / "meta_report.json"
    save_report(report, path)
    assert json.loads(path.read_text())["newest_ts"] == 5_000
    assert load_report(path).newest_ts == 5_000

    legacy = json.loads(path.read_text())
    del legacy["newest_ts"]
    path.write_text(json.dumps(legacy))
    assert load_report(path).newest_ts == 0
