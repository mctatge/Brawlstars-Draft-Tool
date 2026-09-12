"""The crawler's consecutive-retrain-failure alarm.

Added after a run of 38 silent `--max-full-delta` gate refusals left the served model 8 days
stale with no signal anywhere but an unread log line.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from bsdraft.constants import REPO_ROOT


@pytest.fixture
def collect(tmp_path, monkeypatch):
    """scripts/collect.py isn't importable as a package module — load it by path, with its
    streak state redirected into tmp so tests never touch data/processed."""
    spec = importlib.util.spec_from_file_location(
        "collect_under_test", REPO_ROOT / "backend" / "scripts" / "collect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_RETRAIN_STATE_PATH", tmp_path / "retrain_state.json")
    monkeypatch.setattr(mod, "_RETRAIN_DISPATCH_STATE_PATH", tmp_path / "retrain_dispatch_state.json")
    monkeypatch.delenv("BSDRAFT_RETRAIN_WORKFLOW", raising=False)
    monkeypatch.delenv("BSDRAFT_RETRAIN_DISPATCH_COOLDOWN_HOURS", raising=False)
    return mod


def test_streak_counts_up_and_a_success_clears_it(collect):
    assert collect._record_retrain(False, "gate") == 1
    assert collect._record_retrain(False, "gate") == 2
    assert collect._record_retrain(True) == 0
    assert collect._record_retrain(False, "gate") == 1


def test_streak_survives_a_process_restart(collect):
    collect._record_retrain(False, "gate")
    collect._record_retrain(False, "gate")
    # The daemon gets kickstarted a lot; a streak that resets on restart would never alert.
    assert json.loads(collect._RETRAIN_STATE_PATH.read_text())["consecutive_failures"] == 2
    assert collect._record_retrain(False, "gate") == 3


def test_unreadable_state_starts_a_fresh_streak_rather_than_raising(collect):
    collect._RETRAIN_STATE_PATH.write_text("{not json")
    assert collect._record_retrain(False, "gate") == 1


def _calls(collect, monkeypatch):
    seen = []

    def fake_gh(*args):
        seen.append(args)
        if args[:2] == ("issue", "list"):
            return subprocess.CompletedProcess(args, 0, stdout="[]", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(collect.publisher, "_gh", fake_gh)
    return seen


def test_no_alert_below_the_threshold(collect, monkeypatch):
    seen = _calls(collect, monkeypatch)
    for streak in range(1, collect._RETRAIN_ALERT_AFTER):
        collect._alert_retrain_stalled(streak, "gate")
    assert seen == []          # a one-off failure is noise, not an incident


def test_alerts_once_on_crossing_then_daily(collect, monkeypatch):
    seen = _calls(collect, monkeypatch)
    collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER, "gate detail")
    assert any(a[:2] == ("issue", "create") for a in seen)
    # The next few cycles stay quiet …
    seen.clear()
    for extra in range(1, collect._RETRAIN_REALERT_EVERY):
        collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER + extra, "gate detail")
    assert not any(a[:2] == ("issue", "create") for a in seen)
    # … then it nags again a day later.
    seen.clear()
    collect._alert_retrain_stalled(
        collect._RETRAIN_ALERT_AFTER + collect._RETRAIN_REALERT_EVERY, "gate detail")
    assert any(a[:2] == ("issue", "create") for a in seen)


def test_an_open_issue_suppresses_a_duplicate(collect, monkeypatch):
    seen = []

    def fake_gh(*args):
        seen.append(args)
        if args[:2] == ("issue", "list"):
            return subprocess.CompletedProcess(args, 0, stdout='[{"number": 7}]', stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(collect.publisher, "_gh", fake_gh)
    collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER, "gate")
    assert not any(a[:2] == ("issue", "create") for a in seen)


def test_alert_failure_never_propagates(collect, monkeypatch):
    monkeypatch.setattr(collect.publisher, "_gh",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("gh missing")))
    collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER, "gate")  # must not raise


def test_alert_body_carries_the_gate_explanation(collect, monkeypatch):
    seen = _calls(collect, monkeypatch)
    collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER,
                                   "full-comp regression gate: paired logloss delta +0.0025")
    create = [a for a in seen if a[:2] == ("issue", "create")][0]
    assert "paired logloss delta +0.0025" in " ".join(create)


def test_missing_label_falls_back_to_an_unlabelled_issue(collect, monkeypatch):
    # `gh` errors on a label the repo doesn't have; losing the alert over a taxonomy detail
    # would defeat the point.
    seen = []

    def fake_gh(*args):
        seen.append(args)
        if args[:2] == ("issue", "list"):
            return subprocess.CompletedProcess(args, 0, stdout="[]", stderr="")
        if "--label" in args:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="unknown label")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(collect.publisher, "_gh", fake_gh)
    collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER, "gate")
    creates = [a for a in seen if a[:2] == ("issue", "create")]
    assert len(creates) == 2 and "--label" not in creates[1]


def _shift_report(*, new_brawlers=(), shifts=()):
    return SimpleNamespace(
        new_brawlers=list(new_brawlers),
        shifts=[SimpleNamespace(brawler_id=bid, kind=kind) for bid, kind in shifts],
    )


def test_report_fingerprint_ignores_rate_noise(collect):
    a = _shift_report(new_brawlers=[99], shifts=[(1, "buff"), (2, "nerf")])
    b = SimpleNamespace(
        new_brawlers=[99],
        shifts=[
            SimpleNamespace(brawler_id=2, kind="nerf", wr_before=0.1, wr_after=0.2),
            SimpleNamespace(brawler_id=1, kind="buff", wr_before=0.4, wr_after=0.9),
        ],
    )
    assert collect._report_fingerprint(a) == collect._report_fingerprint(b)


def test_remote_retrain_dispatches_workflow_and_records_state(collect, monkeypatch):
    seen = []

    def fake_gh(*args):
        seen.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(collect.publisher, "_gh", fake_gh)
    now = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    assert collect._trigger_remote_retrain(_shift_report(shifts=[(1, "buff")]), now=now)

    assert seen == [(
        "workflow", "run", "retrain-model.yml",
        "-f", "reason=meta_shift",
        "-f", f"report_fingerprint={collect._report_fingerprint(_shift_report(shifts=[(1, 'buff')]))}",
    )]
    state = json.loads(collect._RETRAIN_DISPATCH_STATE_PATH.read_text())
    assert state["last_fingerprint"] == collect._report_fingerprint(_shift_report(shifts=[(1, "buff")]))
    assert state["last_dispatched_at"] == "2026-09-12T12:00:00Z"


def test_remote_retrain_cooldown_suppresses_same_shift(collect, monkeypatch):
    seen = []

    def fake_gh(*args):
        seen.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(collect.publisher, "_gh", fake_gh)
    report = _shift_report(shifts=[(1, "buff")])
    t0 = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    assert collect._trigger_remote_retrain(report, now=t0)
    assert not collect._trigger_remote_retrain(report, now=t0 + timedelta(hours=1))
    assert len(seen) == 1


def test_remote_retrain_new_fingerprint_bypasses_cooldown(collect, monkeypatch):
    seen = []

    def fake_gh(*args):
        seen.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(collect.publisher, "_gh", fake_gh)
    t0 = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    assert collect._trigger_remote_retrain(_shift_report(shifts=[(1, "buff")]), now=t0)
    assert collect._trigger_remote_retrain(_shift_report(shifts=[(2, "nerf")]),
                                           now=t0 + timedelta(hours=1))
    assert len(seen) == 2


def test_remote_retrain_dispatch_failure_does_not_record_success(collect, monkeypatch):
    monkeypatch.setattr(
        collect.publisher, "_gh",
        lambda *a: subprocess.CompletedProcess(a, 1, stdout="", stderr="missing workflow"),
    )
    assert not collect._trigger_remote_retrain(_shift_report(shifts=[(1, "buff")]))
    assert not collect._RETRAIN_DISPATCH_STATE_PATH.exists()
