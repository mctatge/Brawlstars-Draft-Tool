"""Unit tests for the home-Mac IP-lockout watchdog (backend/scripts/watchdog.py).

Pins the pure logic: .env parsing, probe-result classification (only a 403 with
reason ``accessDenied.invalidIp`` is a lockout), and the once-per-outage alert
debounce — including that a mid-outage network blip must not reset it into a second
alert. No network, no filesystem, no notifications.

    PYTHONPATH=backend python -m pytest backend/tests/test_watchdog.py    # or run directly
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "bsdraft_watchdog", Path(__file__).resolve().parents[1] / "scripts" / "watchdog.py"
)
wd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wd)

HOUR = 3600.0


# --- parse_env ---

def test_parse_env_handles_comments_export_and_quotes():
    env = wd.parse_env(
        "# comment\n"
        "\n"
        "export BRAWLSTARS_API_TOKEN='abc.def'\n"
        'PLAYER_TAG="#2YULP2"\n'
        "DATA_URL=https://example.com/a?b=c=d\n"
        "not a kv line\n"
    )
    assert env["BRAWLSTARS_API_TOKEN"] == "abc.def"
    assert env["PLAYER_TAG"] == "#2YULP2"
    assert env["DATA_URL"] == "https://example.com/a?b=c=d"  # split on FIRST '='
    assert len(env) == 3


# --- classify ---

def _body(reason: str) -> str:
    return json.dumps({"reason": reason})


def test_classify_200_is_ok():
    assert wd.classify(200, "") == ("ok", "")


def test_classify_invalid_ip_403_is_lockout():
    state, detail = wd.classify(403, _body("accessDenied.invalidIp"))
    assert state == "lockout"


def test_classify_other_403_and_401_are_auth_error():
    assert wd.classify(403, _body("accessDenied"))[0] == "auth_error"
    assert wd.classify(401, "")[0] == "auth_error"
    assert wd.classify(403, "not json")[0] == "auth_error"  # malformed body still alerts


def test_classify_transient_failures_are_indeterminate():
    for status in (429, 500, 503):
        assert wd.classify(status, "")[0] == "indeterminate"
    assert wd.classify(None, "timed out")[0] == "indeterminate"  # network error


# --- decide (debounce) ---

def test_fresh_lockout_alerts_once_then_stays_silent():
    note, deb = wd.decide({}, "lockout", now=0.0, remind_hours=6)
    assert note == "alert"
    note, deb = wd.decide(deb, "lockout", now=600.0, remind_hours=6)
    assert note is None
    assert deb["outage_started_at"] == 0.0  # outage start preserved


def test_reminder_fires_after_remind_hours():
    _, deb = wd.decide({}, "lockout", now=0.0, remind_hours=6)
    note, deb = wd.decide(deb, "lockout", now=6 * HOUR, remind_hours=6)
    assert note == "reminder"
    note, _ = wd.decide(deb, "lockout", now=6 * HOUR + 600, remind_hours=6)
    assert note is None  # reminder itself is debounced


def test_remind_hours_zero_means_strictly_once_per_outage():
    _, deb = wd.decide({}, "lockout", now=0.0, remind_hours=0)
    note, _ = wd.decide(deb, "lockout", now=100 * HOUR, remind_hours=0)
    assert note is None


def test_network_blip_mid_outage_does_not_restart_the_outage():
    _, deb = wd.decide({}, "lockout", now=0.0, remind_hours=6)
    note, deb = wd.decide(deb, "indeterminate", now=600.0, remind_hours=6)
    assert note is None
    note, deb = wd.decide(deb, "lockout", now=1200.0, remind_hours=6)
    assert note is None  # same outage — no second alert
    assert deb["outage_started_at"] == 0.0


def test_recovery_notifies_once():
    _, deb = wd.decide({}, "lockout", now=0.0, remind_hours=6)
    note, deb = wd.decide(deb, "ok", now=600.0, remind_hours=6)
    assert note == "recovered"
    note, _ = wd.decide(deb, "ok", now=1200.0, remind_hours=6)
    assert note is None


def test_ok_steady_state_is_silent():
    note, deb = wd.decide({}, "ok", now=0.0, remind_hours=6)
    assert note is None
    note, _ = wd.decide(deb, "indeterminate", now=600.0, remind_hours=6)
    assert note is None  # a blip while healthy is not an outage


def test_no_token_is_alertable_like_a_lockout():
    note, deb = wd.decide({}, "no_token", now=0.0, remind_hours=6)
    assert note == "alert"
    # reclassification within one outage (token pasted wrong -> auth_error) stays quiet
    note, _ = wd.decide(deb, "auth_error", now=600.0, remind_hours=6)
    assert note is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
