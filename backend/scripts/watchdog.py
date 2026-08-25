#!/usr/bin/env python3
"""One-shot Supercell key/IP watchdog for the home Mac (stdlib-only).

Comcast rotates the home IP every 1-2 weeks; each rotation 403s the IP-locked key
(``accessDenied.invalidIp``), which silently kills the crawler and the roster tunnel.
The cloud-side keepwarm Action has never caught one in time, so this runs locally:
the ``com.bsdraft.watchdog`` launchd agent (see deploy/) invokes it every ~10 minutes.

Each run probes one cheap API endpoint with the token from the repo-root .env
(re-read every run, so a pasted-in replacement token is picked up with no restart).
On an IP lockout it fires a macOS notification and writes
``data/raw/watchdog_status.json`` with the current public IP and the exact fix —
mint a new key for that IP at https://developer.brawlstars.com, paste it into .env,
then ``launchctl kickstart -k`` both com.bsdraft.api and com.bsdraft.crawler (they
hold the old token in memory). Alerts are debounced to once per outage, with a
reminder every ``WATCHDOG_REMIND_HOURS`` (default 6, 0 disables) in case the first
banner was missed, and a one-shot "recovered" notification when the probe succeeds
again.

Deliberately dependency-free (runs on system python3, no venv, no PYTHONPATH) so it
keeps working when the rest of the stack is broken.

    python3 backend/scripts/watchdog.py                 # one probe + status write
    python3 backend/scripts/watchdog.py --test-notify   # verify notification perms
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = REPO_ROOT / "data" / "raw" / "watchdog_status.json"
PROBE_URL = "https://api.brawlstars.com/v1/brawlers?limit=1"
IP_URL = "https://api.ipify.org"
TIMEOUT_SECONDS = 15.0
TOKEN_KEY = "BRAWLSTARS_API_TOKEN"
KICKSTART_AGENTS = ("com.bsdraft.api", "com.bsdraft.crawler")

# States that warrant a notification. "indeterminate" (network down, 5xx, 429) is
# deliberately not alertable — a flapping cable modem must not cry wolf, and it must
# not clear an ongoing outage either (see decide()).
ALERTABLE = frozenset({"lockout", "auth_error", "no_token"})


def parse_env(text: str) -> dict:
    """Minimal KEY=VALUE parser for the repo-root .env (config.py conventions:
    ``#`` comments, optional ``export`` prefix, optional single/double quotes)."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def load_token() -> str:
    """Environment overrides .env, same precedence as bsdraft.config.Settings."""
    token = os.environ.get(TOKEN_KEY, "").strip()
    if token:
        return token
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        return parse_env(env_path.read_text(encoding="utf-8")).get(TOKEN_KEY, "").strip()
    return ""


def probe(token: str) -> tuple:
    """GET one cheap authenticated endpoint. Returns (http_status | None, body/error)."""
    req = urllib.request.Request(
        PROBE_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            return resp.status, ""
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return e.code, body
    except Exception as e:  # DNS down, timeout, TLS, connection refused, ...
        return None, str(e)


def classify(http_status, body: str) -> tuple:
    """(state, detail): ok / lockout / auth_error / indeterminate."""
    if http_status == 200:
        return "ok", ""
    if http_status in (401, 403):
        try:
            reason = json.loads(body).get("reason", "")
        except (ValueError, AttributeError):
            reason = ""
        if reason == "accessDenied.invalidIp":
            return "lockout", reason
        return "auth_error", reason or f"HTTP {http_status}"
    # Transient or ambiguous (429, 5xx, network error) — not evidence about the key.
    return "indeterminate", body if http_status is None else f"HTTP {http_status}"


def public_ip() -> str:
    try:
        with urllib.request.urlopen(IP_URL, timeout=10) as resp:
            return resp.read().decode("ascii", "replace").strip()
    except Exception:
        return "unknown"


def decide(prev: dict, state: str, now: float, remind_hours: float) -> tuple:
    """Fold a probe state into the persisted debounce record.

    ``prev`` is the previous ``_debounce`` dict ({} on first run). Returns
    (notification, debounce) where notification is None, "alert", "reminder", or
    "recovered". An indeterminate probe changes nothing, so lockout → network blip
    → lockout stays one outage (one alert).
    """
    last = prev.get("last_state", "ok")
    started = prev.get("outage_started_at")
    alerted = prev.get("alerted_at")

    if state == "indeterminate":
        return None, {"last_state": last, "outage_started_at": started, "alerted_at": alerted}
    if state in ALERTABLE:
        if last not in ALERTABLE:
            return "alert", {"last_state": state, "outage_started_at": now, "alerted_at": now}
        started = now if started is None else started
        if remind_hours > 0 and alerted is not None and now - alerted >= remind_hours * 3600:
            return "reminder", {"last_state": state, "outage_started_at": started, "alerted_at": now}
        return None, {"last_state": state, "outage_started_at": started, "alerted_at": alerted}
    # ok
    if last in ALERTABLE:
        return "recovered", {"last_state": "ok", "outage_started_at": None, "alerted_at": None}
    return None, {"last_state": "ok", "outage_started_at": None, "alerted_at": None}


def fix_steps(ip: str) -> list:
    uid = os.getuid()
    return [
        f"1. Mint a new key allowing IP {ip} at https://developer.brawlstars.com (My Account -> Keys).",
        f"2. Paste the token into {TOKEN_KEY} in {REPO_ROOT}/.env",
        *[f"{i}. launchctl kickstart -k gui/{uid}/{agent}"
          for i, agent in enumerate(KICKSTART_AGENTS, start=3)],
        "   (both hold the old token in memory until kickstarted; the tunnel agent has no key)",
        "The watchdog re-reads .env each cycle and notifies 'recovered' once the new token works.",
    ]


def notify(title: str, message: str, sound: bool = True) -> None:
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    script = f'display notification "{esc(message)}" with title "{esc(title)}"'
    if sound:
        script += ' sound name "Basso"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True, timeout=30)
    except Exception:
        pass  # never let a notification failure kill the status write


NOTIFICATIONS = {
    "lockout": (
        "BrawlDraft: Supercell key IP-locked out",
        "Public IP is now {ip}. Mint a key for it at developer.brawlstars.com, paste into "
        ".env, then kickstart com.bsdraft.api + com.bsdraft.crawler. "
        "Fix steps: data/raw/watchdog_status.json",
    ),
    "auth_error": (
        "BrawlDraft: Supercell API auth failing",
        "{detail} (not the usual invalidIp) — check {token_key} in .env, then kickstart "
        "api + crawler. Details: data/raw/watchdog_status.json",
    ),
    "no_token": (
        "BrawlDraft: no Supercell token",
        "{token_key} is missing from .env — the crawler and roster tunnel are down.",
    ),
}


def main(argv) -> int:
    if "--test-notify" in argv:
        notify("BrawlDraft watchdog", "Test notification — alerts are working.")
        print("sent test notification (allow osascript/Script Editor in Notification Center "
              "if nothing appeared)")
        return 0

    now = time.time()
    token = load_token()
    if not token:
        state, http_status, detail = "no_token", None, f"{TOKEN_KEY} missing from .env"
    else:
        http_status, body = probe(token)
        state, detail = classify(http_status, body)
    ip = public_ip()

    try:
        prev = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        prev = {}
    try:
        remind_hours = float(os.environ.get("WATCHDOG_REMIND_HOURS", "6"))
    except ValueError:
        remind_hours = 6.0
    notification, debounce = decide(prev.get("_debounce", {}), state, now, remind_hours)

    def iso(epoch):
        if epoch is None:
            return None
        return datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="seconds")

    status = {
        "state": state,
        "detail": detail,
        "http_status": http_status,
        "checked_at": iso(now),
        "public_ip": ip,
        "last_ok_at": iso(now) if state == "ok" else prev.get("last_ok_at"),
        "last_ok_ip": ip if state == "ok" else prev.get("last_ok_ip"),
        "outage_started_at": iso(debounce["outage_started_at"]),
        "alerted_at": iso(debounce["alerted_at"]),
        "fix": fix_steps(ip) if state in ALERTABLE else None,
        "_debounce": debounce,
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, STATUS_PATH)

    if notification == "recovered":
        notify("BrawlDraft: Supercell key OK again",
               f"Probe succeeded from IP {ip}. If you just swapped the token, remember to "
               "kickstart com.bsdraft.api + com.bsdraft.crawler.")
    elif notification is not None:  # "alert" or "reminder"
        title, message = NOTIFICATIONS[state]
        if notification == "reminder":
            title += " (still)"
        notify(title, message.format(ip=ip, detail=detail, token_key=TOKEN_KEY))

    print(f"{iso(now)} state={state} http={http_status} ip={ip} "
          f"notify={notification or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
