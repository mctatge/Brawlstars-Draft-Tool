"""Measure the power-level win-rate deficit and freeze it as a checked-in reference constant.

    PYTHONPATH=backend python backend/scripts/export_readiness.py

Home-only: needs the full ``data/raw/matches.jsonl``, and makes two passes over it (multi-minute).
Output is ``data/reference/readiness.json`` — small, stdlib-loadable, and checked into git, so the
deployed API gets it from the repo rather than a Release asset. Nothing on the serve path reads it
until the scoring change lands; producing it is a pure addition.

The write is **gated**, not advisory. The artifact is refused when:

  * the placebo contrast exceeds its tolerance (the design is manufacturing an effect),
  * the deficit is not monotone in power (Power 9 must be at least as bad as Power 10),
  * or Power 9 misses the strata / |z| bars.

``--force`` writes anyway and stamps ``"forced": true`` into the artifact so a reader can see that
the gate was overridden. ``--dry-run`` prints the estimate and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bsdraft.constants import REFERENCE_DIR
from bsdraft.data.readiness_build import estimate_readiness

DEFAULT_OUT = REFERENCE_DIR / "readiness.json"


def _fmt(r: dict) -> str:
    if not r.get("n_strata"):
        return "no paired strata"
    return (f"RD {r['rd']*100:+.2f}pp  se {r['se']*100:.2f}  z {r['z']:+.2f}  "
            f"strata {r['n_strata']:,}  n_t {r['n_treated']:,.0f}")


def report(payload: dict) -> None:
    m = payload["meta"]
    print()
    print(f"matches {m['matches']:,}   clean appearances {m['clean_appearances']:,}")
    hist = m["power_hist"]
    total = sum(hist.values()) or 1
    top = [k for k in ("11", "10", "9", "8") if k in hist]
    print("power mix: " + "  ".join(f"P{k} {100*hist[k]/total:.3f}%" for k in top))
    print()
    for lvl in ("9", "10", "le8"):
        r = payload["levels"].get(lvl)
        if not r:
            continue
        mark = "ok " if r.get("shippable") else "-- "
        ship = f"   ship {r['ship']:.3f}" if r.get("shippable") else "   (not shipped)"
        print(f"  {mark}P{lvl:<4} {_fmt(r)}{ship}")
    p = payload["placebo"]
    print()
    print(f"  placebo   {_fmt(p)}   tol ±{p['tol']:.3f}   -> {'PASS' if p['ok'] else 'FAIL'}")
    print(f"  monotone  {'PASS' if payload['monotone'] else 'FAIL'}")
    print(f"  deficit   {payload['power_deficit']['9']:.3f} @P9  "
          f"{payload['power_deficit']['10']:.3f} @P10")
    print()


def _refusals(payload: dict) -> list:
    out = []
    if not payload["placebo"]["ok"]:
        out.append(f"placebo RD {payload['placebo']['rd']*100:+.2f}pp exceeds "
                   f"±{payload['placebo']['tol']*100:.1f}pp — the design is manufacturing an effect")
    if not payload["monotone"]:
        out.append("deficit is not monotone in power (Power 9 must be at least as bad as Power 10)")
    p9 = payload["levels"].get("9") or {}
    if not p9.get("shippable"):
        out.append(f"Power 9 misses the bar ({_fmt(p9)})")
    return out


def export(out: Path = DEFAULT_OUT, matches: Path | None = None,
           dry_run: bool = False, force: bool = False) -> int:
    payload = estimate_readiness(matches)
    report(payload)

    refusals = _refusals(payload)
    if refusals:
        print("REFUSING to write:", file=sys.stderr)
        for r in refusals:
            print(f"  - {r}", file=sys.stderr)
        if not force:
            print("\nre-run with --force to write anyway (stamps \"forced\": true)", file=sys.stderr)
            return 1
        payload["forced"] = True
        payload["refusals"] = refusals
        print("  --force given: writing anyway", file=sys.stderr)

    if dry_run:
        print("--dry-run: nothing written")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1000:.1f} kB)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure the power-level win-rate deficit; write data/reference/readiness.json.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--matches", type=Path, default=None, help="matches jsonl (default: the dataset)")
    ap.add_argument("--dry-run", action="store_true", help="estimate and report, write nothing")
    ap.add_argument("--force", action="store_true", help="write even if the gate refuses")
    args = ap.parse_args()
    raise SystemExit(export(args.out, args.matches, args.dry_run, args.force))


if __name__ == "__main__":
    main()
