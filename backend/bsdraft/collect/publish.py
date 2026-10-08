"""Publish the collected matches (and the trained model) to a GitHub Release for the API to pull.

Gzips ``data/raw/matches.jsonl`` and uploads it as the ``matches.jsonl.gz`` asset on a fixed
release tag (default ``data-latest``), replacing the previous asset; :func:`publish_model`
uploads ``winprob.npz`` alongside it. The cloud API's ``DATA_URL`` / ``MODEL_URL`` point at
those assets' stable download URLs:

    https://github.com/<owner>/<repo>/releases/download/data-latest/matches.jsonl.gz
    https://github.com/<owner>/<repo>/releases/download/data-latest/winprob.npz

Requires the GitHub CLI (`gh`), authenticated, run from inside the repo (gh infers the
owner/repo from the git remote). Keeping this on your machine is what lets the crawl keep
using the IP-locked Supercell key while the cloud stays free.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
import gzip
import shutil
import subprocess
from pathlib import Path

from bsdraft.constants import PROCESSED_DIR, RAW_DIR

MATCHES_PATH = RAW_DIR / "matches.jsonl"
GZ_PATH = RAW_DIR / "matches.jsonl.gz"
MODEL_PATH = PROCESSED_DIR / "winprob.npz"
STATS_PATH = PROCESSED_DIR / "stats.json.gz"
RANK_INDEX_PATH = PROCESSED_DIR / "rank_index.json.gz"       # legacy container (rollback target)
RANK_INDEX_NPZ_PATH = PROCESSED_DIR / "rank_index.npz"       # current container
META_REPORT_PATH = PROCESSED_DIR / "meta_report.json"
ITEMSTATS_PATH = PROCESSED_DIR / "itemstats.json.gz"
DEFAULT_TAG = "data-latest"
METRICS_PATH = PROCESSED_DIR.parent.parent / "docs" / "metrics.json"
from bsdraft.models.releases import POINTER_TAG, parse_manifest, sha256_file, validate_manifest


def validate_model_bundle(model_path: Path, metrics_path: Path) -> dict:
    report = json.loads(metrics_path.read_text())
    import numpy as np
    from bsdraft.data.balance_eras import current_balance_era
    from bsdraft.models.evaluation import load_incumbent
    with np.load(model_path, allow_pickle=False) as archive:
        embedded = json.loads(archive["_evaluation"].item())
        analysis = json.loads(archive["_analysis"].item())
        required_evidence = {"training_run_id", "trained_at", "source_commit", "dataset_sha256", "source_dirty",
                             "data_through_ts", "training_until_ts", "selection_until_ts", "test_start_ts",
                             "n_train", "n_selection", "n_test", "n_total", "publication_gate",
                             "evaluation_reservation", "embedding", "baseline_released_incumbent"}
        if not isinstance(embedded, dict) or not required_evidence <= embedded.keys():
            raise ValueError("model lacks complete publication evidence")
        if any(report.get(key) != value for key, value in embedded.items()):
            raise ValueError("model and report evaluation metadata mismatch")
        if analysis != report.get("analysis"):
            raise ValueError("model and report analysis mismatch")
        cfg = json.loads(archive["_config"].item())
        facts = report.get("model", {})
        parameter_count = sum(int(archive[key].size) for key in archive.files
                              if not key.startswith("_") and key != "brawler_class")
        if (facts.get("config") != cfg or facts.get("parameters") != parameter_count or
                facts.get("pinned_maps") != len(archive["_vocab_map_ids"])):
            raise ValueError("reported model facts do not match the actual weights")
    load_incumbent(model_path, required=True)
    era = current_balance_era()
    if era is None or analysis != {"era_id": era.id, "start_ts": era.start_ts}:
        raise ValueError("publication requires the active balance era")
    if report.get("source_dirty") is not False:
        raise ValueError("publication requires a clean source checkout")
    reservation = report.get("evaluation_reservation") or {}
    if (not reservation.get("reservation_id") or
            reservation.get("consumed_through_ts") != report.get("data_through_ts") or
            reservation.get("dataset_sha256") != report.get("dataset_sha256") or
            reservation.get("source_commit") != report.get("source_commit") or
            report.get("test_start_ts", 0) <= reservation.get("test_after_ts", 0)):
        raise ValueError("missing or inconsistent consumed-test reservation")
    if report.get("n_total") != sum(report.get(key, 0) for key in ("n_train", "n_selection", "n_test")):
        raise ValueError("inconsistent temporal split sample counts")
    if report.get("data_through_ts", 0) < report.get("test_start_ts", 0):
        raise ValueError("test starts beyond dataset snapshot")
    if report.get("weights_sha256") != sha256_file(model_path):
        raise ValueError("model and metrics digest mismatch")
    gate = report.get("publication_gate", {})
    if gate.get("passed") is not True or gate.get("require_incumbent") is not True:
        raise ValueError("model did not pass the mandatory incumbent publication gate")
    import re
    if (not re.fullmatch(r"[0-9a-f]{64}", str(gate.get("incumbent_sha256", ""))) or
            not isinstance(gate.get("delta"), (int, float))):
        raise ValueError("missing paired incumbent comparison")
    expected_delta = report["embedding"]["logloss"] - report["baseline_released_incumbent"]["logloss"]
    if not math.isclose(gate["delta"], expected_delta, rel_tol=0, abs_tol=1e-12):
        raise ValueError("paired delta does not match the reported final-test metrics")
    threshold = gate.get("max_full_delta", -1)
    if not math.isfinite(threshold) or threshold < 0 or not math.isfinite(gate["delta"]) or gate["delta"] > threshold:
        raise ValueError("invalid or failed publication gate")
    if report.get("n_test", 0) < 1000 or report.get("n_train", 0) < 100 or report.get("n_selection", 0) < 100:
        raise ValueError("insufficient temporal evaluation sample")
    if not (0 < report.get("training_until_ts", 0) < report.get("selection_until_ts", 0) < report.get("test_start_ts", 0)):
        raise ValueError("training, selection and final test are not temporally separated")
    for key in ("logloss", "auc", "ece"):
        if not math.isfinite(report.get("embedding", {}).get(key, float("nan"))):
            raise ValueError("missing finite final-test metrics")
    return report


def _checked_gh(*args: str):
    result = _gh(*args)
    if result.returncode:
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {result.stderr.strip()}")
    return result


def _verify_release_bundle(tag: str, model_digest: str, metrics_digest: str) -> None:
    # Download the uploaded bytes before promotion, rather than trusting upload exit status alone.
    with tempfile.TemporaryDirectory(prefix="bsdraft-release-verify-") as directory:
        _checked_gh("release", "download", tag, "--pattern", "winprob.npz", "--pattern", "metrics.json", "--dir", directory)
        for name, expected in (("winprob.npz", model_digest), ("metrics.json", metrics_digest)):
            if sha256_file(Path(directory) / name) != expected:
                raise ValueError(f"uploaded {name} digest mismatch")


def publish_model_bundle(model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH) -> dict:
    report = validate_model_bundle(model_path, metrics_path)
    repository = os.environ.get("GH_REPO") or _checked_gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner").stdout.strip()
    model_digest, metrics_digest = sha256_file(model_path), sha256_file(metrics_path)
    tag = f"model-{model_digest[:16]}-{metrics_digest[:12]}"
    prefix = f"https://github.com/{repository}/releases/download/{tag}"
    manifest = validate_manifest({
        "schema_version": 1, "release_id": tag,
        "published_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trained_at": report["trained_at"], "source_commit": report["source_commit"],
        "dataset_sha256": report["dataset_sha256"], "analysis": report["analysis"],
        "model": {"url": prefix + "/winprob.npz", "sha256": model_digest, "size_bytes": model_path.stat().st_size},
        "metrics": {"url": prefix + "/metrics.json", "sha256": metrics_digest, "size_bytes": metrics_path.stat().st_size},
    })
    existing = _gh("release", "view", tag)
    if existing.returncode:
        with tempfile.TemporaryDirectory(prefix="bsdraft-release-stage-") as directory:
            model_copy, metrics_copy = Path(directory) / "winprob.npz", Path(directory) / "metrics.json"
            shutil.copyfile(model_path, model_copy)
            shutil.copyfile(metrics_path, metrics_copy)
            _checked_gh("release", "create", tag, str(model_copy), str(metrics_copy),
                        "--draft", "--target", report["source_commit"],
                        "--title", tag, "--notes", "Immutable model and matching temporal-test evaluation.", "--latest=false")
    _verify_release_bundle(tag, model_digest, metrics_digest)
    _checked_gh("release", "edit", tag, "--draft=false")
    # GitHub's release body PATCH is atomic. Never delete/clobber the old pointer or model.
    pointer = _gh("api", f"repos/{repository}/releases/tags/{POINTER_TAG}")
    with tempfile.TemporaryDirectory(prefix="bsdraft-release-promote-") as directory:
        body = Path(directory) / "manifest.json"
        body.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        if pointer.returncode == 0:
            release_id = json.loads(pointer.stdout)["id"]
            patch = Path(directory) / "patch.json"
            patch.write_text(json.dumps({"body": body.read_text()}))
            _checked_gh("api", "--method", "PATCH", f"repos/{repository}/releases/{release_id}", "--input", str(patch))
        elif "404" in pointer.stderr:
            _checked_gh("release", "create", POINTER_TAG, "--target", report["source_commit"],
                        "--title", "Current verified model", "--notes-file", str(body), "--latest=false")
        else:
            raise RuntimeError(f"could not read model pointer; leaving it unchanged: {pointer.stderr.strip()}")
    promoted = parse_manifest(_checked_gh("api", f"repos/{repository}/releases/tags/{POINTER_TAG}").stdout)
    if promoted["model"]["sha256"] != model_digest or promoted["metrics"]["sha256"] != metrics_digest:
        raise ValueError("model pointer promotion readback mismatch")
    print(f"published immutable bundle {tag}; promoted {POINTER_TAG}")
    return manifest



def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def _ensure_release(tag: str) -> None:
    if _gh("release", "view", tag).returncode != 0:
        res = _gh(
            "release", "create", tag,
            "--title", "Latest dataset",
            "--notes", "Rolling ranked-match dataset powering the live draft API (updated by the crawler).",
        )
        if res.returncode != 0:
            raise RuntimeError(f"gh release create failed: {res.stderr.strip()}")


def gzip_matches(src: Path = MATCHES_PATH, dst: Path = GZ_PATH) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"No matches file at {src} — run the crawler first.")
    with open(src, "rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout)
    return dst


def publish(tag: str = DEFAULT_TAG) -> None:
    gz = gzip_matches()
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(gz), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload failed: {res.stderr.strip()}")
    print(f"published {gz.name} ({gz.stat().st_size / 1e6:.1f} MB) -> release '{tag}'")


def publish_model(tag: str = DEFAULT_TAG) -> None:
    """Publish a verified immutable bundle and atomically promote its manifest.

    The old data-latest/winprob.npz is deliberately retained during migration; model readers
    must use MODEL_MANIFEST_URL before future publications become visible to them.
    """
    if tag != DEFAULT_TAG:
        raise ValueError("model publication uses immutable versioned tags, not a custom rolling tag")
    publish_model_bundle()


def publish_stats(tag: str = DEFAULT_TAG) -> None:
    """Upload the precomputed stats (stats.json.gz) to the release so an API with STATS_URL set
    loads them instead of rebuilding from the full dataset. Run after scripts/export_stats.py
    (the crawler does this automatically each publish cycle)."""
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"No stats at {STATS_PATH} — build them first (scripts/export_stats.py).")
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(STATS_PATH), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload (stats) failed: {res.stderr.strip()}")
    print(f"published {STATS_PATH.name} ({STATS_PATH.stat().st_size / 1e6:.1f} MB) -> release '{tag}'")


def publish_rank_index(tag: str = DEFAULT_TAG, path: Path = RANK_INDEX_PATH) -> None:
    """Upload a precomputed rank index to the release so an API with RANK_INDEX_URL set loads the
    tag->tier lookup instead of building a ~200 MB dict from the full dataset. Run after
    scripts/export_rank_index.py (the crawler does this each cycle).

    ``path`` picks the container — ``RANK_INDEX_NPZ_PATH`` (current) or ``RANK_INDEX_PATH``
    (legacy gzipped JSON, still published during the migration so reverting RANK_INDEX_URL lands
    on a *fresh* artifact rather than a frozen one). ``gh`` names the Release asset after the
    file's basename, so this argument alone decides which asset is written."""
    if not path.exists():
        raise FileNotFoundError(
            f"No rank index at {path} — build it first (scripts/export_rank_index.py).")
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(path), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload (rank index) failed: {res.stderr.strip()}")
    print(f"published {path.name} ({path.stat().st_size / 1e6:.1f} MB) -> release '{tag}'")


def publish_meta_report(tag: str = DEFAULT_TAG) -> None:
    """Upload the meta-drift report (meta_report.json, a few KB) to the release so an API with
    META_REPORT_URL set *serves* it instead of recomputing drift over the full dataset per
    request — two streaming passes over every match, minutes on a small cloud CPU. The crawler
    writes + publishes it after each cycle's meta check (see scripts/collect.py)."""
    if not META_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"No meta report at {META_REPORT_PATH} — the crawler's meta check writes it.")
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(META_REPORT_PATH), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload (meta report) failed: {res.stderr.strip()}")
    print(f"published {META_REPORT_PATH.name} ({META_REPORT_PATH.stat().st_size / 1e3:.1f} KB) -> release '{tag}'")


def publish_itemstats(tag: str = DEFAULT_TAG) -> None:
    """Upload the per-item win-rate table (itemstats.json.gz) to the release so an API with
    ITEMSTATS_URL set serves data-driven loadout picks instead of the effect heuristic. Run after
    scripts/export_itemstats.py (which needs the collected ownership profiles)."""
    if not ITEMSTATS_PATH.exists():
        raise FileNotFoundError(
            f"No item stats at {ITEMSTATS_PATH} — build them first (scripts/export_itemstats.py).")
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(ITEMSTATS_PATH), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload (itemstats) failed: {res.stderr.strip()}")
    print(f"published {ITEMSTATS_PATH.name} ({ITEMSTATS_PATH.stat().st_size / 1e3:.1f} KB) -> release '{tag}'")


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish the dataset and/or model/stats/rank index to a GitHub Release.")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="release tag to upload to")
    ap.add_argument("--model", action="store_true", help="also upload winprob.npz (the model)")
    ap.add_argument("--stats", action="store_true", help="also upload stats.json.gz (precomputed stats)")
    ap.add_argument("--rank", action="store_true", help="also upload rank_index.json.gz (legacy rank index)")
    ap.add_argument("--rank-npz", action="store_true", help="also upload rank_index.npz (current rank index)")
    ap.add_argument("--meta", action="store_true", help="also upload meta_report.json (drift report)")
    ap.add_argument("--itemstats", action="store_true", help="also upload itemstats.json.gz (per-item win rates)")
    ap.add_argument("--only-model", action="store_true", help="upload only winprob.npz, not the dataset")
    ap.add_argument("--only-stats", action="store_true", help="upload only stats.json.gz, not the dataset")
    ap.add_argument("--only-rank", action="store_true", help="upload only rank_index.json.gz, not the dataset")
    ap.add_argument("--only-rank-npz", action="store_true", help="upload only rank_index.npz, not the dataset")
    ap.add_argument("--only-meta", action="store_true", help="upload only meta_report.json, not the dataset")
    ap.add_argument("--only-itemstats", action="store_true", help="upload only itemstats.json.gz, not the dataset")
    args = ap.parse_args()
    if args.only_model:
        publish_model(args.tag)
        return
    if args.only_stats:
        publish_stats(args.tag)
        return
    if args.only_rank:
        publish_rank_index(args.tag)
        return
    if args.only_rank_npz:
        publish_rank_index(args.tag, RANK_INDEX_NPZ_PATH)
        return
    if args.only_meta:
        publish_meta_report(args.tag)
        return
    if args.only_itemstats:
        publish_itemstats(args.tag)
        return
    publish(args.tag)
    if args.model:
        publish_model(args.tag)
    if args.stats:
        publish_stats(args.tag)
    if args.rank:
        publish_rank_index(args.tag)
    if args.rank_npz:
        publish_rank_index(args.tag, RANK_INDEX_NPZ_PATH)
    if args.meta:
        publish_meta_report(args.tag)
    if args.itemstats:
        publish_itemstats(args.tag)


if __name__ == "__main__":
    main()
