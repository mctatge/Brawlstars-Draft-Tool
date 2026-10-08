"""Load immutable model/metrics bundles, promoting one local pointer after validation."""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
import time
from pathlib import Path

import httpx
import numpy as np

from bsdraft.constants import BRAWLER_CLASSES, PROCESSED_DIR
from bsdraft.models.releases import download_verified, parse_manifest, sha256_file
from bsdraft.models.serve import WinProbModel

logger = logging.getLogger(__name__)
BUNDLE_DIR = PROCESSED_DIR / "model-bundles"
_status: dict = {}


def status() -> dict:
    return dict(_status)


def _paths(manifest: dict, directory: Path):
    # Only validated digests determine filesystem paths; remote labels never do.
    folder = directory / (manifest["model"]["sha256"] + "-" + manifest["metrics"]["sha256"])
    return folder / "winprob.npz", folder / "metrics.json"


def _finite_number(value, *, minimum=None, maximum=None) -> bool:
    return (type(value) in (int, float) and math.isfinite(value) and
            (minimum is None or value >= minimum) and
            (maximum is None or value <= maximum))


def _date_timestamp(value) -> float:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing model date")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("model date must include a timezone")
    timestamp = parsed.timestamp()
    if not _finite_number(timestamp, minimum=1):
        raise ValueError("invalid model date")
    return timestamp


def _validate_report(metrics: dict, manifest: dict) -> None:
    """Validate the complete public report without importing training dependencies."""
    if not isinstance(metrics, dict):
        raise ValueError("model metrics must be an object")
    if metrics.get("weights_sha256") != manifest["model"]["sha256"]:
        raise ValueError("metrics describe different weights")
    if metrics.get("analysis") != manifest["analysis"]:
        raise ValueError("metrics analysis era mismatch")
    for key in ("trained_at", "source_commit", "dataset_sha256"):
        if metrics.get(key) != manifest[key]:
            raise ValueError("metrics provenance mismatch")
    trained = _date_timestamp(metrics.get("trained_at"))
    published = _date_timestamp(manifest["published_at"])
    if published < trained:
        raise ValueError("publication predates training")
    for key in ("training_run_id", "evaluation_kind"):
        if not isinstance(metrics.get(key), str) or not metrics[key].strip():
            raise ValueError("missing evaluation description or identity")
    if metrics.get("source_dirty") is not False:
        raise ValueError("model source checkout was not clean")
    for key, floor in (("n_train", 100), ("n_selection", 100), ("n_test", 1000), ("n_total", 1200)):
        if type(metrics.get(key)) is not int or not floor <= metrics[key] <= 2**53 - 1:
            raise ValueError("invalid evaluation sample count")
    if sum(metrics[key] for key in ("n_train", "n_selection", "n_test")) != metrics["n_total"]:
        raise ValueError("evaluation counts disagree")
    for key in ("training_until_ts", "selection_until_ts", "test_start_ts", "data_through_ts"):
        if type(metrics.get(key)) is not int or not 0 < metrics[key] <= trained:
            raise ValueError("invalid evaluation timestamp")
    if not (manifest["analysis"]["start_ts"] <= metrics["training_until_ts"] <
            metrics["selection_until_ts"] < metrics["test_start_ts"] <= metrics["data_through_ts"]):
        raise ValueError("evaluation timestamps overlap or precede the active era")
    for name in ("embedding", "baseline_released_incumbent"):
        record = metrics.get(name)
        if not isinstance(record, dict):
            raise ValueError("missing evaluated model metrics")
        for key in ("logloss", "acc", "auc", "ece"):
            if not _finite_number(record.get(key), minimum=0,
                                  maximum=None if key == "logloss" else 1):
                raise ValueError("invalid evaluation metrics")
    gate = metrics.get("publication_gate")
    if (not isinstance(gate, dict) or gate.get("passed") is not True or
            gate.get("require_incumbent") is not True):
        raise ValueError("bundle lacks mandatory publication gate")
    if (not re.fullmatch(r"[0-9a-f]{64}", str(gate.get("incumbent_sha256", ""))) or
            not isinstance(gate.get("incumbent_test_overlap"), str) or
            not gate["incumbent_test_overlap"].strip()):
        raise ValueError("missing paired incumbent identity or overlap policy")
    if (not _finite_number(gate.get("delta")) or
            not _finite_number(gate.get("max_full_delta"), minimum=0) or
            gate["delta"] > gate["max_full_delta"]):
        raise ValueError("invalid or failed publication gate")
    expected_delta = metrics["embedding"]["logloss"] - metrics["baseline_released_incumbent"]["logloss"]
    if not math.isclose(gate["delta"], expected_delta, rel_tol=0, abs_tol=1e-12):
        raise ValueError("paired delta disagrees with final-test metrics")
    reservation = metrics.get("evaluation_reservation")
    if (not isinstance(reservation, dict) or reservation.get("schema_version") != 1 or
            not isinstance(reservation.get("reservation_id"), str) or not reservation["reservation_id"].strip() or
            type(reservation.get("test_after_ts")) is not int or reservation["test_after_ts"] < 0 or
            type(reservation.get("consumed_through_ts")) is not int or
            reservation["consumed_through_ts"] != metrics["data_through_ts"] or
            reservation.get("dataset_sha256") != metrics["dataset_sha256"] or
            reservation.get("source_commit") != metrics["source_commit"] or
            reservation["test_after_ts"] >= metrics["test_start_ts"]):
        raise ValueError("missing or inconsistent consumed-test reservation")


def _validate_architecture(archive, cfg: dict) -> None:
    dimensions = ("num_brawlers", "num_maps", "num_modes", "d_brawler", "d_map", "d_mode", "d_hidden", "counter_rank")
    if not isinstance(cfg, dict) or any(type(cfg.get(key)) is not int or cfg[key] <= 0 for key in dimensions):
        raise ValueError("invalid declared model dimensions")
    mask = cfg.get("mask_row")
    if mask is not None and (type(mask) is not int or mask != cfg["num_brawlers"]):
        raise ValueError("invalid declared mask row")
    rows = cfg["num_brawlers"] + int(mask is not None)
    shapes = {
        "brawler.weight": (rows, cfg["d_brawler"]),
        "map_emb.weight": (cfg["num_maps"], cfg["d_map"]),
        "mode_emb.weight": (cfg["num_modes"], cfg["d_mode"]),
        "counter_p.weight": (rows, cfg["counter_rank"]), "counter_q.weight": (rows, cfg["counter_rank"]),
        "strength.0.weight": (cfg["d_hidden"], cfg["d_brawler"] + cfg["d_map"] + cfg["d_mode"]),
        "strength.0.bias": (cfg["d_hidden"],), "strength.3.weight": (1, cfg["d_hidden"]), "strength.3.bias": (1,),
    }
    if cfg.get("class_synergy"):
        shapes.update(class_syn=(len(BRAWLER_CLASSES), len(BRAWLER_CLASSES)), brawler_class=(rows,))
    if any(key not in archive or archive[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("weights differ from declared architecture")
    brawler_ids = archive["_vocab_brawler_ids"]
    if (brawler_ids.ndim != 1 or brawler_ids.dtype.kind not in "iu" or
            len(brawler_ids) != cfg["num_brawlers"] or len(set(brawler_ids.tolist())) != len(brawler_ids)):
        raise ValueError("invalid pinned brawler vocabulary")
    for ids_key, rows_key, size in (("_vocab_map_ids", "_vocab_map_rows", cfg["num_maps"]),
                                    ("_vocab_modes", "_vocab_mode_rows", cfg["num_modes"])):
        ids, indices = archive[ids_key], archive[rows_key]
        if (ids.ndim != 1 or indices.ndim != 1 or not len(ids) or len(ids) != len(indices) or
                len(set(ids.tolist())) != len(ids) or len(set(indices.tolist())) != len(indices) or
                indices.dtype.kind not in "iu" or np.any(indices <= 0) or np.any(indices >= size)):
            raise ValueError("invalid pinned context vocabulary")


def _validate(manifest: dict, directory: Path) -> WinProbModel:
    manifest = parse_manifest(manifest)
    model_path, metrics_path = _paths(manifest, directory)
    for key, path in (("model", model_path), ("metrics", metrics_path)):
        artifact = manifest[key]
        if path.stat().st_size != artifact["size_bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError("bundle artifact identity mismatch")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _validate_report(metrics, manifest)
    with np.load(model_path, allow_pickle=False) as archive:
        evaluation = json.loads(archive["_evaluation"].item())
        required_evidence = {"training_run_id", "trained_at", "source_commit", "dataset_sha256", "source_dirty",
                             "data_through_ts", "training_until_ts", "selection_until_ts", "test_start_ts",
                             "n_train", "n_selection", "n_test", "n_total", "publication_gate",
                             "evaluation_reservation", "embedding", "baseline_released_incumbent"}
        if (not isinstance(evaluation, dict) or not required_evidence <= evaluation.keys() or
                any(metrics.get(key) != value for key, value in evaluation.items())):
            raise ValueError("embedded evaluation disagrees with metrics")
        if json.loads(archive["_analysis"].item()) != manifest["analysis"]:
            raise ValueError("embedded analysis disagrees with manifest")
        cfg = json.loads(archive["_config"].item())
        _validate_architecture(archive, cfg)
        parameters = 0
        tensor_shapes = {}
        for key in archive.files:
            if not key.startswith("_"):
                if not np.isfinite(archive[key]).all():
                    raise ValueError("non-finite model weights")
                tensor_shapes[key] = list(archive[key].shape)
                if key != "brawler_class":
                    parameters += archive[key].size
        facts = metrics.get("model")
        if (not isinstance(facts, dict) or type(facts.get("parameters")) is not int or
                parameters <= 0 or parameters != facts["parameters"] or
                facts.get("config") != cfg or
                type(facts.get("pinned_maps")) is not int or
                facts["pinned_maps"] != len(archive["_vocab_map_ids"]) or
                facts.get("tensor_shapes") != tensor_shapes):
            raise ValueError("model facts disagree with actual weights")
    model = WinProbModel(model_path, validate_current_era=True)
    if not model.available or model.analysis_era_id != manifest["analysis"]["era_id"]:
        raise ValueError("model is unavailable in the current balance era")
    bids, maps, modes = list(model._brawler_rows), list(model._map_rows), list(model._mode_rows)
    if len(bids) < 6 or not maps or not modes:
        raise ValueError("model lacks pinned reference vocabulary")
    probability = model.prob(bids[:3], bids[3:6], maps[0], modes[0])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("invalid model prediction")
    model.release_info = {"release_id": manifest["release_id"], "sha256": manifest["model"]["sha256"],
                          "published_at": manifest["published_at"], "metrics": metrics}
    return model


def load_current(directory: Path = BUNDLE_DIR) -> WinProbModel | None:
    try:
        manifest = parse_manifest((directory / "current.json").read_text(encoding="utf-8"))
        return _validate(manifest, directory)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("cached model bundle rejected (%s)", type(exc).__name__)
        return None


def refresh(url: str, directory: Path = BUNDLE_DIR) -> bool:
    """Return whether the verified active bundle changed. Keep last-good on every failure.

    A missing remote pointer is distinguished from invalid/unreachable so that the API can
    bootstrap the legacy artifact only before the first versioned bundle is published.
    """
    global _status
    _status = {**_status, "last_attempt": time.time()}
    try:
        with httpx.Client(follow_redirects=True, timeout=15) as client:
            with client.stream("GET", url, headers={"Accept": "application/vnd.github+json"}) as response:
                if response.status_code == 404:
                    _status.update(error="pointer_missing", missing=True)
                    return False
                response.raise_for_status()
                payload = bytearray()
                started = time.monotonic()
                for chunk in response.iter_bytes():
                    if time.monotonic() - started > 15:
                        raise TimeoutError("model pointer deadline exceeded")
                    payload.extend(chunk)
                    if len(payload) > 64 * 1024:
                        raise ValueError("model pointer exceeds size limit")
        manifest = parse_manifest(bytes(payload))
        directory.mkdir(parents=True, exist_ok=True)
        current = directory / "current.json"
        # Revalidate cached files as well; a valid pointer alone is insufficient.
        try:
            old = parse_manifest(current.read_text()) if current.exists() else None
        except (ValueError, OSError):
            old = None
        model_path, metrics_path = _paths(manifest, directory)
        for key, path, limit in (("model", model_path, 8 * 1024 * 1024),
                                 ("metrics", metrics_path, 256 * 1024)):
            artifact = manifest[key]
            if artifact["size_bytes"] > limit:
                raise ValueError("bundle artifact exceeds size limit")
            if not path.exists() or sha256_file(path) != artifact["sha256"]:
                download_verified(artifact["url"], path, sha256=artifact["sha256"],
                                  size_bytes=artifact["size_bytes"], max_bytes=limit)
        _validate(manifest, directory)
        temporary = current.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        temporary.replace(current)
        _status.update(last_success=time.time(), error=None, missing=False,
                       release_id=manifest["release_id"], sha256=manifest["model"]["sha256"])
        return old != manifest
    except Exception as exc:
        _status.update(error=type(exc).__name__, missing=False)
        logger.warning("model bundle rejected (%s); keeping last-good bundle", type(exc).__name__)
        return False
