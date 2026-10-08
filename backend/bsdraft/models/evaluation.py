"""Training-only evaluation helpers; not imported by the API serving path."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from bsdraft.models.releases import sha256_file


def temporal_split(timestamps, *, selection_frac=0.15, test_frac=0.15,
                   previous_data_through_ts=0, min_train=100, min_selection=100,
                   min_test=1000):
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.ndim != 1 or not len(ts) or np.any(ts <= 0):
        raise ValueError("temporal evaluation requires positive timestamps on every row")
    if not 0 < selection_frac < 0.5 or not 0 < test_frac < 0.5:
        raise ValueError("selection and test fractions must lie between 0 and 0.5")
    order = np.argsort(ts, kind="stable")
    boundary = int(ts[order[min(len(ts) - 1, int(len(ts) * (1 - test_frac)))]] )
    boundary = max(boundary, int(previous_data_through_ts) + 1)
    test = order[ts[order] >= boundary]
    development = order[ts[order] < boundary]
    if len(development):
        selected = min(len(development) - 1, max(0, len(development) - int(len(ts) * selection_frac)))
        selection_start = int(ts[development[selected]])
        selection = development[ts[development] >= selection_start]
        train = development[ts[development] < selection_start]
    else:
        train = selection = np.array([], dtype=np.int64)
    for name, rows, floor in (("train", train, min_train), ("selection", selection, min_selection),
                              ("test", test, min_test)):
        if len(rows) < floor:
            raise ValueError(f"not enough {name} rows: {len(rows)} < {floor}; collect fresh data")
    return train, selection, test


def calibration_error(probs, labels, n_bins=10):
    probs, labels = np.asarray(probs), np.asarray(labels)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    return float(sum(np.mean(bins == index) * abs(probs[bins == index].mean() -
                     labels[bins == index].mean()) for index in range(n_bins) if np.any(bins == index)))


def metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if not len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("evaluation needs both outcome classes")
    if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("invalid model predictions")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("model predictions outside [0, 1]")
    return {"logloss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "acc": float(((probabilities > 0.5) == labels.astype(bool)).mean()),
            "auc": float(roc_auc_score(labels, probabilities)),
            "ece": calibration_error(probabilities, labels)}


def regression_gate(candidate: dict, incumbent: dict | None, *, require_incumbent: bool,
                    max_full_delta: float, incumbent_sha256: str = "") -> dict:
    if require_incumbent and (incumbent is None or not incumbent_sha256):
        raise ValueError("publication requires an evaluated released incumbent")
    if require_incumbent and (not np.isfinite(max_full_delta) or max_full_delta < 0):
        raise ValueError("mandatory publication gate cannot be disabled")
    for record in (candidate, incumbent):
        if record is not None and any(not np.isfinite(record[k]) for k in ("logloss", "auc", "ece")):
            raise ValueError("non-finite evaluation metrics")
    delta = candidate["logloss"] - incumbent["logloss"] if incumbent else None
    if delta is not None and 0 <= max_full_delta < delta:
        raise ValueError(f"full-comp regression gate: paired test logloss delta {delta:+.6f} "
                         f"exceeds {max_full_delta}; no artifacts written")
    return {"passed": incumbent is not None and max_full_delta >= 0,
            "require_incumbent": require_incumbent, "max_full_delta": max_full_delta,
            "delta": delta, "incumbent_sha256": incumbent_sha256,
            "calibration_policy": "reported; no unvalidated ECE or AUC threshold",
            "single_pick_policy": "diagnostic; empirical 1v0 loss is reported, not a publication gate"}


def load_incumbent(path: Path | None, *, required=False, allow_legacy=False):
    if path is None:
        if required:
            raise ValueError("--require-incumbent requires --incumbent-npz")
        return None, {}, ""
    path = Path(path)
    # Pinned ids are mandatory: a positional legacy model cannot be compared safely after catalog drift.
    with np.load(path, allow_pickle=False) as archive:
        required_keys = {"_config", "_vocab_brawler_ids", "_vocab_map_ids", "_vocab_map_rows",
                         "_vocab_modes", "_vocab_mode_rows", "brawler.weight", "map_emb.weight",
                         "mode_emb.weight", "counter_p.weight", "counter_q.weight",
                         "strength.0.weight", "strength.0.bias", "strength.3.weight", "strength.3.bias"}
        if not required_keys <= set(archive.files):
            raise ValueError("incumbent lacks pinned vocabulary or required weights")
        for key in archive.files:
            if not key.startswith("_") and not np.isfinite(archive[key]).all():
                raise ValueError("incumbent contains non-finite weights")
        evaluation = json.loads(archive["_evaluation"].item()) if "_evaluation" in archive else {}
        if not isinstance(evaluation, dict):
            raise ValueError("invalid incumbent evaluation metadata")
        cfg = json.loads(archive["_config"].item())
        dimensions = ("num_brawlers", "num_maps", "num_modes", "d_brawler", "d_map", "d_mode", "d_hidden", "counter_rank")
        if any(type(cfg.get(key)) is not int or cfg[key] <= 0 for key in dimensions):
            raise ValueError("incumbent declares invalid model dimensions")
        mask = cfg.get("mask_row")
        if mask is not None and (type(mask) is not int or mask != cfg["num_brawlers"]):
            raise ValueError("incumbent mask row is malformed")
        rows = cfg["num_brawlers"] + int(mask is not None)
        expected_shapes = {
            "brawler.weight": (rows, cfg["d_brawler"]),
            "map_emb.weight": (cfg["num_maps"], cfg["d_map"]),
            "mode_emb.weight": (cfg["num_modes"], cfg["d_mode"]),
            "counter_p.weight": (rows, cfg["counter_rank"]),
            "counter_q.weight": (rows, cfg["counter_rank"]),
            "strength.0.weight": (cfg["d_hidden"], cfg["d_brawler"] + cfg["d_map"] + cfg["d_mode"]),
            "strength.0.bias": (cfg["d_hidden"],), "strength.3.weight": (1, cfg["d_hidden"]),
            "strength.3.bias": (1,),
        }
        if cfg.get("class_synergy"):
            from bsdraft.constants import BRAWLER_CLASSES
            expected_shapes.update(class_syn=(len(BRAWLER_CLASSES), len(BRAWLER_CLASSES)), brawler_class=(rows,))
        if any(key not in archive or archive[key].shape != shape for key, shape in expected_shapes.items()):
            raise ValueError("incumbent tensor shapes differ from declared architecture")
        brawler_ids = archive["_vocab_brawler_ids"]
        if (brawler_ids.ndim != 1 or len(set(brawler_ids.tolist())) != len(brawler_ids) or
                len(brawler_ids) != cfg["num_brawlers"]):
            raise ValueError("incumbent brawler vocabulary is malformed")
        for ids_key, rows_key, size in (("_vocab_map_ids", "_vocab_map_rows", cfg["num_maps"]),
                                         ("_vocab_modes", "_vocab_mode_rows", cfg["num_modes"])):
            ids, rows = archive[ids_key], archive[rows_key]
            if (ids.ndim != 1 or rows.ndim != 1 or len(ids) != len(rows) or
                    len(set(ids.tolist())) != len(ids) or np.any(rows <= 0) or np.any(rows >= size)):
                raise ValueError("incumbent context vocabulary is malformed")
        if cfg.get("class_synergy") and not {"class_syn", "brawler_class"} <= set(archive.files):
            raise ValueError("incumbent class synergy weights are missing")
        if "data_through_ts" in evaluation and (type(evaluation["data_through_ts"]) is not int or evaluation["data_through_ts"] <= 0):
            raise ValueError("invalid incumbent snapshot timestamp")
        if not evaluation.get("data_through_ts") and not allow_legacy:
            raise ValueError("legacy incumbent has unknown evaluation overlap; bootstrap explicitly "
                             "with --allow-legacy-incumbent once")
    from bsdraft.models.serve import WinProbModel
    model = WinProbModel(path)
    if not model.available:
        raise ValueError("incumbent model could not load")
    # Force actual inference before expensive training; catches malformed tensor dimensions.
    bids = list(model._brawler_rows)
    maps = list(model._map_rows)
    modes = list(model._mode_rows)
    if len(bids) < 6 or not maps or not modes:
        raise ValueError("incumbent pinned vocabulary is empty or insufficient")
    probe = model.prob(bids[:3], bids[3:6], maps[0], modes[0])
    if not np.isfinite(probe):
        raise ValueError("incumbent inference is non-finite")
    return model, evaluation, sha256_file(path)


def predict_incumbent(model, ds, rows, *, brawler_ids, map_ids, modes):
    """Convert current encoder rows back to ids; incumbent resolves its own pinned rows."""
    predictions = np.empty(len(rows), dtype=np.float64)
    groups = {}
    for output, row in enumerate(rows):
        groups.setdefault((int(ds.map_idx[row]), int(ds.mode_idx[row])), []).append((output, int(row)))
    for (map_row, mode_row), members in groups.items():
        for start in range(0, len(members), 8192):
            batch = members[start:start + 8192]
            a = [[int(brawler_ids[i]) for i in ds.team_a[row]] for _, row in batch]
            b = [[int(brawler_ids[i]) for i in ds.team_b[row]] for _, row in batch]
            probs = model.prob_batch(a, b, int(map_ids[map_row]), str(modes[mode_row]))
            predictions[[output for output, _ in batch]] = probs
    return predictions


@contextmanager
def serving_map_context(model, map_train_rows, minimum=100):
    """Evaluate the exact exported map behavior, including its mean fallback for thin rows.

    Only inference runs inside this context. The checkpoint retains original parameters;
    the exporter pins the same supported rows and the NumPy server computes their mean.
    """
    import torch
    counts = np.asarray(map_train_rows)
    learned = np.flatnonzero(counts >= minimum)
    unknown = np.flatnonzero(counts < minimum)
    if not len(learned):
        raise ValueError("no map has enough training rows for a serving export")
    with torch.no_grad():
        original = model.map_emb.weight[unknown].clone()
        mean = model.map_emb.weight[learned].mean(dim=0)
        model.map_emb.weight[unknown] = mean
    try:
        yield
    finally:
        with torch.no_grad():
            model.map_emb.weight[unknown] = original
