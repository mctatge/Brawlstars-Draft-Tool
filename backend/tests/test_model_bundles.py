"""Verified serving bundles use synthetic weights and never contact a remote service."""
import copy
import hashlib
import importlib.util
import io
import json
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from bsdraft.constants import REPO_ROOT
from bsdraft.data import encoders as E
from bsdraft.data.balance_eras import current_balance_era
from bsdraft.models import bundles, serve
from bsdraft.models.winprob import ModelConfig, WinProbNet


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _artifact(payload, name):
    return {"url": f"https://github.com/example/draft/releases/download/model-synthetic/{name}",
            "sha256": _sha(payload), "size_bytes": len(payload)}


@pytest.fixture
def candidate(tmp_path):
    """Include class buffers, whose array size must not inflate learned parameter count."""
    cfg = ModelConfig(E.num_brawlers(), E.num_maps(), E.num_modes(),
                      mask_row=E.num_brawlers(), class_synergy=True)
    net = WinProbNet(cfg).eval()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
    era = current_balance_era()
    analysis = {"era_id": era.id, "start_ts": era.start_ts}
    trained = datetime.fromtimestamp(era.start_ts + 20000, timezone.utc)
    evaluation = {
        "trained_at": trained.isoformat(), "source_commit": "a" * 40,
        "source_dirty": False, "dataset_sha256": "b" * 64,
        "training_run_id": "synthetic-serving-test", "n_train": 7000,
        "n_selection": 1500, "n_test": 1500, "n_total": 10000,
        "embedding": {"logloss": .693147, "acc": .5, "auc": .5, "ece": 0.0},
        "data_through_ts": era.start_ts + 10000, "training_until_ts": era.start_ts + 7000,
        "selection_until_ts": era.start_ts + 8500, "test_start_ts": era.start_ts + 8501,
        "baseline_released_incumbent": {"logloss": .693147, "acc": .5, "auc": .5, "ece": 0.0},
        "publication_gate": {"passed": True, "require_incumbent": True, "delta": 0.0,
                             "max_full_delta": .0035, "incumbent_sha256": "c" * 64,
                             "incumbent_test_overlap": "fresh_after_previous_snapshot"},
        "evaluation_reservation": {"schema_version": 1, "reservation_id": "synthetic-reservation",
                                   "test_after_ts": era.start_ts + 8000,
                                   "consumed_through_ts": era.start_ts + 10000,
                                   "dataset_sha256": "b" * 64, "source_commit": "a" * 40},
    }
    spec = importlib.util.spec_from_file_location(
        "bundle_test_export", REPO_ROOT / "backend/scripts/export_model.py")
    exporting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporting)
    arrays = {key: value.numpy() for key, value in net.state_dict().items()}
    arrays.update(exporting._vocab())
    arrays.update(_config=np.array(json.dumps(cfg.to_dict())),
                  _analysis=np.array(json.dumps(analysis)),
                  _evaluation=np.array(json.dumps(evaluation)))
    path = tmp_path / "synthetic.npz"
    np.savez(path, **arrays)
    model_bytes = path.read_bytes()
    metrics = {**evaluation, "analysis": analysis, "weights_sha256": _sha(model_bytes),
               "evaluation_kind": "final temporal test; not candidate selection",
               "model": {"parameters": sum(parameter.numel() for parameter in net.parameters()),
                         "config": cfg.to_dict(), "pinned_maps": len(arrays["_vocab_map_ids"]),
                         "tensor_shapes": {key: list(value.shape) for key, value in arrays.items()
                                           if not key.startswith("_")}}}
    metrics_bytes = json.dumps(metrics).encode()
    manifest = {"schema_version": 1, "release_id": "model-synthetic",
                "published_at": (trained + timedelta(minutes=5)).isoformat(),
                "trained_at": evaluation["trained_at"], "source_commit": evaluation["source_commit"],
                "dataset_sha256": evaluation["dataset_sha256"], "analysis": analysis,
                "model": _artifact(model_bytes, "winprob.npz"),
                "metrics": _artifact(metrics_bytes, "metrics.json")}
    return SimpleNamespace(manifest=manifest, metrics=metrics,
                           payloads={manifest["model"]["url"]: model_bytes,
                                     manifest["metrics"]["url"]: metrics_bytes})


def _remote(monkeypatch, candidate):
    """Exercise pointer parsing and staging while replacing every network boundary."""
    real_client = httpx.Client
    def request(request):
        assert request.url == "https://api.github.com/repos/example/draft/releases/tags/model-current"
        return httpx.Response(200, json={"body": json.dumps(candidate.manifest)})
    monkeypatch.setattr(bundles.httpx, "Client", lambda **kwargs:
                        real_client(transport=httpx.MockTransport(request), **kwargs))
    def download(url, destination, **kwargs):
        payload = candidate.payloads[url]
        assert len(payload) == kwargs["size_bytes"] <= kwargs["max_bytes"]
        assert _sha(payload) == kwargs["sha256"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    monkeypatch.setattr(bundles, "download_verified", download)
    monkeypatch.setattr(bundles, "_status", {})
    return "https://api.github.com/repos/example/draft/releases/tags/model-current"


def test_class_synergy_bundle_promotes_matching_model_and_metrics(tmp_path, monkeypatch, candidate):
    url = _remote(monkeypatch, candidate)
    directory = tmp_path / "cache"
    assert bundles.refresh(url, directory) is True
    model = bundles.load_current(directory)
    assert model.available and model.supports_partial and model.cfg["class_synergy"]
    assert model.release_info["metrics"] == candidate.metrics
    assert model.release_info["sha256"] == candidate.manifest["model"]["sha256"]
    assert bundles.status()["error"] is None
    assert bundles.refresh(url, directory) is False


@pytest.mark.parametrize("failure", ["model", "metrics_identity", "metrics_values", "embedded_evaluation"])
def test_invalid_replacement_keeps_active_pointer(tmp_path, monkeypatch, candidate, failure):
    url = _remote(monkeypatch, candidate)
    directory = tmp_path / "cache"
    assert bundles.refresh(url, directory)
    pointer = (directory / "current.json").read_bytes()
    original_manifest = copy.deepcopy(candidate.manifest)
    candidate.manifest["release_id"] = "model-invalid"
    if failure == "model":
        broken = b"HTTP 200, but not a NumPy archive"
        candidate.payloads[candidate.manifest["model"]["url"]] = broken
        candidate.manifest["model"] = _artifact(broken, "winprob.npz")
        candidate.metrics["weights_sha256"] = _sha(broken)
    elif failure == "metrics_identity":
        candidate.metrics["weights_sha256"] = "c" * 64
    elif failure == "metrics_values":
        candidate.metrics["embedding"]["logloss"] = float("nan")
    else:
        candidate.metrics["training_run_id"] = "different-run"
    payload = json.dumps(candidate.metrics).encode()
    candidate.payloads[candidate.manifest["metrics"]["url"]] = payload
    candidate.manifest["metrics"] = _artifact(payload, "metrics.json")
    assert bundles.refresh(url, directory) is False
    assert (directory / "current.json").read_bytes() == pointer
    model = bundles.load_current(directory)
    assert model.available
    assert model.release_info["sha256"] == original_manifest["model"]["sha256"]
    assert bundles.status()["error"] is not None


def test_corrupt_local_pointer_recovers_from_verified_remote(tmp_path, monkeypatch, candidate):
    url = _remote(monkeypatch, candidate)
    directory = tmp_path / "cache"
    directory.mkdir()
    (directory / "current.json").write_text("{corrupt")
    assert bundles.load_current(directory) is None
    assert bundles.refresh(url, directory) is True
    assert json.loads((directory / "current.json").read_text()) == candidate.manifest
    assert bundles.load_current(directory).available


def test_model_endpoint_reports_only_loaded_bundle_statistics(tmp_path, monkeypatch, candidate):
    from bsdraft.api import main
    url = _remote(monkeypatch, candidate)
    directory = tmp_path / "cache"
    assert bundles.refresh(url, directory)
    model = bundles.load_current(directory)
    monkeypatch.setattr(main, "_engine", SimpleNamespace(model=model))
    response = TestClient(main.app).get("/api/model")
    assert response.status_code == 200
    result = response.json()
    assert result["available"] is True and result["evaluation_status"] == "verified_bundle"
    assert result["metrics"] == candidate.metrics
    assert result["sha256"] == candidate.manifest["model"]["sha256"]
    assert result["release_id"] == candidate.manifest["release_id"]
    assert result["published_at"] == candidate.manifest["published_at"]
    assert result["note"] is None


def test_legacy_available_model_has_no_verified_statistics(tmp_path, monkeypatch, candidate):
    from bsdraft.api import main
    path = tmp_path / "legacy.npz"
    path.write_bytes(candidate.payloads[candidate.manifest["model"]["url"]])
    model = serve.WinProbModel(path)
    assert model.available
    monkeypatch.setattr(main, "_engine", SimpleNamespace(model=model))
    monkeypatch.setattr(main.sync, "MODEL_PATH", path)
    monkeypatch.setattr(main.sync, "_MODEL_SHA_PATH", tmp_path / "no-cached-digest")
    result = TestClient(main.app).get("/api/model").json()
    assert result["available"] is True
    assert result["metrics"] is None and result["evaluation_status"] == "unavailable"
    assert result["release_id"] is None
    assert "Historical reports are not current" in result["note"]


def test_corrupt_cold_model_degrades_to_empirical_stats(tmp_path, monkeypatch):
    from bsdraft.api import main
    path = tmp_path / "winprob.npz"
    path.write_bytes(b"corrupt cold archive")
    monkeypatch.setattr(serve, "DEFAULT_PATH", path)
    monkeypatch.setattr(main.sync, "MODEL_PATH", path)
    monkeypatch.setattr(main.settings, "model_manifest_url", "")
    model = main._load_model()
    assert model.available is False
    assert model.prob([], [], 0, "Gem Grab") == .5


def _repack(candidate, *, sync_config=False):
    """Keep bytes/digests/embedded evidence coherent while testing semantic validation."""
    model_url = candidate.manifest["model"]["url"]
    with np.load(io.BytesIO(candidate.payloads[model_url]), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if sync_config:
        arrays["_config"] = np.array(json.dumps(candidate.metrics["model"]["config"]))
    old_evaluation = json.loads(arrays["_evaluation"].item())
    evaluation = {key: candidate.metrics[key] for key in old_evaluation if key in candidate.metrics}
    arrays["_evaluation"] = np.array(json.dumps(evaluation))
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    payload = stream.getvalue()
    candidate.payloads[model_url] = payload
    candidate.manifest["model"] = _artifact(payload, "winprob.npz")
    candidate.metrics["weights_sha256"] = _sha(payload)
    candidate.manifest["trained_at"] = candidate.metrics.get("trained_at", "missing")
    payload = json.dumps(candidate.metrics).encode()
    candidate.manifest["metrics"] = _artifact(payload, "metrics.json")
    candidate.payloads[candidate.manifest["metrics"]["url"]] = payload


@pytest.mark.parametrize("failure", [
    "missing_test_start", "missing_evaluation_kind", "missing_gate_delta", "missing_accuracy",
    "missing_overlap", "nonfinite_threshold", "wrong_delta", "missing_baseline",
    "short_test", "boolean_count", "overlapping_timestamps", "out_of_range_timestamp",
    "bad_trained_date", "bad_published_date", "publication_before_training", "dirty_source",
    "wrong_reservation_digest", "wrong_reservation_watermark", "wrong_reservation_source",
    "consumed_test_overlap", "bad_incumbent_digest", "wrong_config", "declared_architecture", "wrong_shapes", "wrong_map_count",
])
def test_self_consistent_malformed_report_never_replaces_valid_bundle(tmp_path, monkeypatch, candidate, failure):
    url = _remote(monkeypatch, candidate)
    directory = tmp_path / "cache"
    assert bundles.refresh(url, directory)
    pointer = (directory / "current.json").read_bytes()
    report = candidate.metrics
    if failure == "missing_test_start":
        del report["test_start_ts"]
    elif failure == "missing_evaluation_kind":
        del report["evaluation_kind"]
    elif failure == "missing_gate_delta":
        del report["publication_gate"]["delta"]
    elif failure == "missing_accuracy":
        del report["embedding"]["acc"]
    elif failure == "missing_overlap":
        del report["publication_gate"]["incumbent_test_overlap"]
    elif failure == "nonfinite_threshold":
        report["publication_gate"]["max_full_delta"] = float("inf")
    elif failure == "wrong_delta":
        report["publication_gate"]["delta"] = -.001
    elif failure == "missing_baseline":
        del report["baseline_released_incumbent"]
    elif failure == "short_test":
        report["n_test"] = 999
        report["n_total"] = report["n_train"] + report["n_selection"] + report["n_test"]
    elif failure == "boolean_count":
        report["n_train"] = True
    elif failure == "overlapping_timestamps":
        report["test_start_ts"] = report["selection_until_ts"]
    elif failure == "out_of_range_timestamp":
        report["data_through_ts"] = 10**20
        report["evaluation_reservation"]["consumed_through_ts"] = 10**20
    elif failure == "bad_trained_date":
        report["trained_at"] = "unparseable"
    elif failure == "bad_published_date":
        candidate.manifest["published_at"] = "unparseable"
    elif failure == "publication_before_training":
        candidate.manifest["published_at"] = (datetime.fromisoformat(report["trained_at"]) - timedelta(seconds=1)).isoformat()
    elif failure == "dirty_source":
        report["source_dirty"] = True
    elif failure == "wrong_reservation_digest":
        report["evaluation_reservation"]["dataset_sha256"] = "d" * 64
    elif failure == "wrong_reservation_watermark":
        report["evaluation_reservation"]["consumed_through_ts"] -= 1
    elif failure == "wrong_reservation_source":
        report["evaluation_reservation"]["source_commit"] = "d" * 40
    elif failure == "consumed_test_overlap":
        report["evaluation_reservation"]["test_after_ts"] = report["test_start_ts"]
    elif failure == "bad_incumbent_digest":
        report["publication_gate"]["incumbent_sha256"] = "bad-digest"
    elif failure in ("wrong_config", "declared_architecture"):
        report["model"]["config"]["counter_rank"] += 1
    elif failure == "wrong_shapes":
        report["model"]["tensor_shapes"]["brawler.weight"] = [1, 1]
    elif failure == "wrong_map_count":
        report["model"]["pinned_maps"] += 1
    _repack(candidate, sync_config=failure == "declared_architecture")
    assert bundles.refresh(url, directory) is False
    assert (directory / "current.json").read_bytes() == pointer
    assert bundles.load_current(directory).available


@pytest.mark.parametrize("missing, cached, expected", [(False, False, False), (False, True, False),
                                                        (True, True, False), (True, False, True)])
def test_legacy_download_only_before_first_bundle_pointer(tmp_path, monkeypatch, missing, cached, expected):
    from bsdraft.api import main
    monkeypatch.setattr(main.settings, "model_manifest_url", "https://example.invalid/model-current")
    monkeypatch.setattr(main.settings, "model_url", "https://example.invalid/legacy.npz")
    monkeypatch.setattr(main.bundles, "refresh", lambda url: False)
    monkeypatch.setattr(main.bundles, "status", lambda: {"missing": missing})
    monkeypatch.setattr(main.bundles, "load_current", lambda: object() if cached else None)
    calls = []
    monkeypatch.setattr(main.sync, "sync_model", lambda url: calls.append(url) or True)
    assert main._sync_model() is expected
    assert bool(calls) is expected
