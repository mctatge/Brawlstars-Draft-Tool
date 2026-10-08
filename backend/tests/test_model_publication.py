"""Failure injection proves publication never deletes/clobbers the incumbent model."""
import json
import subprocess
from pathlib import Path
import urllib.error

import pytest
import numpy as np
import importlib.util
import torch
from bsdraft.constants import REPO_ROOT
from bsdraft.data import encoders as E
from bsdraft.models.winprob import ModelConfig, WinProbNet


def synthetic_incumbent(path, *, through):
    cfg = ModelConfig(E.num_brawlers(), E.num_maps(), E.num_modes(), mask_row=E.num_brawlers(), class_synergy=True)
    model = WinProbNet(cfg).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    spec = importlib.util.spec_from_file_location("export_fixture", REPO_ROOT / "backend/scripts/export_model.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    arrays = {key: value.numpy() for key, value in model.state_dict().items()}
    arrays.update(exporter._vocab())
    arrays["_config"] = np.array(json.dumps(cfg.to_dict()))
    arrays["_evaluation"] = np.array(json.dumps({"data_through_ts": through}))
    np.savez(path, **arrays)
from bsdraft.data.balance_eras import current_balance_era

from bsdraft.collect import publish as P
from bsdraft.models import releases as R


@pytest.fixture
def bundle(tmp_path):
    model, metrics = tmp_path / "winprob.npz", tmp_path / "metrics.json"
    synthetic_incumbent(model, through=current_balance_era().start_ts + 10000)
    report = {"weights_sha256": R.sha256_file(model), "n_test": 1500, "n_train": 7000,
              "n_selection": 1500, "training_until_ts": 100, "selection_until_ts": 200,
              "test_start_ts": 201, "trained_at": "2026-10-08T00:00:00Z", "source_commit": "a" * 40,
              "dataset_sha256": "b" * 64, "analysis": {"era_id": "2026-09-16", "start_ts": 1789516800},
              "embedding": {"logloss": .66, "auc": .63, "ece": .01},
              "publication_gate": {"passed": True, "require_incumbent": True,
                                   "incumbent_sha256": "c" * 64, "delta": .001, "max_full_delta": .0035}}
    start = current_balance_era().start_ts
    report["training_run_id"] = "synthetic-run"
    report["baseline_released_incumbent"] = {"logloss": .659, "auc": .63, "ece": .01, "acc": .59}
    report.update(n_total=10000, source_dirty=False, data_through_ts=start + 10000,
                  training_until_ts=start + 7000, selection_until_ts=start + 8500,
                  test_start_ts=start + 8501, analysis={"era_id": current_balance_era().id, "start_ts": start})
    report["evaluation_reservation"] = {"reservation_id": "synthetic-reservation", "test_after_ts": start + 8000,
                                        "consumed_through_ts": report["data_through_ts"],
                                        "dataset_sha256": report["dataset_sha256"], "source_commit": report["source_commit"]}
    embedded = {key: value for key, value in report.items() if key != "weights_sha256"}
    with np.load(model) as archive:
        arrays = dict(archive)
    arrays["_evaluation"] = np.array(json.dumps(embedded))
    arrays["_analysis"] = np.array(json.dumps(report["analysis"]))
    np.savez(model, **arrays)
    report["model"] = {"config": json.loads(arrays["_config"].item()),
                       "parameters": sum(int(value.size) for key, value in arrays.items()
                                         if not key.startswith("_") and key != "brawler_class"),
                       "pinned_maps": len(arrays["_vocab_map_ids"])}
    report["weights_sha256"] = R.sha256_file(model)
    metrics.write_text(json.dumps(report))
    return model, metrics


class FakeGitHub:
    def __init__(self, bundle, failure=""):
        self.bundle = bundle
        self.failure = failure
        self.calls = []
        self.pointer = {"id": 12, "body": "old pointer survives"}
        self.uploaded = False
    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("release", "view"):
            return subprocess.CompletedProcess(args, 1, "", "not found")
        if args[:2] == ("release", "create"):
            if self.failure == "upload":
                return subprocess.CompletedProcess(args, 1, "", "upload interrupted")
            self.uploaded = True
        if args[:2] == ("release", "download"):
            directory = Path(args[args.index("--dir") + 1])
            for original in self.bundle:
                (directory / original.name).write_bytes(original.read_bytes())
            if self.failure == "digest":
                (directory / "winprob.npz").write_bytes(b"corruption")
        if args[0] == "api" and "PATCH" in args:
            if self.failure == "promotion":
                return subprocess.CompletedProcess(args, 1, "", "PATCH refused")
            request = json.loads(Path(args[args.index("--input") + 1]).read_text())
            self.pointer.update(request)
        if args[0] == "api" and args[-1].endswith("/tags/model-current"):
            return subprocess.CompletedProcess(args, 0, json.dumps(self.pointer), "")
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.mark.parametrize("failure", ["upload", "digest", "promotion"])
def test_publication_failure_retains_incumbent_and_pointer(bundle, monkeypatch, failure):
    fake = FakeGitHub(bundle, failure)
    monkeypatch.setattr(P, "_gh", fake)
    monkeypatch.setenv("GH_REPO", "owner/repository")
    with pytest.raises((ValueError, RuntimeError)):
        P.publish_model_bundle(*bundle)
    assert fake.pointer["body"] == "old pointer survives"
    assert not any("--clobber" in call or "delete" in call for call in fake.calls)
    if failure in ("upload", "digest"):
        assert not any("PATCH" in call for call in fake.calls)


def test_valid_bundle_is_verified_before_atomic_manifest_promotion(bundle, monkeypatch):
    fake = FakeGitHub(bundle)
    monkeypatch.setattr(P, "_gh", fake)
    monkeypatch.setenv("GH_REPO", "owner/repository")
    manifest = P.publish_model_bundle(*bundle)
    assert R.parse_manifest(fake.pointer) == manifest
    assert manifest["model"]["sha256"] == R.sha256_file(bundle[0])
    download = next(i for i, args in enumerate(fake.calls) if args[:2] == ("release", "download"))
    promotion = next(i for i, args in enumerate(fake.calls) if "PATCH" in args)
    assert download < promotion and not any("--clobber" in call for call in fake.calls)


@pytest.mark.parametrize("mutation", ["digest", "no_gate", "disabled_gate", "no_test", "overlap", "nan"])
def test_unqualified_bundle_cannot_reach_network(bundle, monkeypatch, mutation):
    report = json.loads(bundle[1].read_text())
    if mutation == "digest": report["weights_sha256"] = "d" * 64
    if mutation == "no_gate": report["publication_gate"]["require_incumbent"] = False
    if mutation == "disabled_gate": report["publication_gate"]["max_full_delta"] = -1
    if mutation == "no_test": report["n_test"] = 12
    if mutation == "overlap": report["test_start_ts"] = 100
    if mutation == "nan": report["embedding"]["ece"] = float("nan")
    bundle[1].write_text(json.dumps(report))
    monkeypatch.setattr(P, "_gh", lambda *args: pytest.fail("unqualified model reached GitHub"))
    with pytest.raises(ValueError): P.publish_model_bundle(*bundle)


def test_incumbent_download_does_not_fallback_after_network_or_integrity_failure(tmp_path, monkeypatch):
    destination = tmp_path / "incumbent.npz"
    destination.write_bytes(b"last good")
    monkeypatch.setattr(R.urllib.request, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(
        urllib.error.HTTPError("https://api.github.com", 503, "unavailable", {}, None)))
    monkeypatch.setattr(R, "download_verified", lambda *args, **kwargs: pytest.fail("transport error fell back"))
    with pytest.raises(urllib.error.HTTPError): R.download_incumbent("owner/repository", destination)
    assert destination.read_bytes() == b"last good"


def test_only_pointer_404_permits_explicit_legacy_download(tmp_path, monkeypatch):
    monkeypatch.setattr(R.urllib.request, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(
        urllib.error.HTTPError("https://api.github.com", 404, "not found", {}, None)))
    seen = []
    monkeypatch.setattr(R, "download_verified", lambda url, path, **kwargs: seen.append(url))
    assert R.download_incumbent("owner/repository", tmp_path / "incumbent.npz") is None
    assert seen == ["https://github.com/owner/repository/releases/download/data-latest/winprob.npz"]


def test_matching_digest_cannot_make_corrupt_model_publishable(bundle, monkeypatch):
    bundle[0].write_bytes(b"not an npz")
    report = json.loads(bundle[1].read_text())
    report["weights_sha256"] = R.sha256_file(bundle[0])
    bundle[1].write_text(json.dumps(report))
    monkeypatch.setattr(P, "_gh", lambda *args: pytest.fail("corrupt model reached GitHub"))
    with pytest.raises(ValueError):
        P.publish_model_bundle(*bundle)


def test_slow_release_download_has_total_deadline_and_retains_last_good(tmp_path, monkeypatch):
    import io
    destination = tmp_path / "model.npz"
    destination.write_bytes(b"last good")
    monkeypatch.setattr(R.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"slow data"))
    ticks = iter([0.0, 1.0, 61.0])
    monkeypatch.setattr(R.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="total deadline"):
        R.download_verified("https://github.com/owner/repository/releases/download/version/model.npz", destination)
    assert destination.read_bytes() == b"last good"
    assert not destination.with_name(destination.name + ".download").exists()
