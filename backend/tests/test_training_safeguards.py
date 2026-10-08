"""Publication gates exercised without optimization or production files/network access."""
import importlib.util
import json
import sys
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bsdraft.constants import REPO_ROOT
from bsdraft.data import encoders as E
from bsdraft.data.balance_eras import current_balance_era
from bsdraft.models import evaluation as EV
from bsdraft.models.winprob import ModelConfig, WinProbNet


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "backend" / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_incumbent(path, *, through=None):
    cfg = ModelConfig(E.num_brawlers(), E.num_maps(), E.num_modes(),
                      mask_row=E.num_brawlers(), class_synergy=True)
    model = WinProbNet(cfg).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    arrays = {key: value.numpy() for key, value in model.state_dict().items()}
    arrays.update(script("export_model")._vocab())
    arrays["_config"] = np.array(json.dumps(cfg.to_dict()))
    if through is not None:
        arrays["_evaluation"] = np.array(json.dumps({"data_through_ts": through}))
    np.savez(path, **arrays)
    return model


def test_temporal_split_keeps_ties_and_final_test_out_of_selection():
    ts = np.repeat(np.arange(1, 5001), 2)
    train, selection, test = EV.temporal_split(ts, previous_data_through_ts=4500)
    assert len(test) == 1000
    assert ts[train].max() < ts[selection].min() <= ts[selection].max() < ts[test].min()
    assert ts[test].min() > 4500
    assert len(set(train) | set(selection) | set(test)) == len(ts)


def test_not_enough_new_data_refuses_retrain():
    with pytest.raises(ValueError, match="not enough test"):
        EV.temporal_split(np.arange(1, 10001), previous_data_through_ts=9500)


@pytest.mark.parametrize("bad", [0, -1])
def test_missing_timestamps_refuse_temporal_claim(bad):
    with pytest.raises(ValueError, match="positive timestamps"):
        EV.temporal_split([1, 2, bad])


def test_gate_is_fail_closed_and_cannot_be_disabled():
    candidate = {"logloss": .69, "auc": .51, "ece": .01}
    with pytest.raises(ValueError, match="requires an evaluated"):
        EV.regression_gate(candidate, None, require_incumbent=True, max_full_delta=.0035)
    with pytest.raises(ValueError, match="cannot be disabled"):
        EV.regression_gate(candidate, candidate, require_incumbent=True, max_full_delta=-1,
                           incumbent_sha256="a" * 64)
    with pytest.raises(ValueError, match="regression gate"):
        EV.regression_gate(candidate, {**candidate, "logloss": .65}, require_incumbent=True,
                           max_full_delta=.0035, incumbent_sha256="a" * 64)


@pytest.mark.parametrize("path_kind", ["missing", "corrupt"])
def test_unusable_incumbent_is_never_treated_as_bootstrap(tmp_path, path_kind):
    path = tmp_path / "incumbent.npz"
    if path_kind == "corrupt":
        path.write_bytes(b"not a model")
    with pytest.raises((OSError, ValueError)):
        EV.load_incumbent(path, required=True, allow_legacy=True)


def test_legacy_overlap_requires_explicit_acknowledgement(tmp_path):
    path = tmp_path / "incumbent.npz"
    synthetic_incumbent(path)
    with pytest.raises(ValueError, match="bootstrap explicitly"):
        EV.load_incumbent(path, required=True)
    model, evidence, digest = EV.load_incumbent(path, required=True, allow_legacy=True)
    assert model.available and evidence == {} and len(digest) == 64


def test_require_incumbent_fails_before_any_training_even_with_local_pt(tmp_path, monkeypatch):
    training = script("train")
    (tmp_path / "winprob.pt").write_bytes(b"stale local checkpoint must never become incumbent")
    monkeypatch.setattr(training.D, "build_dataset", lambda **kwargs: pytest.fail("dataset read before incumbent check"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--require-incumbent", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="requires --incumbent-npz"):
        training.main()
    assert not (tmp_path / "metrics.json").exists()


def test_full_train_export_path_uses_selection_then_disjoint_test(tmp_path, monkeypatch):
    training = script("train")
    start = current_balance_era().start_ts + 1
    incumbent_path = tmp_path / "incumbent.npz"
    synthetic_incumbent(incumbent_path, through=start + 7999)
    n = 10000
    ds = SimpleNamespace(team_a=np.tile([0, 1, 2], (n, 1)), team_b=np.tile([3, 4, 5], (n, 1)),
                         map_idx=np.ones(n, dtype=np.int64), mode_idx=np.ones(n, dtype=np.int64),
                         y=(np.arange(n) % 2).astype(np.float32), ts=start + np.arange(n),
                         queue_type=np.full(n, "soloRanked"))
    class SizedDataset(SimpleNamespace):
        def __len__(self):
            return len(self.y)
    ds = SizedDataset(**vars(ds))
    monkeypatch.setattr(training.D, "build_dataset", lambda **kwargs: ds)
    calls = []
    def candidate(seed, cfg, args, shared):
        calls.append((shared["tr_i"].copy(), shared["vai"].numpy().copy()))
        model = WinProbNet(cfg).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        result = EV.metrics(shared["yv"], np.full(len(shared["yv"]), .5))
        return {"seed": seed, "model": model, "m_ll": result["logloss"], "m_auc": result["auc"],
                "m_acc": result["acc"], "m_ece": result["ece"], "history_mix": [.69], "history_full": [.69]}
    monkeypatch.setattr(training, "_train_candidate", candidate)
    source = tmp_path / "matches.jsonl"
    source.write_text("synthetic dataset identity\n")
    report_path, output = tmp_path / "metrics.json", tmp_path / "output"
    reservation = tmp_path / "reservation.json"
    from bsdraft.models.releases import sha256_file
    reservation.write_text(json.dumps({"reservation_id": "synthetic-reservation", "dataset_sha256": sha256_file(source),
                                       "test_after_ts": start + 7999, "consumed_through_ts": start + n - 1,
                                       "source_commit": "a" * 40}))
    monkeypatch.setattr(training.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, "a" * 40 if "rev-parse" in args else "", ""))
    argv = ["train.py", "--require-incumbent", "--evaluation-reservation", str(reservation), "--incumbent-npz", str(incumbent_path),
            "--matches", str(source), "--output-dir", str(output), "--metrics", str(report_path),
            "--candidates", "3", "--no-charts"]
    monkeypatch.setattr(sys, "argv", argv)
    training.main()
    report = json.loads(report_path.read_text())
    assert report["n_train"] == 7000 and report["n_selection"] == 1500 and report["n_test"] == 1500
    assert all(max(train) < min(selection) and max(selection) < 8500 for train, selection in calls)
    assert report["publication_gate"]["passed"] is True
    assert report["publication_gate"]["incumbent_test_overlap"] == "fresh_after_previous_snapshot"
    exporting = script("export_model")
    exported = output / "winprob.npz"
    exporting.export(output / "winprob.pt", exported, incumbent_path=incumbent_path,
                     require_incumbent=True, metrics_path=report_path)
    from bsdraft.collect.publish import validate_model_bundle
    assert validate_model_bundle(exported, report_path)["model"]["parameters"] > 0
    saved_report = report_path.read_text()
    modified = json.loads(saved_report)
    modified["publication_gate"]["delta"] = -0.02
    report_path.write_text(json.dumps(modified))
    with pytest.raises(SystemExit, match="metadata differs"):
        exporting.export(output / "winprob.pt", exported, incumbent_path=incumbent_path,
                         require_incumbent=True, metrics_path=report_path)
    report_path.write_text(saved_report)
    with np.load(exported) as z:
        assert json.loads(z["_evaluation"].item())["training_run_id"] == report["training_run_id"]
    # A stronger incumbent makes the same pipeline refuse BEFORE writing any replacement.
    monkeypatch.setattr(training.EV, "predict_incumbent", lambda model, ds, rows, **kwargs: .1 + .8 * ds.y[rows])
    output.joinpath("winprob.pt").unlink()
    with pytest.raises(SystemExit, match="regression gate"):
        training.main()
    assert not output.joinpath("winprob.pt").exists()


def test_incumbent_comparison_uses_ids_not_current_encoder_positions():
    calls = []
    class Incumbent:
        def prob_batch(self, a, b, map_id, mode):
            calls.append((a, b, map_id, mode))
            return [.6] * len(a)
    ds = SimpleNamespace(team_a=np.array([[0, 1, 2]]), team_b=np.array([[3, 4, 5]]),
                         map_idx=np.array([1]), mode_idx=np.array([1]))
    got = EV.predict_incumbent(Incumbent(), ds, [0], brawler_ids=dict(enumerate([99, 11, 55, 44, 77, 22])),
                              map_ids={1: 1005}, modes={1: "Gem Grab"})
    assert calls == [([[99, 11, 55]], [[44, 77, 22]], 1005, "Gem Grab")]
    assert got.tolist() == [.6]


def test_workflow_requires_real_incumbent_and_executes_guard_tests():
    workflow = (REPO_ROOT / ".github/workflows/retrain-model.yml").read_text()
    assert workflow.index("download_incumbent.py") < workflow.index("python backend/scripts/train.py")
    command = workflow.split("python backend/scripts/train.py", 1)[1].split("python -m bsdraft.collect.publish", 1)[0]
    assert command.count("--require-incumbent") == 2
    assert "--incumbent-npz" in command and "--metrics docs/metrics.json" in command
    assert workflow.index("reserve_evaluation.py") < workflow.index("python backend/scripts/train.py")
    assert "--evaluation-reservation" in command
    assert "test_training_safeguards.py" in workflow and "test_partial_draft.py" in workflow
    assert "allow_legacy_incumbent:" in workflow and "default: false" in workflow
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
    assert "pull_request:" in ci and "pytest backend/tests" in ci
    assert "npm test" in ci and "tsc --noEmit" in ci and "npm run build" in ci


def test_failed_attempt_consumes_test_snapshot_and_reservation_must_read_back(tmp_path, monkeypatch):
    from bsdraft.models import evaluation_ledger as ledger
    state = {"id": 99, "body": json.dumps({"schema_version": 1, "consumed_through_ts": 500})}
    calls = []
    def gh(*args):
        calls.append(args)
        if "PATCH" in args:
            state.update(json.loads(Path(args[args.index("--input") + 1]).read_text()))
        return subprocess.CompletedProcess(args, 0, json.dumps(state), "")
    from pathlib import Path
    monkeypatch.setattr(ledger, "_gh", gh)
    result = ledger.reserve_snapshot("owner/repository", through_ts=1000, dataset_sha256="d" * 64,
                                     incumbent_through_ts=400, source_commit="a" * 40)
    assert result["test_after_ts"] == 500
    assert json.loads(state["body"])["consumed_through_ts"] == 1000
    # Simulate the actual model gate failing later. The next attempt must still refuse these rows.
    with pytest.raises(ValueError, match="already consumed"):
        ledger.reserve_snapshot("owner/repository", through_ts=1000, dataset_sha256="d" * 64,
                                 incumbent_through_ts=400, source_commit="a" * 40)
    assert not any("--clobber" in call for call in calls)
    monkeypatch.setattr(ledger, "_gh", lambda *args: subprocess.CompletedProcess(args, 1, "", "HTTP 503"))
    with pytest.raises(RuntimeError, match="unavailable"):
        ledger.reserve_snapshot("owner/repository", through_ts=2000, dataset_sha256="d" * 64,
                                 incumbent_through_ts=400, source_commit="a" * 40)


def test_ledger_refuses_ambiguous_reservation_without_readback(tmp_path, monkeypatch):
    from bsdraft.models import evaluation_ledger as ledger
    old = {"id": 99, "body": json.dumps({"schema_version": 1, "consumed_through_ts": 500})}
    monkeypatch.setattr(ledger, "_gh", lambda *args: subprocess.CompletedProcess(args, 0, json.dumps(old), ""))
    with pytest.raises(RuntimeError, match="readback failed"):
        ledger.reserve_snapshot("owner/repository", through_ts=2000, dataset_sha256="d" * 64,
                                 incumbent_through_ts=400, source_commit="a" * 40)


def test_undersized_new_window_does_not_advance_attempt_watermark(monkeypatch):
    from bsdraft.models import evaluation_ledger as ledger
    old = {"id": 99, "body": json.dumps({"schema_version": 1, "consumed_through_ts": 500})}
    calls = []
    def gh(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps(old), "")
    monkeypatch.setattr(ledger, "_gh", gh)
    with pytest.raises(ValueError, match="not enough"):
        ledger.reserve_snapshot("owner/repository", through_ts=600, dataset_sha256="d" * 64,
                                 incumbent_through_ts=400, source_commit="a" * 40,
                                 validate_floor=lambda floor: EV.temporal_split(np.arange(1, 601), previous_data_through_ts=floor))
    assert len(calls) == 1  # only GET, never PATCH/create


def test_candidate_test_predictions_match_exported_unlearned_map_fallback(tmp_path):
    from bsdraft.models.serve import WinProbModel
    torch.manual_seed(17)
    cfg = ModelConfig(E.num_brawlers(), E.num_maps(), E.num_modes(), mask_row=E.num_brawlers(), class_synergy=True)
    model = WinProbNet(cfg).eval()
    arrays = {key: value.numpy() for key, value in model.state_dict().items()}
    exporter = script("export_model")
    vocab = exporter._vocab()
    map_counts = [200] + [0] * (len(vocab["_vocab_map_ids"]) - 1)
    vocab.update(exporter._learned_maps_only(vocab, map_counts))
    path = tmp_path / "candidate.npz"
    np.savez(path, _config=np.array(json.dumps(cfg.to_dict())), **arrays, **vocab)
    served = WinProbModel(path)
    counts = np.zeros(E.num_maps(), dtype=int)
    counts[1] = 200
    before = model.map_emb.weight.detach().clone()
    with torch.no_grad(), EV.serving_map_context(model, counts):
        candidate = torch.sigmoid(model(torch.tensor([[0, 1, 2]]), torch.tensor([[3, 4, 5]]),
                                       torch.tensor([2]), torch.tensor([1]))).item()
    brawlers = [bid for bid, row in sorted(E.brawler_encoder().items(), key=lambda pair: pair[1])]
    maps = {row: mid for mid, row in E.map_encoder().items()}
    modes = {row: mode for mode, row in E.mode_encoder().items()}
    assert abs(candidate - served.prob(brawlers[:3], brawlers[3:6], maps[2], modes[1])) < 1e-6
    assert torch.equal(model.map_emb.weight, before)


def test_incumbent_rejects_declared_shape_mismatch_even_if_probe_rows_exist(tmp_path):
    path = tmp_path / "incumbent.npz"
    synthetic_incumbent(path, through=1000)
    with np.load(path) as archive:
        arrays = dict(archive)
    arrays["map_emb.weight"] = arrays["map_emb.weight"][:-1]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="tensor shapes"):
        EV.load_incumbent(path, required=True)
