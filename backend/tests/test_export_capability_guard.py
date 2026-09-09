"""The export-time capability-regression guard (scripts/export_model.py).

Added 2026-09-03 after the live win-prob model was found to have silently lost its
class-level within-team synergy term (``class_synergy`` / ``class_syn``, commit 5ff34c9).
Nothing errored: ``train.py`` declared ``--class-synergy`` as ``store_true``, ``collect.py``'s
unattended ``--retrain-on-shift`` argv never passed it, and ``collect.py`` publishes on
training success unconditionally — so every automatic retrain re-exported a weaker model and
shipped it to the live API's hot-swap. These tests pin the structural fix: an export that
drops a capability the artifact it replaces already has must refuse unless asked.

    PYTHONPATH=backend python -m pytest backend/tests/test_export_capability_guard.py
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from bsdraft.constants import REPO_ROOT


@pytest.fixture(scope="module")
def em():
    """scripts/export_model.py isn't a package module — load it by path. It imports torch at
    module scope, so skip cleanly on the serve deploy where torch isn't installed."""
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "export_model_under_test", REPO_ROOT / "backend" / "scripts" / "export_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- capability_regressions: the pure decision --------------------------------------------

def test_flag_flipping_true_to_false_is_a_regression(em):
    lost = em.capability_regressions({"class_synergy": True}, [], {"class_synergy": False}, [])
    assert len(lost) == 1 and "class_synergy" in lost[0]


def test_flag_vanishing_entirely_is_a_regression(em):
    """A checkpoint from before the term existed carries no key at all — still a downgrade."""
    lost = em.capability_regressions({"class_synergy": True}, [], {}, [])
    assert len(lost) == 1 and "absent" in lost[0]


def test_dropped_weight_array_is_a_regression(em):
    lost = em.capability_regressions({}, ["brawler.weight", "class_syn"], {}, ["brawler.weight"])
    assert len(lost) == 1 and "class_syn" in lost[0]


def test_the_actual_2026_09_03_regression_is_caught(em):
    """The exact shape of the incident: the flag flips and both arrays disappear."""
    prev_cfg = {"class_synergy": True, "num_brawlers": 107, "mask_row": 107}
    prev_keys = ["brawler.weight", "class_syn", "brawler_class", "_config", "_vocab_map_ids"]
    cfg = {"class_synergy": False, "num_brawlers": 107, "mask_row": 107}
    new_keys = ["brawler.weight", "_config", "_vocab_map_ids"]
    lost = em.capability_regressions(prev_cfg, prev_keys, cfg, new_keys)
    assert len(lost) == 3
    assert any("class_synergy" in m for m in lost)
    assert any("'class_syn'" in m for m in lost)
    assert any("'brawler_class'" in m for m in lost)


def test_export_bookkeeping_keys_are_not_capabilities(em):
    """`_config` and the pinned vocab are rewritten wholesale every run; a vocabulary that
    legitimately shrank (a brawler leaving the catalog) must not read as a lost capability."""
    assert em.capability_regressions(
        {}, ["_config", "_vocab_map_ids", "_vocab_modes"], {}, ["_config"]) == []


def test_int_config_values_do_not_false_positive(em):
    """`1 == True` in Python, so a truthiness test would report a dimension change as a lost
    capability. Only an actual bool counts."""
    assert em.capability_regressions({"mask_row": 1}, [], {"mask_row": 0}, []) == []
    assert em.capability_regressions({"d_hidden": 64}, [], {"d_hidden": 32}, []) == []


def test_unchanged_and_upgraded_exports_pass(em):
    cfg = {"class_synergy": True}
    assert em.capability_regressions(cfg, ["class_syn"], cfg, ["class_syn"]) == []
    # gaining a capability is not a regression
    assert em.capability_regressions(
        {"class_synergy": False}, [], {"class_synergy": True}, ["class_syn"]) == []


def test_false_to_false_is_not_a_regression(em):
    assert em.capability_regressions({"class_synergy": False}, [], {"class_synergy": False}, []) == []


# --- _previous_export: reading the artifact being replaced --------------------------------

def test_missing_previous_export_is_not_an_obstacle(em, tmp_path):
    assert em._previous_export(tmp_path / "nope.npz") == ({}, set())


def test_corrupt_previous_export_does_not_block(em, tmp_path):
    """A predecessor we cannot read must never stop a good export — the guard is here to catch
    silent downgrades, not to gate on the health of the file it replaces."""
    bad = tmp_path / "winprob.npz"
    bad.write_bytes(b"not an npz at all")
    assert em._previous_export(bad) == ({}, set())


def test_reads_config_and_keys_from_a_real_npz(em, tmp_path):
    import json
    p = tmp_path / "winprob.npz"
    np.savez(p, _config=np.array(json.dumps({"class_synergy": True})),
             **{"brawler.weight": np.zeros((2, 2)), "class_syn": np.zeros((7, 7))})
    cfg, keys = em._previous_export(p)
    assert cfg == {"class_synergy": True}
    assert {"brawler.weight", "class_syn", "_config"} <= keys


# --- losing a non-bool capability: mask_row / partial-draft support -----------------------

def test_losing_mask_row_is_a_regression(em):
    """serve.supports_partial is exactly `cfg["mask_row"] is not None`, so a checkpoint that
    drops it silently removes partial-draft scoring — a worse regression than class synergy."""
    lost = em.capability_regressions({"mask_row": 107}, [], {"mask_row": None}, [])
    assert len(lost) == 1 and "mask_row" in lost[0]
    lost = em.capability_regressions({"mask_row": 107}, [], {}, [])
    assert len(lost) == 1 and "absent" in lost[0]


def test_gaining_mask_row_is_not_a_regression(em):
    assert em.capability_regressions({"mask_row": None}, [], {"mask_row": 107}, []) == []


def test_a_changed_setting_is_not_a_regression(em):
    """Only losing a setting counts. Retuning one (a different dimension, a different row) is
    an ordinary retrain, not a capability loss."""
    assert em.capability_regressions({"mask_row": 107}, [], {"mask_row": 108}, []) == []
    assert em.capability_regressions({"d_hidden": 64}, [], {"d_hidden": 128}, []) == []


def test_zero_and_empty_settings_are_not_treated_as_missing(em):
    """`0` and `""` are falsy but present — only None/absent is a loss."""
    assert em.capability_regressions({"counter_rank": 16}, [], {"counter_rank": 0}, []) == []
