"""Unit tests for the serving model's out-of-vocabulary guard (bsdraft.models.serve).

The reference catalog ships in git (Render redeploy) while the model ships as a GitHub Release
asset (hot-swap), so between a catalog commit and the next retrain the catalog legitimately
knows brawlers/maps the export has no embedding row for. Before the guard, that raised
IndexError and every draft touching the new brawler 500'd.

Uses a tiny synthetic export (no torch, no network, independent of the committed winprob.npz
whose shapes change on every retrain).

    PYTHONPATH=backend python -m pytest backend/tests/test_serve_vocab.py   # or run directly
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from bsdraft.models import serve as S

N_BRAWLER, N_MAP, N_MODE, D, DCTX, R = 4, 3, 2, 4, 3, 2


def _synthetic(tmp: Path) -> Path:
    """A minimal winprob.npz with the exact key set serve.py reads."""
    rng = np.random.default_rng(0)
    hidden = 5
    arrays = {
        "_config": np.array(json.dumps({"synthetic": True})),
        "brawler.weight": rng.normal(size=(N_BRAWLER, D)).astype(np.float32),
        "map_emb.weight": rng.normal(size=(N_MAP, 2)).astype(np.float32),
        "mode_emb.weight": rng.normal(size=(N_MODE, 1)).astype(np.float32),
        "strength.0.weight": rng.normal(size=(hidden, D + 3)).astype(np.float32),
        "strength.0.bias": rng.normal(size=(hidden,)).astype(np.float32),
        "strength.3.weight": rng.normal(size=(1, hidden)).astype(np.float32),
        "strength.3.bias": rng.normal(size=(1,)).astype(np.float32),
        "counter_p.weight": rng.normal(size=(N_BRAWLER, R)).astype(np.float32),
        "counter_q.weight": rng.normal(size=(N_BRAWLER, R)).astype(np.float32),
    }
    path = tmp / "winprob.npz"
    np.savez(path, **arrays)
    return path


class _Encoders:
    """Stand-in for bsdraft.data.encoders that returns exactly the indices a test asks for."""
    def __init__(self):
        self.brawler = {}
        self.map_idx = 0
        self.mode_idx = 0

    def encode_brawler(self, bid):
        return self.brawler.get(bid, 0)

    def encode_map(self, mid):
        return self.map_idx

    def encode_mode(self, mode):
        return self.mode_idx


def _model_with(enc: _Encoders, path: Path):
    """Build a model with the encoder layer swapped out; caller restores."""
    S.E, prev = enc, S.E
    try:
        return S.WinProbModel(path), prev
    except Exception:
        S.E = prev
        raise


def test_fallback_row_appended_and_existing_rows_untouched():
    with tempfile.TemporaryDirectory() as td:
        path = _synthetic(Path(td))
        raw = np.load(path, allow_pickle=False)
        m = S.WinProbModel(path)
        for key, n in (("brawler.weight", N_BRAWLER), ("map_emb.weight", N_MAP),
                       ("mode_emb.weight", N_MODE), ("counter_p.weight", N_BRAWLER)):
            assert m._vocab[key] == n, key
            assert m._w[key].shape[0] == n + 1, key           # exactly one row appended
            # Existing rows are byte-identical -> known ids cannot regress.
            assert np.allclose(m._w[key][:n], raw[key]), key
            # The appended row is the mean of the originals (a neutral "average" embedding).
            assert np.allclose(m._w[key][n], raw[key].mean(axis=0), atol=1e-6), key


def test_safe_maps_out_of_vocab_to_fallback():
    with tempfile.TemporaryDirectory() as td:
        m = S.WinProbModel(_synthetic(Path(td)))
        assert m._safe(0, "brawler.weight") == 0                   # in range, untouched
        assert m._safe(N_BRAWLER - 1, "brawler.weight") == N_BRAWLER - 1
        assert m._safe(N_BRAWLER, "brawler.weight") == N_BRAWLER    # out of range -> fallback
        assert m._safe(N_BRAWLER + 99, "brawler.weight") == N_BRAWLER
        got = m._safe(np.array([[0, N_BRAWLER + 5, 1]]), "brawler.weight")
        assert got.tolist() == [[0, N_BRAWLER, 1]]                  # array form, shape preserved


def test_out_of_vocab_brawler_does_not_raise():
    with tempfile.TemporaryDirectory() as td:
        enc = _Encoders()
        # ids 1..3 are known; 99 is a brand-new brawler the export predates.
        enc.brawler = {1: 0, 2: 1, 3: 2, 99: N_BRAWLER, 98: N_BRAWLER + 1}
        m, prev = _model_with(enc, _synthetic(Path(td)))
        try:
            p = m.prob([99, 1, 2], [1, 2, 3], map_id=0, mode="gemGrab")
            assert 0.0 <= p <= 1.0
            # Two different unknown brawlers both use the mean row -> identical prediction.
            assert abs(p - m.prob([98, 1, 2], [1, 2, 3], map_id=0, mode="gemGrab")) < 1e-9
            # Batch path is guarded too.
            probs = m.prob_batch([[99, 1, 2], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]], 0, "gemGrab")
            assert len(probs) == 2 and all(0.0 <= x <= 1.0 for x in probs)
        finally:
            S.E = prev


def test_out_of_vocab_map_and_mode_do_not_raise():
    with tempfile.TemporaryDirectory() as td:
        enc = _Encoders()
        enc.brawler = {1: 0, 2: 1, 3: 2}
        enc.map_idx = N_MAP + 7      # a map added after the export
        enc.mode_idx = N_MODE + 1    # a mode added after the export
        m, prev = _model_with(enc, _synthetic(Path(td)))
        try:
            p = m.prob([1, 2, 3], [1, 2, 3], map_id=123, mode="newMode")
            assert 0.0 <= p <= 1.0
            # Mirror match on the same map/mode is a coin flip by antisymmetry.
            assert abs(p - 0.5) < 1e-6
        finally:
            S.E = prev


def test_pinned_vocab_maps_ids_to_trained_rows():
    """With a pinned vocabulary the row comes from the id, not from the live catalog ordering —
    so refreshing data/reference/ can no longer shift a map onto a neighbour's trained row."""
    with tempfile.TemporaryDirectory() as td:
        path = _synthetic(Path(td))
        raw = dict(np.load(path, allow_pickle=False))
        raw["_vocab_brawler_ids"] = np.array([700, 701, 702, 703], dtype=np.int64)
        raw["_vocab_map_ids"] = np.array([900, 901], dtype=np.int64)
        raw["_vocab_map_rows"] = np.array([2, 1], dtype=np.int64)   # deliberately NOT positional
        raw["_vocab_modes"] = np.array(["Gem Grab"])
        raw["_vocab_mode_rows"] = np.array([1], dtype=np.int64)
        np.savez(path, **raw)
        m = S.WinProbModel(path)
        assert m._brawler_row(702) == 2
        assert m._map_row(900) == 2 and m._map_row(901) == 1
        assert m._mode_row("gemGrab") == 1          # camelCase battle-log name is translated
        # Ids the export predates fall past the vocabulary -> _safe sends them to the mean row.
        assert m._safe(m._brawler_row(999), "brawler.weight") == N_BRAWLER
        assert m._safe(m._map_row(999), "map_emb.weight") == N_MAP
        # The weight matrices exclude the _vocab_* arrays.
        assert not any(k.startswith("_vocab_") for k in m._w)


def test_map_fallback_row_is_the_mean_of_pinned_rows_only():
    """The map vocab is the whole ranked-mode catalog; exports pin only learned rows. Row 0 (the
    unknown bucket) and never-trained catalog maps keep their init, so averaging them in would
    pull the "average map" toward zero. Unpinned ids — including in-matrix ones — use the mean
    of the pinned rows."""
    with tempfile.TemporaryDirectory() as td:
        path = _synthetic(Path(td))
        raw = dict(np.load(path, allow_pickle=False))
        raw["_vocab_map_ids"] = np.array([900], dtype=np.int64)
        raw["_vocab_map_rows"] = np.array([2], dtype=np.int64)       # rows 0 and 1 unlearned
        np.savez(path, **raw)
        m = S.WinProbModel(path)
        assert np.allclose(m._w["map_emb.weight"][N_MAP], raw["map_emb.weight"][2], atol=1e-6)
        assert m._safe(m._map_row(901), "map_emb.weight") == N_MAP  # in-catalog, unpinned


def test_missing_pinned_vocab_falls_back_to_encoders():
    with tempfile.TemporaryDirectory() as td:
        enc = _Encoders()
        enc.brawler = {5: 1}
        m, prev = _model_with(enc, _synthetic(Path(td)))
        try:
            assert not m._brawler_rows          # no pinned vocab in this artifact
            assert m._brawler_row(5) == 1       # …so the live encoder is used, as before
        finally:
            S.E = prev


def test_missing_export_still_degrades_to_half():
    m = S.WinProbModel(Path("/nonexistent/winprob.npz"))
    assert not m.available
    assert m.prob([1, 2, 3], [4, 5, 6], 0, "gemGrab") == 0.5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
