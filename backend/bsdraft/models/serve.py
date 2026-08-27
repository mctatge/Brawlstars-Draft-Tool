"""Serve the win-probability model in pure NumPy — no torch at runtime.

Loads the weights exported by ``scripts/export_model.py`` (``winprob.npz``) and replicates
``WinProbNet.forward`` exactly:

    logit = [ S(A, ctx) - S(B, ctx) ] + [ PA·QB - PB·QA ] + [ T(A) - T(B) ]

where ctx = concat(map_emb, mode_emb); S is the strength MLP over the mean brawler
embedding + ctx (Linear -> ReLU -> [Dropout, a no-op at eval] -> Linear); P/Q are the
low-rank counter embeddings; and T(team) = sum_{i<j} M[cls(i), cls(j)] is the optional
class-level within-team synergy term (present only when the export carries ``class_syn`` —
older artifacts omit it and this file skips the term). Training still uses PyTorch
(``scripts/train.py``); only inference is reimplemented here so the deployed API needs
neither torch nor the training deps.

Partial drafts: artifacts trained with masked drafts carry a ``mask_row`` in their config —
a trained "unknown slot" embedding row. When present (``supports_partial``), teams with
fewer than 3 picks are padded with that row and scored directly; the prediction
marginalizes over how real drafts continued from such boards. Legacy artifacts reject
partial teams (callers gate on ``supports_partial`` and fall back to team completion).

Degrades gracefully in two ways:

  * if no export exists yet, ``available`` is False and ``prob`` returns 0.5, so the engine can
    still run on empirical stats alone;
  * if the reference catalog is **newer than the model** — a brawler or map the export has no
    embedding row for — that id falls back to the mean embedding instead of raising. This is not
    hypothetical: the catalog ships in git (Render redeploy) while the model ships as a GitHub
    Release asset (hot-swap), so a new brawler is live in the catalog from the moment it is
    committed until the next retrain. Without the guard, indexing past the export's vocabulary
    raises ``IndexError`` and every draft touching that brawler/map 500s.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from bsdraft.constants import BRAWLER_CLASSES, MODE_CAMEL_TO_DISPLAY, PROCESSED_DIR, TEAM_SIZE
from bsdraft.data import encoders as E

DEFAULT_PATH = PROCESSED_DIR / "winprob.npz"
logger = logging.getLogger(__name__)

_N_CLASSES = len(BRAWLER_CLASSES)          # class-level synergy: 7 archetypes; index 7 = unknown/mask

# Embedding matrices, grouped by the vocabulary that indexes them.
_BRAWLER_MATRICES = ("brawler.weight", "counter_p.weight", "counter_q.weight")
_MAP_MATRICES = ("map_emb.weight",)
_MODE_MATRICES = ("mode_emb.weight",)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class WinProbModel:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.cfg: Optional[dict] = None
        self._w: Optional[Dict[str, np.ndarray]] = None
        # Vocabulary sizes this export was trained with (before the fallback row is appended).
        self._vocab: Dict[str, int] = {}
        self._warned = False
        # id -> trained row, from the vocabulary pinned into the export. Empty for older
        # artifacts, which fall back to the live positional encoders.
        self._brawler_rows: Dict[int, int] = {}
        self._map_rows: Dict[int, int] = {}
        self._mode_rows: Dict[str, int] = {}
        # class index per brawler row (len num_brawlers+2: real rows, mask, OOV/fallback), for the
        # optional class-level synergy term; None when the export has no class_syn matrix.
        self._class_rows: Optional[np.ndarray] = None
        if self.path.exists():
            data = np.load(self.path, allow_pickle=False)
            self.cfg = json.loads(data["_config"].item())
            self._w = {k: data[k].astype(np.float32) for k in data.files
                       if k != "_config" and not k.startswith("_vocab_")}
            self._load_vocab(data)
            self._add_fallback_rows()
            self._class_rows = self._build_class_rows()

    def _load_vocab(self, data) -> None:
        """Read the pinned id->row tables written by ``scripts/export_model.py``. Without them a
        catalog refresh can silently re-map an existing map onto a neighbour's trained row (map
        indices are positional over a (mode, name)-sorted list); with them the mapping is exact
        and refreshes are harmless."""
        files = set(data.files)
        if {"_vocab_brawler_ids"} <= files:
            ids = data["_vocab_brawler_ids"].tolist()
            self._brawler_rows = {int(b): i for i, b in enumerate(ids)}
        if {"_vocab_map_ids", "_vocab_map_rows"} <= files:
            self._map_rows = {int(m): int(r) for m, r in
                              zip(data["_vocab_map_ids"].tolist(), data["_vocab_map_rows"].tolist())}
        if {"_vocab_modes", "_vocab_mode_rows"} <= files:
            self._mode_rows = {str(m): int(r) for m, r in
                               zip(data["_vocab_modes"].tolist(), data["_vocab_mode_rows"].tolist())}

    def _add_fallback_rows(self) -> None:
        """Append a mean-embedding row to every lookup matrix and remember the original vocab
        size. Ids beyond that size (catalog newer than the model) are steered to this row: an
        "average" brawler/map is a neutral prior, whereas clamping to row 0 would silently
        impersonate a specific brawler (index 0 is Shelly)."""
        w = self._w
        for key in _BRAWLER_MATRICES + _MAP_MATRICES + _MODE_MATRICES:
            m = w.get(key)
            if m is None:
                continue
            self._vocab[key] = m.shape[0]
            base = m
            if key in _BRAWLER_MATRICES and self.supports_partial:
                base = m[: int(self.cfg["mask_row"])]   # the mask row is not a real brawler
            w[key] = np.vstack([m, base.mean(axis=0, keepdims=True)])

    def _build_class_rows(self) -> Optional[np.ndarray]:
        """Class index for every possible brawler row, for the class-level synergy term. Built from
        the reference catalog + the fixed ``BRAWLER_CLASSES`` order — the exact mapping training
        used — keyed by the export's pinned brawler vocabulary. The mask row and the appended
        OOV/fallback row map to the unknown class (``_N_CLASSES``) and are excluded from synergy.
        Returns None when the export carries no ``class_syn`` matrix or predates pinned vocab."""
        if self._w is None or "class_syn" not in self._w or not self._brawler_rows:
            return None
        from bsdraft.data import reference as R
        cls_idx = {c: i for i, c in enumerate(BRAWLER_CLASSES)}
        cls_by_id = {b.id: b.cls for b in R.load_brawlers()}
        n1 = self._vocab.get("brawler.weight")           # num_brawlers + 1 (real rows + mask)
        if n1 is None:
            return None
        rows = np.full(n1 + 1, _N_CLASSES, dtype=np.int64)   # +1 covers the OOV/fallback row
        for bid, row in self._brawler_rows.items():
            if 0 <= row < n1 - 1:                        # real brawler rows only
                rows[row] = cls_idx.get(cls_by_id.get(bid), _N_CLASSES)
        return rows

    def _brawler_row(self, brawler_id: int) -> int:
        """Trained row for a brawler id. An id the export predates returns a sentinel past the
        vocabulary, which :meth:`_safe` turns into the mean-embedding fallback."""
        if self._brawler_rows:
            return self._brawler_rows.get(int(brawler_id), self._vocab.get("brawler.weight", 0))
        return E.encode_brawler(brawler_id)

    def _map_row(self, map_id) -> int:
        if self._map_rows:
            return self._map_rows.get(int(map_id), self._vocab.get("map_emb.weight", 0))
        return E.encode_map(map_id)

    def _mode_row(self, mode: str) -> int:
        if self._mode_rows:
            display = MODE_CAMEL_TO_DISPLAY.get(mode, mode)
            return self._mode_rows.get(display, self._vocab.get("mode_emb.weight", 0))
        return E.encode_mode(mode)

    def _safe(self, idx, key: str):
        """Map any out-of-vocabulary index to the appended mean row. Accepts a scalar or an
        ndarray; returns the same shape."""
        n = self._vocab.get(key)
        if n is None:
            return idx
        arr = np.asarray(idx)
        if bool((arr >= n).any()):
            if not self._warned:
                self._warned = True
                logger.warning(
                    "reference catalog is newer than %s (%s has %d rows): unknown ids are using "
                    "the mean embedding — retrain + re-export to give them real rows",
                    self.path.name, key, n)
            arr = np.where(arr >= n, n, arr)
        return arr if np.ndim(idx) else int(arr)

    @property
    def available(self) -> bool:
        return self._w is not None

    @property
    def supports_partial(self) -> bool:
        """True when the artifact carries a trained "unknown slot" row, so ``prob`` /
        ``prob_batch`` accept teams with fewer than 3 picks. Legacy exports don't."""
        return bool(self.cfg) and self.cfg.get("mask_row") is not None

    def _team_rows(self, team: Sequence[int]) -> List[int]:
        """Embedding rows for one team, padding missing slots with the mask row."""
        if len(team) > TEAM_SIZE:
            raise ValueError(f"team has {len(team)} brawlers; draft teams hold at most {TEAM_SIZE}")
        rows = [self._brawler_row(x) for x in team]
        if len(rows) < TEAM_SIZE:
            if not self.supports_partial:
                raise ValueError(
                    f"{self.path.name} predates partial-draft support: pass full "
                    f"{TEAM_SIZE}-brawler teams, or check supports_partial before calling")
            rows += [int(self.cfg["mask_row"])] * (TEAM_SIZE - len(rows))
        return rows

    def prob(self, team_a_ids: Sequence[int], team_b_ids: Sequence[int], map_id: int, mode: str) -> float:
        """P(team_a beats team_b). Teams are 0-3 brawler ids; short teams need a
        partial-draft artifact (``supports_partial``) and score the board as it stands."""
        if not self.available:
            return 0.5
        return self.prob_batch([list(team_a_ids)], [list(team_b_ids)], map_id, mode)[0]

    def prob_batch(
        self,
        teams_a: List[List[int]],
        teams_b: List[List[int]],
        map_id: int,
        mode: str,
    ) -> List[float]:
        return self._forward(teams_a, teams_b, map_id, mode, rowwise=False)

    def prob_marginals(
        self,
        teams_a: List[List[int]],
        teams_b: List[List[int]],
        map_id: int,
        mode: str,
    ) -> List[float]:
        """Batched ``prob`` for many boards at once, **bit-for-bit identical** to calling
        :meth:`prob` per pair. Used by the pick recommender, which scores ~100 candidates that
        share one enemy team and differ only by the added ally.

        The gather / context / counter stages are already per-row independent, so batching them
        is exact. The only stage BLAS accumulates differently for a tall matrix than for a single
        row is the strength MLP's matmul — so it is computed one board at a time (``rowwise``),
        each an M=1 GEMM identical to the per-candidate call. This still amortizes all the
        per-candidate Python/array overhead (row lookups, ``_safe`` reductions, ``tile``,
        ``concatenate``, ``.tolist``) that dominated the loop, without changing any result."""
        return self._forward(teams_a, teams_b, map_id, mode, rowwise=True)

    def _forward(
        self,
        teams_a: List[List[int]],
        teams_b: List[List[int]],
        map_id: int,
        mode: str,
        rowwise: bool,
    ) -> List[float]:
        if not self.available:
            return [0.5] * len(teams_a)
        w = self._w
        n = len(teams_a)
        # Rows come from the export's pinned vocabulary when it has one (exact, immune to catalog
        # reordering); otherwise from the live positional encoders. _safe() then steers any id the
        # export has no row for to the appended mean embedding instead of raising IndexError.
        # (The mask row pads short teams first and is a trained row, in range by construction.)
        a = self._safe(np.array([self._team_rows(t) for t in teams_a]),
                       "brawler.weight")                                   # (N, 3)
        b = self._safe(np.array([self._team_rows(t) for t in teams_b]),
                       "brawler.weight")                                   # (N, 3)

        # ctx = concat(map_emb, mode_emb), broadcast across the batch
        ctx = np.concatenate(
            [
                np.tile(w["map_emb.weight"][self._safe(self._map_row(map_id), "map_emb.weight")], (n, 1)),
                np.tile(w["mode_emb.weight"][self._safe(self._mode_row(mode), "mode_emb.weight")], (n, 1)),
            ],
            axis=1,
        )

        def strength(team: np.ndarray) -> np.ndarray:
            team_vec = w["brawler.weight"][team].mean(axis=1)        # (N, d_brawler), order-invariant
            h = np.concatenate([team_vec, ctx], axis=1)
            if rowwise:
                # One M=1 GEMM per board: reproduces prob()'s exact accumulation order (BLAS
                # blocks a tall matmul differently, which would perturb the logit by ~1 float32
                # ULP). Everything else here is already batched.
                w0, b0 = w["strength.0.weight"].T, w["strength.0.bias"]
                w3, b3 = w["strength.3.weight"].T, w["strength.3.bias"]
                out = np.empty(h.shape[0], dtype=h.dtype)
                for i in range(h.shape[0]):
                    hi = h[i:i + 1] @ w0 + b0                        # Linear (M=1)
                    hi = np.maximum(hi, 0.0)                         # ReLU (Dropout no-op at eval)
                    out[i] = (hi @ w3 + b3)[0, 0]
                return out
            h = h @ w["strength.0.weight"].T + w["strength.0.bias"]  # Linear
            h = np.maximum(h, 0.0)                                   # ReLU (Dropout is a no-op at eval)
            out = h @ w["strength.3.weight"].T + w["strength.3.bias"]
            return out[:, 0]

        s = strength(a) - strength(b)
        pa = w["counter_p.weight"][a].sum(axis=1)  # (N, r)
        qa = w["counter_q.weight"][a].sum(axis=1)
        pb = w["counter_p.weight"][b].sum(axis=1)
        qb = w["counter_q.weight"][b].sum(axis=1)
        counter = (pa * qb).sum(axis=1) - (pb * qa).sum(axis=1)

        logit = s + counter
        # Class-level synergy: sum of the symmetric class-pair weight over the 3 slot pairs,
        # excluding mask / unknown-class slots. Present only when the export carries class_syn.
        csyn_w = w.get("class_syn")
        if csyn_w is not None and self._class_rows is not None:
            M = csyn_w + csyn_w.T                                       # symmetrize (N_CLASSES^2)
            cr = self._class_rows

            def team_csyn(team_rows: np.ndarray) -> np.ndarray:
                c = cr[team_rows]                                       # (N, 3) class idx
                real = c < _N_CLASSES
                cc = np.minimum(c, _N_CLASSES - 1)                     # safe gather
                tot = np.zeros(team_rows.shape[0], dtype=np.float64)
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    both = (real[:, i] & real[:, j]).astype(np.float64)
                    tot = tot + both * M[cc[:, i], cc[:, j]]
                return tot
            logit = logit + (team_csyn(a) - team_csyn(b))
        return _sigmoid(logit).tolist()
