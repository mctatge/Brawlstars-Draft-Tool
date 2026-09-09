"""Export the trained PyTorch checkpoint to a NumPy archive the API can serve without torch.

    PYTHONPATH=backend python backend/scripts/export_model.py

Run this on the training machine (torch is needed here only). The API loads the resulting
``winprob.npz`` and runs inference in pure NumPy (see bsdraft/models/serve.py), so the
deployed backend needs neither torch nor the rest of the training dependencies. The npz is
tiny (~50 KB), so it's committed and ships with the repo to the cloud host.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import torch

from bsdraft.constants import PROCESSED_DIR

DEFAULT_PT = PROCESSED_DIR / "winprob.pt"
DEFAULT_NPZ = PROCESSED_DIR / "winprob.npz"


def _vocab() -> dict:
    """Pin the vocabularies this export was trained against.

    Brawler rows are addressed by *position* in the reference catalog, and ranked maps are
    positional over a (mode, name)-sorted list — so refreshing ``data/reference/`` can shift an
    existing map onto a neighbour's trained row. That lookup stays in range, so no
    out-of-vocabulary guard can detect it; only the ids themselves can. Storing them here lets
    :class:`bsdraft.models.serve.WinProbModel` map id -> row exactly as trained, no matter how
    the catalog has moved since.
    """
    from bsdraft.data import encoders as E
    from bsdraft.data import reference as R
    return {
        "_vocab_brawler_ids": np.array([b.id for b in R.load_brawlers()], dtype=np.int64),
        "_vocab_map_ids": np.array([m.id for m in R.load_ranked_maps()], dtype=np.int64),
        "_vocab_map_rows": np.array([E.encode_map(m.id) for m in R.load_ranked_maps()],
                                    dtype=np.int64),
        "_vocab_modes": np.array(list(E.mode_encoder().keys())),
        "_vocab_mode_rows": np.array(list(E.mode_encoder().values()), dtype=np.int64),
    }


def capability_regressions(prev_cfg: dict, prev_keys: Iterable[str],
                           cfg: dict, new_keys: Iterable[str]) -> List[str]:
    """Capabilities the artifact currently on disk has that this export would silently drop.

    Three kinds:

      * a boolean config flag flipping ``True`` -> ``False`` (or vanishing entirely);
      * a config value that had a setting going ``None``/absent — this is how partial-draft
        support would disappear, since ``serve.supports_partial`` is exactly
        ``cfg["mask_row"] is not None``;
      * a weight/buffer array present in the old export and absent from the new one.

    On 2026-09-03 the live model was found to have lost its class-level within-team synergy
    term (``class_synergy``/``class_syn``, commit 5ff34c9). Nothing was broken: ``train.py``
    defined ``--class-synergy`` as ``store_true`` and ``collect.py``'s unattended
    ``--retrain-on-shift`` argv never passed it, so every automatic retrain produced a model
    without the term and ``collect.py`` published it unconditionally on training success. The
    capability was gone from the deployed artifact for a full retrain cycle with no error
    anywhere. This function is the structural fix: a downgrade now has to be asked for.

    Keys beginning with ``_`` are export bookkeeping (``_config`` and the pinned vocabulary)
    and are rewritten wholesale on every run, so they are never treated as capabilities.

    Deliberately retiring a config field will trip the second rule. That is intended: it costs
    one ``--allow-capability-downgrade`` and is far cheaper than the silence it replaces.
    """
    missing = object()
    lost: List[str] = []
    for k, v in sorted(prev_cfg.items()):
        now = cfg.get(k, missing)
        # `is True` rather than truthiness: config also carries ints (mask_row, d_hidden),
        # and `1 == True` would make a dimension change masquerade as a lost capability.
        if v is True and now is not True:
            lost.append(f"config flag {k!r}: True -> "
                        f"{'<absent>' if now is missing else repr(now)}")
        elif v is not None and v is not True and (now is missing or now is None):
            lost.append(f"config value {k!r}: {v!r} -> "
                        f"{'<absent>' if now is missing else 'None'}")
    dropped = {k for k in prev_keys if not k.startswith("_")} - set(new_keys)
    lost.extend(f"array {k!r}: present in the current export, absent from this one"
                for k in sorted(dropped))
    return lost


def _previous_export(npz_path: Path) -> Tuple[Dict, Set[str]]:
    """``(config, array keys)`` of the artifact already at ``npz_path``.

    Returns empty values when there is no previous export or it cannot be read: a missing or
    corrupt predecessor must never block a good export. The guard exists to catch a silent
    capability downgrade, not to gate on the health of the file it is replacing.
    """
    if not npz_path.exists():
        return {}, set()
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            cfg = json.loads(str(z["_config"])) if "_config" in z.files else {}
            return cfg, set(z.files)
    except Exception as e:  # noqa: BLE001 - see docstring: never block on a bad predecessor
        print(f"note: could not read {npz_path} for the capability check ({e}) - skipping it")
        return {}, set()


def export(pt_path: Path, npz_path: Path, allow_downgrade: bool = False) -> None:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    weights = {k: v.detach().cpu().numpy() for k, v in ckpt["state_dict"].items()}
    vocab = _vocab()
    # The vocab is pinned from the LIVE reference, but the checkpoint was trained against the
    # reference as of training time. If the catalog changed in between, the pinned ids would
    # silently address wrong rows — and with a mask row, a new brawler would pin exactly ONTO
    # the "unknown slot" row (in range, so no serving guard can catch it). Fail loudly instead:
    # retrain and export against the same snapshot.
    cfg = ckpt["config"]
    trained_vocab = ckpt.get("vocab")
    if trained_vocab is not None:
        # Checkpoints carry the exact trained id-by-row lists: compare identity, so even a
        # same-size swap (which count checks can't see) is caught.
        for what, live, trained in (
            ("brawler ids", vocab["_vocab_brawler_ids"].tolist(), trained_vocab["brawler_ids"]),
            ("map ids", vocab["_vocab_map_ids"].tolist(), trained_vocab["map_ids"]),
            ("modes", vocab["_vocab_modes"].tolist(), trained_vocab["modes"]),
        ):
            if list(live) != list(trained):
                raise SystemExit(
                    f"reference catalog changed since training ({what} differ) — rerun "
                    f"scripts/train.py against the current reference, then export")
    else:
        # Older checkpoint without pinned ids: counts are the best available check.
        for what, live, trained in (
            ("brawlers", len(vocab["_vocab_brawler_ids"]), cfg["num_brawlers"]),
            ("maps", len(vocab["_vocab_map_ids"]) + 1, cfg["num_maps"]),   # +1: unknown bucket
            ("modes", len(vocab["_vocab_modes"]) + 1, cfg["num_modes"]),
        ):
            if live != trained:
                raise SystemExit(
                    f"reference catalog changed since training ({what}: live {live} != trained "
                    f"{trained}) — rerun scripts/train.py against the current reference, then export")
    # Refuse to replace a more capable artifact with a less capable one (see
    # capability_regressions). This is the last check before the write, so a downgrade is
    # caught whether it came from a missing training flag, a rolled-back config, or a
    # checkpoint from an older code path.
    prev_cfg, prev_keys = _previous_export(npz_path)
    lost = capability_regressions(prev_cfg, prev_keys, cfg,
                                  set(weights) | set(vocab) | {"_config"})
    if lost:
        detail = "\n".join(f"  - {m}" for m in lost)
        if not allow_downgrade:
            raise SystemExit(
                f"refusing to export: this checkpoint drops capabilities that the model "
                f"already at {npz_path} has:\n{detail}\n\n"
                f"Retrain with the flag that produces them (the unattended path in "
                f"backend/scripts/collect.py pins --class-synergy), or pass "
                f"--allow-capability-downgrade if the removal is deliberate.")
        print(f"WARNING: exporting a capability downgrade (--allow-capability-downgrade):\n{detail}")

    np.savez(npz_path, _config=np.array(json.dumps(cfg)), **weights, **vocab)
    size_kb = npz_path.stat().st_size / 1024
    print(f"exported {pt_path}  ->  {npz_path}  ({size_kb:.1f} KB, {len(weights)} tensors "
          f"+ pinned vocabulary)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export winprob.pt -> winprob.npz for NumPy serving.")
    ap.add_argument("--pt", type=Path, default=DEFAULT_PT, help="input torch checkpoint")
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="output NumPy archive")
    ap.add_argument("--allow-capability-downgrade", action="store_true",
                    help="permit an export that drops a capability the artifact being replaced "
                         "has (a True config flag going False, or a weight array disappearing). "
                         "Off by default so an unattended retrain cannot silently ship a weaker "
                         "model; turn it on for a deliberate rollback.")
    args = ap.parse_args()
    if not args.pt.exists():
        raise SystemExit(f"No checkpoint at {args.pt}. Train first: scripts/train.py")
    export(args.pt, args.npz, allow_downgrade=args.allow_capability_downgrade)


if __name__ == "__main__":
    main()
