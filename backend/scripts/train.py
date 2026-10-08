"""Train candidates on chronological train/selection splits, evaluate once on a later test.

Production requires the downloaded released NPZ and a persisted, read-back dataset reservation.
Early stopping and best-of-N selection never read the final test. The publication gate compares
candidate and actual incumbent on identical final-test rows, using each model's pinned vocabulary.
A failed attempt still consumes its test snapshot; later runs need at least 1,000 newer rows.
Metrics include calibration and partial-draft diagnostics. These are reported rather than
inventing unvalidated hard AUC/ECE or single-pick thresholds. Research runs can omit the incumbent,
but their artifacts cannot pass the separate publisher checks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bsdraft.constants import PROCESSED_DIR, REPO_ROOT, RAW_DIR  # noqa: E402
from bsdraft.data.balance_eras import current_balance_era  # noqa: E402
from bsdraft.data import dataset as D  # noqa: E402
from bsdraft.data import encoders as E  # noqa: E402
from bsdraft.models.winprob import ModelConfig, WinProbNet  # noqa: E402

from bsdraft.models import evaluation as EV
from bsdraft.models.releases import sha256_file

DOCS = REPO_ROOT / "docs"


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(total)


# Draft states (known picks per side) sampled during training, besides the full (3, 3).
# (0, 0) is excluded: antisymmetry makes its logit identically 0 with zero gradient, so
# training on it is pure wasted compute — it still predicts exactly 0.5 at inference.
PARTIAL_STATES = np.array([(a, b) for a in range(4) for b in range(4)
                           if (a, b) not in ((3, 3), (0, 0))], dtype=np.int64)


def mask_to_known(team: np.ndarray, k, mask_row: int, rng: np.random.Generator) -> np.ndarray:
    """Keep a uniformly random subset of ``k`` picks per row of a (N, 3) team array and
    replace the rest with ``mask_row``. ``k`` is a scalar or an (N,) array."""
    known = np.argsort(rng.random(team.shape), axis=1) < np.asarray(k).reshape(-1, 1)
    return np.where(known, team, mask_row)


def mask_teams(team_a: np.ndarray, team_b: np.ndarray, mask_row: int, p_full: float,
               rng: np.random.Generator) -> tuple:
    """Masked copies of (N, 3) team arrays. Each row keeps the full comp with probability
    ``p_full``; otherwise a draft state (k_a, k_b) is drawn uniformly from PARTIAL_STATES
    and a uniformly random subset of each team beyond k known picks is replaced by
    ``mask_row``. Fresh masks per call = free augmentation across epochs."""
    n = team_a.shape[0]
    ka = np.full(n, 3, dtype=np.int64)
    kb = np.full(n, 3, dtype=np.int64)
    partial = rng.random(n) >= p_full
    if partial.any():
        states = PARTIAL_STATES[rng.integers(len(PARTIAL_STATES), size=int(partial.sum()))]
        ka[partial], kb[partial] = states[:, 0], states[:, 1]
    return mask_to_known(team_a, ka, mask_row, rng), mask_to_known(team_b, kb, mask_row, rng)


def brawler_diff_features(team_a: np.ndarray, team_b: np.ndarray, n_brawlers: int) -> np.ndarray:
    """+1 per team_a brawler, -1 per team_b brawler — an antisymmetric linear baseline."""
    x = np.zeros((len(team_a), n_brawlers), dtype=np.float32)
    rows = np.arange(len(team_a))[:, None]
    np.add.at(x, (rows, team_a), 1.0)
    np.add.at(x, (rows, team_b), -1.0)
    return x


def _train_candidate(seed: int, cfg: ModelConfig, args, shared: dict) -> dict:
    """Train one masked model at ``seed`` and score it on the SHARED selection split.

    Returns the fitted (best-epoch) model plus its full-comp val metrics and training curves.
    Deliberately does no artifact writing, no gate check, and no partial-state eval — the caller
    keeps only the winning candidate and does those once. Every candidate reads the same
    seed-independent tensors from ``shared`` (including the val split), so candidates differ
    purely in weight init and per-epoch mask draws and are directly comparable on identical rows.
    """
    ta, tb, mp, mo = shared["ta"], shared["tb"], shared["mp"], shared["mo"]
    tr_i, vai, yv = shared["tr_i"], shared["vai"], shared["yv"]
    ta_tr, tb_tr = shared["ta_tr"], shared["tb_tr"]
    y_tr, wt_tr, mp_tr, mo_tr = shared["y_tr"], shared["wt_tr"], shared["mp_tr"], shared["mo_tr"]

    torch.manual_seed(seed)
    model = WinProbNet(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(seed)

    # Fixed masked copy of the val split (same p-full mixture as training): early stopping
    # runs on this — it contains full comps, so full-comp regressions still move it — while
    # the headline metrics below stay unmasked and comparable across retrains.
    va_m, vb_m = mask_teams(shared["ta_val_np"], shared["tb_val_np"], cfg.mask_row, args.p_full,
                            np.random.default_rng(seed + 1))
    tam_v, tbm_v = torch.from_numpy(va_m), torch.from_numpy(vb_m)

    def batches(n_rows, bs):
        order = torch.randperm(n_rows)
        for k in range(0, n_rows, bs):
            yield order[k:k + bs]

    # Early stopping tracks the mixed (masked) loss — the model's actual job — while the
    # full-comp loss is recorded alongside so a full-comp regression is visible per epoch.
    history_mix, history_full = [], []
    best_ll, best_state, bad, patience = float("inf"), None, 0, 6
    for _ in range(args.epochs):
        ta_m, tb_m = mask_teams(ta_tr, tb_tr, cfg.mask_row, args.p_full, rng)
        tam, tbm = torch.from_numpy(ta_m), torch.from_numpy(tb_m)
        model.train()
        for bi in batches(len(tr_i), args.batch):
            opt.zero_grad()
            logit = model(tam[bi], tbm[bi], mp_tr[bi], mo_tr[bi])
            loss = (bce(logit, y_tr[bi]) * wt_tr[bi]).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad(), EV.serving_map_context(model, shared["map_train_rows"]):
            pv_mix = torch.sigmoid(model(tam_v, tbm_v, mp[vai], mo[vai])).numpy()
            pv_full = torch.sigmoid(model(ta[vai], tb[vai], mp[vai], mo[vai])).numpy()
        vll = log_loss(yv, pv_mix, labels=[0, 1])
        history_mix.append(vll)
        history_full.append(float(log_loss(yv, pv_full, labels=[0, 1])))
        if vll < best_ll - 1e-4:
            best_ll, bad = vll, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), EV.serving_map_context(model, shared["map_train_rows"]):
        pv = torch.sigmoid(model(ta[vai], tb[vai], mp[vai], mo[vai])).numpy()
        pv_mix = torch.sigmoid(model(tam_v, tbm_v, mp[vai], mo[vai])).numpy()
    return {
        "seed": seed, "model": model, "pv": pv,
        "m_ll": float(log_loss(yv, pv, labels=[0, 1])),
        "m_auc": float(roc_auc_score(yv, pv)),
        "m_acc": float(((pv > 0.5) == yv.astype(bool)).mean()),
        "m_ece": float(expected_calibration_error(pv, yv)),
        "mix_ll": float(log_loss(yv, pv_mix, labels=[0, 1])),
        "history_mix": history_mix, "history_full": history_full,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15, help="chronological selection fraction")
    ap.add_argument("--test-frac", type=float, default=0.15, help="final chronological test fraction")
    ap.add_argument("--min-test-rows", type=int, default=1000)
    ap.add_argument("--incumbent-npz", type=Path, help="downloaded released model, never a local .pt")
    ap.add_argument("--require-incumbent", action="store_true", help="fail closed for publication")
    ap.add_argument("--allow-legacy-incumbent", action="store_true",
                    help="one-time explicit migration: incumbent training/test overlap is unknown")
    ap.add_argument("--matches", type=Path, default=RAW_DIR / "matches.jsonl")
    ap.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--metrics", type=Path, default=DOCS / "metrics.json")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--evaluation-reservation", type=Path,
                    help="read-back reservation of this dataset from the persistent attempt ledger")
    ap.add_argument("--halflife-days", type=float, default=30.0)
    ap.add_argument("--all-eras", action="store_true",
                    help="explicit research/backtest mode: train on pre-balance history too")
    ap.add_argument("--min-ts", type=int, default=0,
                    help="custom inclusive UTC epoch cutoff; mutually exclusive with --all-eras")
    # On by default since 2026-09-03. It was `store_true`, and collect.py's unattended
    # --retrain-on-shift argv never passed it, so every automatic retrain quietly produced a
    # model without the term and published it — the deployed artifact lost the capability for a
    # full retrain cycle with nothing erroring. Two other guards now back this up: collect.py
    # names the flag explicitly, and export_model.py refuses a capability downgrade.
    # This is only the default for NEW training runs. ModelConfig.class_synergy stays
    # default-False deliberately: that default is deserialization semantics for checkpoints
    # written before the term existed (the paired-baseline load below rebuilds a ModelConfig
    # from a stored config dict, and would fail load_state_dict if "absent" meant "on").
    ap.add_argument("--class-synergy", action=argparse.BooleanOptionalAction, default=True,
                    help="learnable symmetric class x class within-team synergy matrix "
                         "(archetype-level; pools every same-class pairing into one estimate), "
                         "interpretable as 'which archetype pairs win together'. On by default; "
                         "--no-class-synergy trains without it. The learned signal is weak "
                         "— see docs/model-evaluation.md.")
    ap.add_argument("--p-full", type=float, default=0.7,
                    help="probability a training example keeps its full 3v3 (rest are masked "
                         "to random partial draft states). 0.7 held full-comp parity with the "
                         "unmasked control while matching 0.5's partial-state quality; raise "
                         "it if the paired full-comp gate ever regresses")
    ap.add_argument("--max-full-delta", type=float, default=0.002,
                    help="hard gate: abort (exit 1, no artifacts written) if full-comp test "
                         "logloss exceeds the released incumbent's by more than this on the "
                         "same rows — keeps the unattended --retrain-on-shift path from "
                         "publishing a regressed model. Set <0 to disable.")
    ap.add_argument("--seed", type=int, default=0,
                    help="base RNG seed for weight initialization and masks; candidates share "
                         "the same chronological selection split.")
    ap.add_argument("--candidates", type=int, default=1,
                    help="best-of-N: train this many models (seeds seed..seed+N-1) on the SAME "
                         "selection split and keep the lowest selection logloss; the test gate is applied to "
                         "the winner only. The paired full-comp delta swings more between seeds "
                         "(~0.0035) than the 0.002 gate, so a single unattended retrain passes or "
                         "fails by luck — the crawler's --retrain-on-shift path uses N>1 to fix "
                         "that. Costs Nx training time. N=1 reproduces single-seed training.")
    args = ap.parse_args()

    if args.candidates < 1:
        raise SystemExit("--candidates must be >= 1")
    if args.min_ts < 0:
        raise SystemExit("--min-ts must be >= 0")
    if args.all_eras and args.min_ts:
        raise SystemExit("--all-eras and --min-ts are mutually exclusive")

    era = current_balance_era()
    if args.all_eras:
        analysis_start_ts, analysis_era_id = 0, ""
    elif args.min_ts:
        analysis_start_ts = args.min_ts
        analysis_era_id = era.id if era and era.start_ts == analysis_start_ts else f"custom:{analysis_start_ts}"
    elif era:
        analysis_start_ts, analysis_era_id = era.start_ts, era.id
    else:
        analysis_start_ts, analysis_era_id = 0, ""

    if args.require_incumbent and args.min_test_rows < 1000:
        raise SystemExit("publication requires at least 1000 final-test rows")
    if args.require_incumbent and (args.max_full_delta < 0 or not np.isfinite(args.max_full_delta)):
        raise SystemExit("--require-incumbent cannot disable the regression gate")
    if args.require_incumbent and (not era or analysis_start_ts != era.start_ts or analysis_era_id != era.id):
        raise SystemExit("publication training must use the active balance era")
    try:
        incumbent, incumbent_evaluation, incumbent_sha = EV.load_incumbent(
            args.incumbent_npz, required=args.require_incumbent,
            allow_legacy=args.allow_legacy_incumbent)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"unusable released incumbent: {exc}") from exc

    dataset_digest = sha256_file(args.matches)
    reservation = json.loads(args.evaluation_reservation.read_text()) if args.evaluation_reservation else None
    if args.require_incumbent and reservation is None:
        raise SystemExit("publication requires --evaluation-reservation from the persistent attempt ledger")
    if reservation is not None and (reservation.get("dataset_sha256") != dataset_digest or
                                    not reservation.get("reservation_id")):
        raise SystemExit("evaluation reservation belongs to a different dataset")
    test_after = max(int(incumbent_evaluation.get("data_through_ts", 0)),
                     int(reservation.get("test_after_ts", 0)) if reservation else 0)
    ds = D.build_dataset(path=args.matches, min_ts=analysis_start_ts)
    if reservation is not None and int(reservation.get("consumed_through_ts", 0)) != int(ds.ts.max()):
        raise SystemExit("dataset timestamp does not match the reserved evaluation snapshot")
    n = len(ds)
    print(f"dataset ({analysis_era_id or 'all eras'}): {D.summary(ds)}")
    try:
        tr_i, selection_i, test_i = EV.temporal_split(
            ds.ts, selection_frac=args.val_frac, test_frac=args.test_frac,
            previous_data_through_ts=test_after,
            min_test=args.min_test_rows)
        for name, rows in (("train", tr_i), ("selection", selection_i), ("test", test_i)):
            if set(np.unique(ds.y[rows])) != {0, 1}:
                raise ValueError(f"{name} split needs both outcome classes")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"temporal split: train={len(tr_i)}, selection={len(selection_i)}, test={len(test_i)}")
    ta, tb = torch.tensor(ds.team_a), torch.tensor(ds.team_b)
    mp, mo = torch.tensor(ds.map_idx), torch.tensor(ds.mode_idx)
    y = torch.tensor(ds.y)
    # The training anchor and weights use only training rows, never final-test timestamps.
    if args.halflife_days > 0:
        w = np.power(0.5, (int(ds.ts[tr_i].max()) - ds.ts[tr_i]) /
                     (args.halflife_days * 86400.0)).astype(np.float32)
        w /= w.mean()
    else:
        w = np.ones(len(tr_i), dtype=np.float32)
    tri, vai = torch.tensor(tr_i), torch.tensor(selection_i)
    cfg = ModelConfig(E.num_brawlers(), E.num_maps(), E.num_modes(), mask_row=E.num_brawlers(),
                      class_synergy=args.class_synergy)
    shared = {
        "ta": ta, "tb": tb, "mp": mp, "mo": mo,
        "tr_i": tr_i, "vai": vai, "yv": ds.y[selection_i],
        "ta_tr": ds.team_a[tr_i], "tb_tr": ds.team_b[tr_i],
        "ta_val_np": ds.team_a[selection_i], "tb_val_np": ds.team_b[selection_i],
        "y_tr": y[tri], "wt_tr": torch.tensor(w), "mp_tr": mp[tri], "mo_tr": mo[tri],
        "map_train_rows": np.bincount(ds.map_idx[tr_i], minlength=E.num_maps()),
    }
    candidates = []
    for index in range(args.candidates):
        seed = args.seed + index
        print(f"candidate {index + 1}/{args.candidates} (seed {seed})")
        candidate = _train_candidate(seed, cfg, args, shared)
        candidates.append(candidate)
        print(f"selection full-comp logloss {candidate['m_ll']:.6f}")
    best = min(candidates, key=lambda candidate: candidate["m_ll"])
    model, chosen_seed = best["model"], best["seed"]
    history_mix, history_full = best["history_mix"], best["history_full"]
    n_cand = args.candidates
    # Final test rows have never guided candidate/epoch selection.
    val_i, n_val = test_i, len(test_i)
    vai, yv = torch.tensor(test_i), ds.y[test_i]
    with torch.no_grad(), EV.serving_map_context(model, shared["map_train_rows"]):
        pv = torch.sigmoid(model(ta[vai], tb[vai], mp[vai], mo[vai])).numpy()
    final = EV.metrics(yv, pv)
    m_ll, m_auc, m_acc, m_ece = (final[key] for key in ("logloss", "auc", "acc", "ece"))
    const_ll = EV.metrics(yv, np.full(len(yv), 0.5))["logloss"]
    x_tr = brawler_diff_features(ds.team_a[tr_i], ds.team_b[tr_i], E.num_brawlers())
    x_test = brawler_diff_features(ds.team_a[test_i], ds.team_b[test_i], E.num_brawlers())
    logreg = LogisticRegression(max_iter=2000, C=1.0).fit(x_tr, ds.y[tr_i])
    lr_metrics = EV.metrics(yv, logreg.predict_proba(x_test)[:, 1])
    lr_ll, lr_auc, lr_acc = (lr_metrics[key] for key in ("logloss", "auc", "acc"))
    by_brawler_row = {row: bid for bid, row in E.brawler_encoder().items()}
    by_map_row = {0: 0, **{row: mid for mid, row in E.map_encoder().items()}}
    by_mode_row = {0: "", **{row: mode for mode, row in E.mode_encoder().items()}}
    baseline = None
    if incumbent is not None:
        baseline = EV.metrics(yv, EV.predict_incumbent(
            incumbent, ds, test_i, brawler_ids=by_brawler_row,
            map_ids=by_map_row, modes=by_mode_row))
    try:
        gate = EV.regression_gate(final, baseline, require_incumbent=args.require_incumbent,
                                  max_full_delta=args.max_full_delta, incumbent_sha256=incumbent_sha)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    overlap = ("fresh_after_previous_snapshot" if incumbent_evaluation.get("data_through_ts") else
               "unknown_legacy" if incumbent is not None else "no_incumbent")
    gate["incumbent_test_overlap"] = overlap
    print("final temporal-test metrics:", json.dumps(final, sort_keys=True))
    print("paired publication gate:", json.dumps(gate, sort_keys=True))
    mix_rng = np.random.default_rng(chosen_seed + 1)
    va_m, vb_m = mask_teams(ds.team_a[test_i], ds.team_b[test_i], cfg.mask_row, args.p_full, mix_rng)
    with torch.no_grad(), EV.serving_map_context(model, shared["map_train_rows"]):
        mix_p = torch.sigmoid(model(torch.from_numpy(va_m), torch.from_numpy(vb_m), mp[vai], mo[vai])).numpy()
    mix_ll = EV.metrics(yv, mix_p)["logloss"]

    # --- partial-draft states: how much is knowing more of the draft worth? ---
    # Each row masks the whole final test split to one fixed (known_ours, known_theirs) state.
    # The 1v0 row is also compared against a shrunk brawler-map winrate marginal built on the
    # train split — the cheapest possible single-pick predictor. If the net loses to it, the
    # mask-in-mean design is washing out the single-pick signal and needs rework.
    wins = np.zeros((E.num_maps(), int(cfg.mask_row) + 1), dtype=np.float64)
    games = np.zeros_like(wins)
    for team, won in ((ds.team_a[tr_i], ds.y[tr_i]), (ds.team_b[tr_i], 1.0 - ds.y[tr_i])):
        for j in range(3):
            np.add.at(games, (ds.map_idx[tr_i], team[:, j]), 1.0)
            np.add.at(wins, (ds.map_idx[tr_i], team[:, j]), won)
    emp_wr = (wins + 5.0) / (games + 10.0)   # shrunk toward 0.5 with 10 pseudo-games

    partial_metrics = {"mixture_logloss": mix_ll, "p_full": args.p_full, "states": {}}
    print("\n=== partial draft states, final temporal test (value of knowing more of the draft) ===")
    print(f"{'state':<10}{'logloss':>10}{'AUC':>8}{'ECE':>8}{'mean|p-.5|':>12}")
    for ka, kb in ((1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)):
        srng = np.random.default_rng(chosen_seed + 100 + 10 * ka + kb)
        sa_np = mask_to_known(ds.team_a[val_i], ka, cfg.mask_row, srng)
        sb_np = mask_to_known(ds.team_b[val_i], kb, cfg.mask_row, srng)
        sa, sb = torch.from_numpy(sa_np), torch.from_numpy(sb_np)
        with torch.no_grad(), EV.serving_map_context(model, shared["map_train_rows"]):
            ps = torch.sigmoid(model(sa, sb, mp[vai], mo[vai])).numpy()
        s_ll = float(log_loss(yv, ps, labels=[0, 1]))
        s_auc = float(roc_auc_score(yv, ps))
        s_ece = float(expected_calibration_error(ps, yv))
        spread = float(np.abs(ps - 0.5).mean())
        partial_metrics["states"][f"{ka}v{kb}"] = {
            "logloss": s_ll, "auc": s_auc, "ece": s_ece, "mean_abs_edge": spread,
        }
        print(f"{f'{ka}v{kb}':<10}{s_ll:>10.4f}{s_auc:>8.3f}{s_ece:>8.3f}{spread:>12.3f}")
        if (ka, kb) == (1, 0):
            known_ids = sa_np[sa_np != cfg.mask_row]                     # the one known pick/row
            p_emp = emp_wr[ds.map_idx[val_i], known_ids]
            e_ll = float(log_loss(yv, p_emp, labels=[0, 1]))
            partial_metrics["empirical_1v0_logloss"] = e_ll
            verdict = "net >= empirical marginal, OK" if s_ll <= e_ll else \
                "NET LOSES to the empirical marginal — single-pick signal is being washed out"
            print(f"{'  1v0 emp':<10}{e_ll:>10.4f}{'-':>8}{'-':>8}{'-':>12}   DIAGNOSTIC: {verdict}")

    # --- save artifacts ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The exact vocabulary this model was trained against, by embedding row. export_model.py
    # compares these ids to the live reference at export time — identity, not just counts, so
    # a same-size catalog swap between train and export fails loudly instead of silently
    # re-pinning ids onto neighbours' trained rows.
    map_rows = np.bincount(ds.map_idx[tr_i], minlength=E.num_maps())
    trained_vocab = {
        "brawler_ids": [int(b) for b, _ in sorted(E.brawler_encoder().items(), key=lambda kv: kv[1])],
        "map_ids": [int(m) for m, _ in sorted(E.map_encoder().items(), key=lambda kv: kv[1])],
        "modes": [s for s, _ in sorted(E.mode_encoder().items(), key=lambda kv: kv[1])],
        # Training rows per map, aligned with map_ids. The vocab is the whole ranked-mode catalog,
        # most of which never carries a game; export_model.py pins only rows these counts show
        # were actually learned, so an untrained map serves the mean trained row, not its init.
        "map_train_rows": [int(map_rows[r]) for _, r in
                           sorted(E.map_encoder().items(), key=lambda kv: kv[1])],
    }
    trained_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                   check=True, capture_output=True, text=True).stdout.strip()
    evaluation = {
        "training_run_id": str(uuid.uuid4()), "trained_at": trained_at,
        "source_commit": source_commit, "dataset_sha256": dataset_digest,
        "source_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal",
                                             "--", "backend", ".github", "data/reference", "frontend"],
                                            cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()),
        "evaluation_reservation": reservation,
        "data_through_ts": int(ds.ts.max()), "training_until_ts": int(ds.ts[tr_i].max()),
        "selection_until_ts": int(ds.ts[selection_i].max()),
        "test_start_ts": int(ds.ts[test_i].min()), "n_train": len(tr_i),
        "n_selection": len(selection_i), "n_test": len(test_i), "n_total": n,
        "split_method": "chronological; timestamp ties kept together; final test excluded from selection",
        "publication_gate": gate, "embedding": final, "logreg": lr_metrics,
        "const": {"logloss": float(const_ll)}, "embedding_partial": partial_metrics,
        "baseline_released_incumbent": baseline,
        "n_candidates": n_cand, "chosen_seed": chosen_seed,
        "candidates": [{"seed": c["seed"], "selection_logloss": c["m_ll"]} for c in candidates],
    }
    if reservation is not None and reservation.get("source_commit") != source_commit:
        raise SystemExit("source commit changed since evaluation reservation")
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "vocab": trained_vocab,
        "analysis": {"era_id": analysis_era_id, "start_ts": analysis_start_ts},
        "evaluation": evaluation,
    },
               args.output_dir / "winprob.pt")
    DOCS.mkdir(parents=True, exist_ok=True)
    metrics = {
        **evaluation, "n_val": int(n_val),
        "analysis": {"era_id": analysis_era_id, "start_ts": analysis_start_ts},
        "evaluation_kind": "final temporal test; not candidate selection",
        "const": {"logloss": float(const_ll)},
        "logreg": lr_metrics,
        "embedding": {"logloss": float(m_ll), "acc": m_acc, "auc": float(m_auc), "ece": m_ece},
        "embedding_partial": partial_metrics,
        # Actual released incumbent scored on the identical final-test rows.
        "baseline_prev_checkpoint": baseline,
        # Best-of-N provenance: which seed shipped and how the candidates compared (empty deltas
        # when there was no paired baseline, e.g. after a vocabulary change).
        "n_candidates": n_cand,
        "chosen_seed": chosen_seed,
        "candidates": evaluation["candidates"],
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, allow_nan=False))
    if args.no_charts:
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history_mix, marker="o", ms=3, label="masked mixture")
    ax[0].plot(history_full, marker="o", ms=3, label="full comps")
    ax[0].axhline(0.6931, ls="--", c="gray", label="always 0.5")
    ax[0].axhline(lr_ll, ls=":", c="orange", label="logreg (full comps)")
    ax[0].set_title("validation log-loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    edges = np.linspace(0, 1, 11)
    mids, accs = [], []
    for i in range(10):
        m = (pv > edges[i]) & (pv <= edges[i + 1]) if i else (pv >= edges[i]) & (pv <= edges[i + 1])
        if m.sum():
            mids.append(pv[m].mean()); accs.append(yv[m].mean())
    ax[1].plot([0, 1], [0, 1], ls="--", c="gray")
    ax[1].plot(mids, accs, marker="o")
    ax[1].set_title(f"calibration (ECE={m_ece:.3f})")
    ax[1].set_xlabel("predicted P(win)"); ax[1].set_ylabel("observed win-rate")
    fig.tight_layout()
    fig.savefig(args.metrics.parent / "training.png", dpi=120)

    print(f"\nsaved model  -> {args.output_dir / 'winprob.pt'}")
    print(f"saved charts -> {DOCS / 'training.png'}")
    print(f"saved metrics-> {args.metrics}")


if __name__ == "__main__":
    main()
