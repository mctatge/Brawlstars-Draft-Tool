# Reliability and model publication

The product is an experimental public beta on a small Render instance and an optional home
roster service. The acceptance target is correct, bounded behavior for a small number of
visitors, plus a usable first demonstration. Large-scale capacity and improved player win
rate are unmeasured. Do not use production load as an acceptance test.

## Draft and account correctness

- General, personal and top-meta results carry the complete request identity. Changing map,
  board, active slot or personalization invalidates success, error and loading callbacks.
  Previous advice is hidden immediately, including during the debounce window.
- Every placement checks uniqueness and current ownership/power eligibility for the user's
  seat, including cards and the brawler pool.
- Rank requests share cancellation across manual lookup and refresh. Clearing/changing the
  account cannot restore it through an old response. A live unplaced rank is authoritative.
- A failed roster refresh, including HTTP 200 with `loaded:false`, retains a last-good roster
  only for the same normalized tag and visibly marks it stale. Switching accounts clears it.
- Requests bound connection and response-body reads; boot has a 90-second total budget and
  then offers retry. The recorded `/demo` needs neither live API nor player account.

## Service bounds and last-good artifacts

The configured Uvicorn service is one process. Limits below are **per process**, not shared
across multiple workers, the home host and a separate crawler.

- At most two personal-history scans run concurrently across warm, direct and startup paths.
  Cold/older-than-one-hour history is omitted while a bounded background warm runs; a recent
  cached history can be used while refreshing. The request never runs a whole dataset scan.
- Roster and live rank share profile single-flight, a common limiter, an eight-request pending
  bound, an eight-second fetch deadline, successful TTL caching and a ten-second failure cache.
  This reduces repeated upstream calls; it is not abuse protection or a capacity guarantee.
- Downloads stage before promotion. Gzip streams must finish; structured artifacts must pass
  format/load checks and applicable balance-era checks before file, SHA or ETag changes.
  Multi-GB matches get transport validation plus first/last-row schema checks, not an exhaustive
  replay of every row on the public host.
- A configured but unavailable meta report returns unavailable rather than scanning the archive.
  If published stats are unavailable, dataset updates refresh the capped current-era fallback.
- `/api/health` returns HTTP 503 without usable empirical matches. Otherwise it reports readiness,
  degraded syncs, source/count/era, model identity, newest-match age and per-artifact attempt,
  success and error categories. Data age is unknown without a usable timestamped meta report.

## Which safeguards run where

| Check | Pull-request CI | Retraining workflow |
| --- | --- | --- |
| Backend regressions and Torch/NumPy parity | Full backend suite | Focused safeguard/parity suite |
| Frontend request ordering, account failure and deadline regressions | Yes | No |
| Frontend TypeScript and static build | Yes | No |
| Compare against actual released NumPy incumbent | Synthetic failure tests | Mandatory real incumbent |
| Chronological selection/final test separation | Synthetic split tests | Actual dataset |
| Persistent consumed-test watermark | Mocked GitHub tests | Reserve and read back before training |
| Full-team log-loss regression bound | Synthetic rejection tests | Mandatory, 0.0035 |
| Native partial-draft capability/export guard | Synthetic parity/capability tests | Train and export |
| AUC, calibration, empirical first-pick comparison | Reporting tested | Diagnostic, not threshold gates |
| Immutable upload verification and pointer promotion | Failure injection | Actual GitHub promotion |

`.github/workflows/ci.yml` runs on pull requests and main pushes without credentials or live
data. `.github/workflows/retrain-model.yml` is dispatch-only and does not run a full retrain
on every pull request. Check GitHub run results before calling checks green; workflow source
alone does not prove a run occurred. Branch-protection required contexts must be configured
after `backend` and `frontend` have successful runs.

The previous retraining workflow downloaded only matches. With no checkpoint baseline its
regression comparison could be absent (`delta=n/a`), so a successful historical run did not
prove the documented incumbent gate ran. The corrected path downloads released weights and
passes `--require-incumbent` to both training and export. Missing or corrupt baseline fails closed.

## Evaluation and provenance

The actual incumbent's pinned IDs resolve its own vocabulary; current catalog row order cannot
silently change the comparison. Target splits are chronological 70/15/15, with tied timestamps
kept together. Early stopping and best-candidate selection use selection rows only. At least
1,000 final test rows, all newer than the incumbent's full snapshot and previous reserved
attempt, are required. Fractions can shift when the fresh-data constraint moves the boundary.

The serialized retraining workflow reserves the snapshot's maximum eligible timestamp in the
`model-evaluation` GitHub release body **before** training and verifies readback. Failed attempts
also consume the snapshot. Retries must wait for sufficient newer data. Do not reset this ledger
to bypass a refusal. Other concurrent publishers are unsupported; the workflow concurrency
group serializes its own writers.

The final paired full-team log-loss change must not exceed 0.0035. This allows a bounded
regression; it does not promise every published model is better. AUC, ECE and empirical 1v0
results are recorded, but no validated threshold enforces them. The legacy bootstrap has
unknown incumbent/test overlap and says so in metrics. No unqualified “untouched test” claim
applies to that comparison. Offline predictive evaluation is not a causal win-rate study.

Metrics record the source commit, dirty-source status, dataset hash, split counts and timestamps,
training run, analysis era, incumbent hash, reservation, gate result and exact weights hash.
The NPZ embeds matching evaluation provenance. Dirty or inconsistent bundles cannot publish.

## Immutable model bundles

Model publication creates `model-<weights digest prefix>-<metrics digest prefix>` containing
`winprob.npz` and `metrics.json`. It verifies the uploaded bytes and publishes that release
before atomically replacing the JSON body of `model-current`. No model asset is deleted or
uploaded with `--clobber`. Failure before pointer promotion leaves the old pointer intact;
an upload failure can leave an unreferenced draft release.

`MODEL_MANIFEST_URL` reads the public GitHub `model-current` release response. The API stages
weights and metrics in a version directory, verifies hashes, sizes, embedded/report provenance,
parameter count, era and actual inference, then replaces its local pointer. It keeps the last
usable bundle on a rejected update. `/api/model` reports that loaded bundle's metrics. A legacy
model has no verified report; historical `docs/metrics.json` is never substituted.

`MODEL_URL` remains a migration fallback only while the remote pointer is absent (HTTP 404)
and no usable versioned bundle is cached. Network/integrity failure does not trigger a legacy
download. Rollback promotes a previously verified immutable manifest; it does not overwrite
weights. Rolling back preserves the evaluation watermark.

## Rollout

1. Review the change and require successful backend/frontend pull-request checks before merge.
   Merging main deploys the API/frontend through the existing host integrations.
2. After successful CI runs exist, require `backend` and `frontend` in branch protection.
3. Restart the long-lived home crawler after merging builder changes; otherwise its imported
   old code can keep publishing legacy statistics. Rebuild/publish current-era stats and
   verify `/api/health` shows the expected stats era/source. This is a separate operational action.
4. For the **first** bundle only, dispatch `retrain-model` with `allow_legacy_incumbent=true`.
   This explicitly acknowledges unknown legacy overlap. Leave it false thereafter. Confirm the
   attempt reservation, incumbent comparison, export guard and pointer readback in the run.
5. Verify `/api/model` shows the promoted release/digest and `/model` displays that same report.
   A refusal is expected when fresh test data or other safeguards are insufficient. Do not
   relax the gate merely to obtain a green run.
6. Exercise rapid map/pick/account changes and a failed roster refresh on the deployed build.
   Confirm `/demo` works while the live API is unavailable. No production stress test is required.

## Recorded example

`frontend/data/demo-draft.json` contains only aggregate recommendation outputs, hashes and
sample metadata. The initial snapshot uses 5,990 labeled current-era matches from a bounded
local sample and identified published weights. It is not a full-population evaluation.
`backend/scripts/export_demo.py` captures three boards from explicit matching-era model/stats
files with no network call. Refresh the example deliberately, review outputs and keep its date,
sample scope and hashes visible. No player tags, raw battle logs or credentials belong in it.
