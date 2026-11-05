# Phase 5 validation — November 4, 2026

This validation used the existing **SYNTHETIC** Phase 3 snapshot. No live source
was fabricated or substituted. The fixture deliberately gives qualifying and
race identical classifications, so qualifying-based references can be perfect.
These metrics validate software behavior, not real Formula 1 performance or
calibration. Existing live-data completion/availability and roster reviews are
still needed before an honest real-data evaluation.

## Checks performed

```sh
.venv/bin/pytest -q tests/test_race.py
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/pytest -q
.venv/bin/python -m build --no-isolation
```

25 new race/podium tests passed; **123 tests passed overall**. Lint and formatting
passed; wheel and source distribution built. Focused checks cover:

- Every model family, all historical strategies, dynamic full-field orders,
  unseen drivers/teams and missing optional practice.
- Target-race reads forbidden, raw status/grid columns rejected by the allowlist,
  actual qualifying required, and grid evidence strictly separated from qualifying.
- Numeric classified-retirement/DSQ/DNS positions retained; missing positions
  masked and reported; a missing-position DNS remains a known podium negative.
- Tuning before reserved postprocessing events, each calibration input generated
  by an earlier trained model, and outer/future label changes unable to affect
  fold parameters, calibration or thresholds.
- Known-value metrics/reliability, independent probability sums, differing top
  threes, single-class models/calibration fallback, and missing postprocessing.
- Cutoff/publication/selection guards, sealed final test, model bundle integrity,
  saved preprocessing, metadata, and end-to-end combined prediction CLI.

The sandbox emitted one nonfatal joblib physical-core discovery warning during
pytest and CPU-cache `sysctl` probe messages during standalone Parquet reads.
Commands completed successfully; these were not data-source failures.

## Preserved experiment and earlier work

Verified the existing manifest and immutable output receipts:

| Object | ID |
| --- | --- |
| Frozen manifest | `3cc7b3bf005e808e360595275f3e24c35bbd8a26c962bafdef8e6ee766f968dd` |
| Phase 3 features | `ccc37825281ab0102fb44f15dc516e04211ad3896fec45bfbf78b71699c3a64d` |
| Legacy baseline artifacts | `78096ffbb4fd06a12d7f662dec9199dcb04c4b1555cd9142e3eac1d294578371` |
| Phase 4 comparison | `911ef333cd8d600a851515974d9717a63813aaf980b5870c944c98e2a8bc8839` |
| Phase 4 prediction | `0147bf7fab7804aa0c4599e4f74ad38fe6d02405ac08456f3dfdfb2727fe29ef` |

The final block is still `2026-04` through `2026-07`. No feature implementation,
source snapshot, manifest or earlier artifact was changed. The existing
qualifying fold-membership checker now also accepts an explicit race task;
qualifying behavior and all prior tests remain intact.

## Default comparison

Exact command executed from the project root:

```sh
.venv/bin/f1ml --config configs/default.toml race-compare \
  --manifest data/phase3-final/manifest.json \
  --features data/phase3-final/features --output data/phase5-final
```

Receipt: `891b065891bb56ae9dc6b116ac4089a5d50f999ab38c6c5d9928bd9dc6d6093a`.

Saved 42 outer-fold bundles and two selected refits, 54 event-metric rows, 22
common-cohort rows, 110 raw/delivered reliability-bin rows and 46 driver/event
target-coverage rows. The original 2026 final test remains sealed. The default
two-earlier-2026-event minimum makes A ineligible at `2026-01` and `2026-02`.
Each task therefore shares only `2026-03` across all nine learned cells and both
rules in its primary comparison. The report retains other eligible B/C folds.

For `2026-03`, B/C podium models fit calibration and thresholds on chronological
OOF predictions from `2026-01` and `2026-02`. A has insufficient earlier OOF
events and explicitly retains raw probabilities and threshold 0.5. With no
earlier parameter-tuning window after reserving these events, those outer podium
models use predeclared fixed parameter/weight defaults. Refit B/C models have
an earlier tuning event and two later postprocessing events. Tests with a smaller
predeclared postprocessing minimum additionally exercise tuned outer podium folds.

The selected learned race cell is Ridge/equal (common synthetic MAE 0, Spearman
1); other models and the qualifying reference tie on that event, so predeclared
tie order chooses Ridge. The selected learned podium cell is gradient boosting/
weighted (log loss 0.01955, Brier 0.0003761, average precision 1, ECE 0.01936).
The hard qualifying reference is perfect on this deliberately identical-order
fixture and outperforms the learned classifiers. Calibration worsens some other
cells, which is visible in the raw/delivered metrics. No calibration benefit is
assumed and no result is presented as a real-data accuracy claim.

See `data/phase5-final/report.md`, `summary.json`, `eligibility.json`,
`target_coverage.parquet`, `calibration.parquet`, and `tuning.json` for details.

## Combined cutoff prediction

The roster `data/phase5-inputs/2026-03-race.csv` was filtered from the existing
synthetic Phase 3 entrant CSV to event `2026-03`, task `race`. Its cutoff matches
the existing cutoff CSV. Both chosen fold models were trained/tuned/calibrated
only on earlier events; this example intentionally uses the historical fold
bundles, not later selected refits.

```sh
.venv/bin/f1ml predict-race \
  --race-model data/phase5-final/fold_models/race/2026-03/ridge/equal \
  --podium-model data/phase5-final/fold_models/podium/2026-03/logistic/equal \
  --dataset data/phase3-final/dataset \
  --manifest data/phase3-final/manifest.json --event 2026-03 \
  --entrants data/phase5-inputs/2026-03-race.csv \
  --cutoff 2026-03-23T16:00:00Z --output data/phase5-prediction
```

Prediction receipt:
`4ce9873f6b9cbafcda9b64c324df7bb014224f77947dc1f520fbcc7c21c51dd7`.
Verified seven unique entrants and a complete position permutation 1..7, separate
raw scores/probabilities, qualifying inputs present and unverified grid missing.
The receipt links the source/feature files, both model IDs/metadata and cutoff
audit. Actual race statuses remain outside prediction inputs/outputs.

The independent podium probability sum is **3.5528208**, unchanged by any
normalization. The race model's top three are synthetic drivers 3, 4, 5; the
classifier's probability top three are 2, 3, 4. Both sets and their disagreement
are saved explicitly. The threshold was selected on earlier development OOF
events; it happens to be 0.5 in this example.

Outputs are immutable; choose new output paths to repeat a run. The
[race/podium guide](race-podium.md) includes a fresh-demo recipe and the command
contract for later unseen events after their qualifying information is available.
No dashboard, deployment or final-test opening command was added.
