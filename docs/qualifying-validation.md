# Phase 4 validation — November 4, 2026

All model metrics below come from **SYNTHETIC fixtures**, not observed F1 events.
No live source was substituted, no new download was needed, and no final-test
labels were loaded for model comparison. The existing real-data cutoff/roster
review and timing-availability limitations remain unresolved.

## Verification

Executed with the existing `.venv` (Python 3.13.13, scikit-learn 1.9.1):

```sh
.venv/bin/pytest -q tests/test_qualifying.py
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/pytest -q
.venv/bin/python -m build --no-isolation
```

- 20 new qualifying tests passed; **98 total tests passed**.
- Lint and formatting checks passed; wheel and source distribution built.
- Tested nested selection, outer/future-label invariance, 2026-only eligibility,
  shared comparison cohorts, weight preservation with variable field sizes,
  known-value metrics, clean-practice fallback, missing training targets,
  unseen categories, fold-only preprocessing, permutation failures, strict
  chronology, publication-time guards, future-event CLI prediction, and bundle
  integrity verification before unpickling.
- One nonfatal joblib physical-core discovery warning appeared in this sandbox.
  Standalone Parquet commands also logged denied CPU-cache `sysctl` probes.
  All relevant commands completed successfully; these were not source failures.

## Existing snapshots preserved

Loaded and verified the existing frozen manifest and Phase 3 receipts:

| Object | ID |
| --- | --- |
| Manifest | `3cc7b3bf005e808e360595275f3e24c35bbd8a26c962bafdef8e6ee766f968dd` |
| Phase 3 features | `ccc37825281ab0102fb44f15dc516e04211ad3896fec45bfbf78b71699c3a64d` |
| Existing baseline artifacts | `78096ffbb4fd06a12d7f662dec9199dcb04c4b1555cd9142e3eac1d294578371` |

The sealed block is still `2026-04`, `2026-05`, `2026-06`, `2026-07`. No Phase 3
feature implementation was changed, so existing feature/code fingerprints remain
valid. Ingestion, legacy baseline training/prediction and prior artifacts remain.

## Default comparison run

Exact command executed from the project root:

```sh
.venv/bin/f1ml --config configs/default.toml qualifying-compare \
  --manifest data/phase3-final/manifest.json \
  --features data/phase3-final/features --output data/phase4-final
cat data/phase4-final/report.md
```

Receipt: `911ef333cd8d600a851515974d9717a63813aaf980b5870c944c98e2a8bc8839`.

The tiny source has four earlier 2024/2025 development weekends and three 2026
development weekends. Default A requires two earlier 2026 training weekends, so
`2026-01` and `2026-02` are explicitly ineligible for A. All nine learned cells
and both fixed rules share only `2026-03` in the primary comparison. Additional
B/C fold results remain visible but are excluded from the primary averages.
Saved 21 outer model bundles, 27 event-metric rows, 11 common-cohort rows,
candidate searches and one selected refit. Tests additionally use a smaller
predeclared search/minimum to exercise nested A tuning and a two-event common
cohort; they do not change the frozen test block.

On the single common synthetic event, the best learned cell was Ridge/equal:
MAE 2.5714 positions, Spearman −0.1071, pole accuracy 0, top-three overlap 1/3.
The recent baseline did better: MAE 2.0000, Spearman 0.1429, pole accuracy 0,
top-three overlap 2/3. Practice fell back completely to recent form because the
minimal demo has no Phase 3 comparable practice conditions. Its metrics match.
Top-ten overlap is trivially 1 for the seven-driver fixture. These outcomes
validate reporting and software behavior; they offer no real accuracy evidence.
See `report.md` and `summary.json` in the output for every model/strategy result.

## Cutoff prediction run

The input CSV `data/phase4-inputs/2026-03-qualifying.csv` was copied by filtering
the existing synthetic `data/phase3-final/entrants.csv` to event `2026-03`, task
`qualifying`; no entrant was inferred from the target classification. The cutoff
matches the existing Phase 3 cutoff CSV. Exact prediction command:

```sh
.venv/bin/f1ml predict-qualifying \
  --model data/phase4-final/fold_models/2026-03/ridge/equal \
  --dataset data/phase3-final/dataset \
  --manifest data/phase3-final/manifest.json --event 2026-03 \
  --entrants data/phase4-inputs/2026-03-qualifying.csv \
  --cutoff 2026-03-23T13:59:59Z --mode historical_reconstruction \
  --output data/phase4-prediction
```

Prediction receipt:
`0147bf7fab7804aa0c4599e4f74ad38fe6d02405ac08456f3dfdfb2727fe29ef`.
Verified all seven eligible entrants appear exactly once and the output positions
are `[1, 2, 3, 4, 5, 6, 7]`. Raw scores remain separate, including estimates above
1; they are not clipped or called probabilities. The fold model's training and
tuning dependencies precede this event. The selected refit would be rejected
for this same historical event because selection/refitting consumed it.

Output paths are immutable. Use new output paths when repeating these commands.
The [qualifying guide](qualifying.md) includes a complete fresh-demo recipe and
the future-event command contract. No dashboard or final-test runner was added.
