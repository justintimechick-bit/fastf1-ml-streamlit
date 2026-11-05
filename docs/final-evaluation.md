# Locked final evaluation (Phase 7)

The user's explicit authorization now permits one evaluation of the reserved
block after a model/protocol freeze. Ordinary feature, prediction and development
commands continue to reject test IDs. The only existing frozen experiment uses
**synthetic fixtures**: its evaluation validates software, not real F1 accuracy.
The live catalog lacks reviewed cutoffs/rosters and a genuine frozen experiment.

## Audit and lock

`final-lock` verifies receipt integrity, source/manifest/feature lineage, latest
eligible 2026 block membership, disjoint weekend roles and current code hashes.
Every saved development model and result table is audited, including nonwinning
candidates. Recorded feature, training, weighting, selection and calibration IDs
must exclude the holdout. Outer/inner folds must use earlier complete weekends.
No held-out component is loaded during input preparation, audit or locking.

This audit cannot prove absence of unlogged human access or establish independence
for fixtures sharing a generator. Manifest eligibility previously inspected target
availability/classification validity, not model performance. These limits are saved.

Keep the established selection rule: compare all three learned families with A
(earlier 2026), B (equal pooled 2024–2026) and C (2026-favored) on the same eligible
events. Minimize equal-event positional MAE or delivered podium log loss, with
predeclared family/strategy tie order. Baselines remain fixed references; the
best learned model is not claimed best overall if a reference wins.

A requires two earlier 2026 training weekends. With only three 2026 development
weekends, its first two folds are ineligible and the primary A/B/C comparison
contains just one common event. Additional early B/C scores remain separate.
Core features are retained because optional groups failed their development
benefit/coverage/complexity gates. This is weak selection evidence, not a real-data
performance conclusion.

Copy already fitted selected pre-block refits: preprocessing, features, parameters,
weights, calibration and thresholds. Numeric median imputation/missing indicators,
scaling and unknown-safe encoding remain train-only. Podium sigmoid calibration
and F1 thresholds remain development-OOF-only. No final-evaluation fitting occurs.
The immutable lock includes the original manifest, independent entrant/cutoff
inputs, exact model bytes, development comparisons/feature ablations, coverage,
source/code hashes and scoring protocol. Changing code invalidates the lock.

## One-time execution

`final-evaluate` exclusively claims a persistent ledger keyed by manifest ID
before any held-out observation read. It is independent of output/model-lock
paths. A second primary run is rejected; a failed claimed attempt remains
consumed because outcomes may have been exposed. Do not delete/move the ledger
to manufacture another primary run. Local hashes prevent accidental reuse,
not deliberate tampering outside the project.

1. Fix feature history to pre-block development weekends. No earlier block
   outcome enters history, training, weighting or calibration.
2. Seal **all qualifying forecasts** before held-out qualifying outcomes can be
   used as race inputs.
3. Build race/podium inputs after available GP qualifying, using independently
   evidenced grid only. Seal **all race/podium forecasts**.
4. Only then read scoring labels and diagnostic target-session weather.

The source capability permits only current eligible practice, GP qualifying for
race inputs, and pre-block history. It does not modify the frozen manifest or
core feature implementation. Race labels/status, laps, weather, pits and other
block weekends remain forbidden as predictors.

Targets remain official qualifying/race classification, not grid/pure pace.
Preserve ordered retirement/administrative positions and status separately;
missing positions stay missing. Field mismatches/unresolved targets exclude the
affected task/metric transparently. Do not invent a tail or silently drop drivers.
Raw ranking scores remain separate, with stable driver-ID ties and positions
1..N. Independent podium probabilities are never normalized to sum to three;
save order-model and classifier top threes separately, including disagreements.

## Results and uncertainty

Save per-event/equal-event aggregate metrics, paired baseline loss differences,
qualifying/race predicted-versus-actual CSV/Parquet, podium probabilities,
raw/delivered calibration bins and PNG/SVG figures, subgroup errors, coverage and
exclusions. Synthetic row tables explicitly identify data kind and fixture name.

Weather is diagnostic-only after prediction seals. At least three known rain
records are needed for an observed wet/dry tag; missing weather is **unknown**.
Tags describe available samples, not full weather reconstruction. Subgroup means
require three eligible weekends; suppress smaller estimates and retain counts.

Uncertainty uses weekends, not driver rows. For four weekends, report event
ranges, sample SD and leave-one-event-out mean ranges as sensitivity summaries,
**not confidence intervals**. Descriptive event bootstrap intervals are enabled
only at eight or more eligible events, with fixed seed and serial-dependence
caveats. Sparse calibration bins and one common development weekend cannot
establish calibration quality or significance. No accuracy target is promised.

## Commands

The first helper is restricted to independent synthetic fixture inputs; genuine
inputs must come from reviewed sources. Paths are exclusive-create.

```sh
f1ml synthetic-final-inputs --dataset data/phase3-final/dataset \
  --manifest data/phase3-final/manifest.json --output data/phase7-inputs
f1ml final-lock --dataset data/phase3-final/dataset \
  --manifest data/phase3-final/manifest.json --features data/phase3-final/features \
  --qualifying data/phase4-final --race data/phase5-final \
  --experiments data/phase6-verified-experiments \
  --entrants data/phase7-inputs/entrants.csv --cutoffs data/phase7-inputs/cutoffs.csv \
  --output data/phase7-lock
f1ml final-evaluate --lock data/phase7-lock --dataset data/phase3-final/dataset \
  --output data/phase7-final
```

Default ledger: `data/final-evaluation-ledger`. Real source review overlays remain
supported. `--exploratory` labels deliberate test reuse; code/model changes after
consumption require a new exploratory lock. A **fresh future holdout** is required
for confirmation after any test-informed change. No final-score optimization is
performed. See the [results/verification record](final-evaluation-validation.md).
