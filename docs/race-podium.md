# Race classification and podium prediction (Phase 5)

The primary prediction cutoff is **after available official GP qualifying and
before the race starts**. Phase 3's race inputs supply actual qualifying position,
separate Q1/Q2/Q3 comparable pace, shifted prior history and completed practice.
No target-race results, retirement/DNS/DSQ status, laps, weather, pit stops or
Sprint observations enter either model. See [methodology](methodology.md) and
[features](features.md) for availability, revisions and feature formulas.

Qualifying classification and grid remain distinct. `qualifying_position` is
official qualifying order; `known_grid_position` is missing by default. A grid
update requires an independently sourced roster value, evidence and
`grid_known_at <= cutoff`. Unverified/late submitted grid values are rejected;
omit them to retain missing grid inputs. A retrospective race-result grid is
never imported automatically. Neither qualifying nor grid is a race label.

## Official classification and target exclusions

Race regression uses `(official_position - 1)/(official_field_size - 1)`.
It predicts **official classification, not pure pace**. Retain a numeric official
position for a classified retirement or an explicitly ordered DNS/DSQ/non-classified
tail. Actual outcome, original status and classification code remain separate
label-side fields. Missing/invalid/ambiguous positions produce missing rank
targets, never an invented finishing order or an assumed common retirement rank.

Mask unknown training targets but retain the complete feature/entrant field.
Podium labels are one for official positions 1–3, with the existing explicit
DNS/DSQ/excluded/withdrawn/DNQ/non-classified policy assigning zero. An unknown
outcome without an official position has an unknown podium target. Contradictory
positive special-status labels are rejected. A numerically classified retirement
can still have an official podium.

Full-order evaluation requires exactly positions 1..N with matching normalized
targets. An incomplete classification is excluded from primary full-order
comparison and reported in `eligibility.json`; the full source remains in
`evaluation_labels.parquet` and driver-level `target_coverage.parquet`. A known
DNS without a position can exclude race-order evaluation while remaining a known
negative for podium evaluation. Podium comparison requires the complete field's
known binary labels and an unambiguous official podium. Each task has its own
all-strategy common event cohort. No optional practice/pace requirement removes
otherwise eligible drivers or events.

## Models, historical strategies and reference rules

Race: Ridge, random forest regression, histogram gradient-boosted regression.
Podium: logistic regression, random forest classification, histogram
gradient-boosted classification. All use training-fold-only median imputation,
missing indicators, scaling and unknown-safe one-hot encoding. Boosting disables
internal random holdouts. Single-class podium training uses an explicit constant
prior classifier; its status is recorded rather than failing or inventing classes.

The A/B/C strategies and event-balanced sample weights match
[qualifying](qualifying.md): earlier 2026 examples only; equal earlier 2024–2026
weekends; or season/recency weights favoring 2026. A's features may still include
permitted older historical priors. Event weights are divided across labeled
drivers, then all row weights are normalized together so larger fields do not
dominate and temporal weights do not cancel. Parameter grids, seed, eligibility
minimums and postprocessing settings live under `[race]` in the TOML config.

| Fixed reference | Race order | Podium probability |
| --- | --- | --- |
| Qualifying | Current official qualifying normalized rank; missing → 0.5 | Hard qualifying top-three indicator, 0/1; unknown → 0.5 |
| Recent | Shifted race rank mean 3, then mean 5, then previous rank, then qualifying, then 0.5 | Shifted podium rate 5, then rate 3, then qualifying indicator/0.5 |

These deliberately simple rules are untuned. The qualifying probability baseline
is an overconfident reference, not a calibrated forecast. Log loss clips 0/1 to
`[1e-12, 1-1e-12]` for numerical scoring only; stored probabilities stay intact.

## Chronological parameter search, calibration and thresholds

Outer evaluation uses the existing frozen expanding 2026 development folds.
Every driver's rows from a weekend stay together. For each outer training block:

1. Find earlier eligible inner manifest folds wholly inside that block; retain
   at most `max_inner_folds` (default 3).
2. For race models, search parameters/weight profiles over these inner folds by
   equal-event mean position MAE. For podium models, reserve the latest
   `postprocess_folds` (default 2); only earlier inner folds select parameters and
   weights, minimizing raw-probability log loss. Exact ties keep candidate order.
3. Generate predictions for each reserved podium event with a model trained only
   on its earlier weekends and using the already selected parameters. These are
   chronological out-of-fold predictions, never in-sample fitted probabilities.
4. If at least two reserved events and both classes are available, fit fixed-C
   logistic sigmoid calibration on raw log odds, with equal total event weight.
   Log-odds calculation clips to `[1e-6, 1-1e-6]`. Otherwise retain the raw
   probabilities and record why calibration was ineligible.
5. With at least two reserved events, select a decision threshold from the
   predeclared 0.1..0.9 grid by maximum equal-event F1 on those earlier development
   events. Ties prefer closest to 0.5, then the higher threshold. Otherwise use
   the fixed default 0.5 and label it explicitly. Calibration and threshold
   selection share this reserved block; their fitting scores are not evaluation.
6. Refit base preprocessing/estimator on the allowed outer training examples,
   attach calibration/threshold, and evaluate once on the later outer event.

The later outer outcomes do not select their own parameters, calibration or
threshold. With insufficient earlier tuning folds, use the first predeclared
candidate and mark `fixed_default_no_tuning_folds`. Fewer than the configured
training weekends makes a strategy ineligible. Primary model comparisons use
the intersection of events eligible for all three strategies, with equal event
weight. Additional B/C events remain visible separately, never substituted into
the primary average. No common cohort means no selected model for that task and
CLI status 1; neither the minimum nor final block is silently reduced.

Select the best learned race cell by common-event position MAE and the best
learned podium cell by delivered-probability log loss. Ties use predeclared family
then A/B/C order. Repeat the temporal search/postprocessing protocol on development
events and refit selected base models. This remains development selection, not
an unbiased performance estimate. Refitting after chronological OOF calibration
can shift score distributions; few weekends provide weak calibration evidence.
No final-test outcome fits or selects any step. There is no final-test runner.

## Metrics and probability interpretation

Race metrics: full-field position MAE, Spearman rank correlation, winner accuracy,
and predicted top-three overlap with the official podium (`min(3, N)` denominator).
Podium metrics: log loss, Brier score, average precision, precision/recall/F1 at
the earlier-selected threshold, and fixed-bin expected calibration error (ECE).
Reports include raw and delivered log loss/Brier/ECE. Zero predicted positives
gives precision 0; no actual positives gives recall 0 and undefined average
precision. Counts and thresholds accompany event metrics.

`calibration.parquet` gives reliability cells for raw/delivered probabilities on
the common outer development events, with driver count, event-balanced mass,
mean probability, observed podium frequency and pooled ECE. Empty bins remain
empty. Mean per-event ECE and pooled event-balanced ECE are distinct statistics.
Brier/log loss also reflect discrimination; they alone do not establish good
calibration. The implementation was checked against installed scikit-learn 1.9.1
and its official [calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
and [classifier interface](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).

Order regression retains unclipped `raw_score` and sorts ascending by exact score,
then stable driver ID, assigning positions 1..N. These are ranking scores, not
probabilities or precise lap-time forecasts. Podium probabilities are independent
marginals: they **need not sum to three** and are never rescaled. Threshold-positive
drivers need not number three either. In combined output:

- `race_order_top_three` comes from predicted official classification.
- `classifier_top_three` comes from descending podium probability, with stable
  driver-ID ties; `podium_probability_rank` preserves this separate ordering.
- `predicted_podium` is the threshold decision, distinct from both top-three sets.

`top_three_comparison.json` compares corresponding development cells. A combined
prediction receipt lists both sets and flags disagreement; it does not overwrite
one with the other or force probabilities to agree with the race regressor.

## Artifacts, traceability and commands

`race-compare` saves `report.md`, `summary.json`, per-event metrics/predictions,
eligibility, coverage, calibration bins, tuning records, source-label snapshots,
outer-fold model bundles and `selected_models/{race,podium}`. Each learned prediction
records its model receipt ID. Bundles contain preprocessing, estimator,
calibrator, threshold, exact feature schema/code/snapshot IDs, parameters, actual
event weights, training/tuning/calibration/selection event IDs and their latest
information timestamps. Reserved OOF rows include the base model's training IDs
and latest weekend end in `postprocess_training.parquet`. All outputs are hashed.
Load only trusted local pickle/joblib bundles; hashes detect changes but do not
authenticate outside senders.

Prediction saves the actual cutoff features with a receipt linking both model
IDs/metadata, source revision, cutoff/source audit and independent top-three sets.
It rejects a target used for any training, tuning, postprocessing or model
selection; dependency weekends must end before the target weekend starts. Final
test IDs remain sealed. Publication mode additionally requires genuine snapshot
training and exact source/label revisions available by cutoff. Reconstructed
historical results can contain later revisions; those publication-time limitations
from earlier phases still apply.

Fresh **synthetic** workflow (choose fresh output paths when repeating):

```sh
f1ml demo --output data/race-demo
f1ml --config configs/default.toml race-compare \
  --manifest data/race-demo/manifest.json \
  --features data/race-demo/features --output data/race-models
cat data/race-models/report.md
python - <<'PY'
import pandas as pd
rows = pd.read_csv("data/race-demo/entrants.csv", dtype=str)
rows.loc[rows.event_id.eq("2026-03") & rows.task.eq("race")].to_csv(
    "data/race-demo/2026-03-race-entrants.csv", index=False, mode="x"
)
PY
f1ml predict-race \
  --race-model data/race-models/fold_models/race/2026-03/ridge/equal \
  --podium-model data/race-models/fold_models/podium/2026-03/logistic/equal \
  --dataset data/race-demo/dataset --manifest data/race-demo/manifest.json \
  --event 2026-03 --entrants data/race-demo/2026-03-race-entrants.csv \
  --cutoff 2026-03-23T16:00:00Z --output data/race-prediction
```

For a later unseen event after qualifying is available, use the selected bundles
and a reviewed event roster/cutoff. Phase 2 sources accept `--availability` and
`--event-context` reviews. No source race result is needed at inference. Do not
use a later selected refit to claim an earlier historical forecast. The tiny demo
has only one common event at default minima and supplies no real F1 accuracy or
calibration evidence. Existing real-data access/completion/roster review gaps
remain; no fabricated live data, dashboard or deployment was added.
